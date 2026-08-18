"""Krea 2 backend (12B single-stream MMDiT, Qwen3-VL, Qwen-Image VAE).

LoRA-only: a second copy of the 12B transformer will not fit. The frozen
reference is the base model with the adapter disabled.

Krea 2 Turbo is TDM-distilled: 8 steps, CFG off (``guidance=0``),
cond-anchored CFG when guidance is enabled
(``v = cond + g * (cond - uncond)``). Raw (``krea/Krea-2-Raw``, the
default hub id) is the undistilled midtrain checkpoint: ~28 steps,
guidance 4.5.

``model_id`` can also be a local ComfyUI ``.safetensors`` (bf16 or
Kitchen NVFP4). VAE + Qwen3-VL still come from the Raw hub skeleton;
the file replaces only the transformer. Names containing ``turbo``
force the distilled 8-step schedule.

Working latents are *packed* the way the official pipeline keeps them
(seq, C·p·p) so the scheduler and transformer share one layout.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from diffusers import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda
from conceptmod.backends.krea_weights import (
    SKELETON_MODEL,
    load_comfy_krea_transformer,
    looks_turbo,
    resolve_local_transformer,
)

DEFAULT_MODEL = "krea/Krea-2-Raw"
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
# Krea stacks 12 x 2560-d layer taps at seq 512: ~30 MiB bf16 per prompt.
# ++ caches a new random probe almost every step; leaving those on GPU is
# a ~60 MiB/step leak and OOM around step 100-200.
_TEXT_CACHE_MAX = 16


def _load_pipeline(model_id: str) -> Krea2Pipeline:
    local = resolve_local_transformer(model_id)
    if local is None:
        return Krea2Pipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    print(f"krea local transformer: {local}")
    transformer = load_comfy_krea_transformer(local, skeleton=SKELETON_MODEL)
    pipe = Krea2Pipeline.from_pretrained(
        SKELETON_MODEL, transformer=transformer, dtype=torch.bfloat16)
    if looks_turbo(local) and not pipe.config.is_distilled:
        pipe.register_to_config(is_distilled=True)
        print("krea: local file looks like turbo; is_distilled=True")
    return pipe


class KreaBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 768, lora_rank: int | None = None,
                 generate_steps: int | None = None,
                 generate_guidance: float | None = None):
        self.device = str(require_cuda(device))
        self.resolution = resolution
        self.pipe = _load_pipeline(model_id)
        # 12B DiT + 4B Qwen3-VL + VAE do not all fit next to training
        # activations. Only the transformer lives on GPU; encode/generate
        # pull the others over for the call.
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to(self.device)
        self.pipe.transformer.to(self.device)
        print(f"krea transformer+text_encoder on {self.device}; vae parked on cpu")
        self.pipe.set_progress_bar_config(disable=True)

        self.is_distilled = bool(getattr(self.pipe.config, "is_distilled", False))
        if generate_steps is None:
            generate_steps = 8 if self.is_distilled else 28
        if generate_guidance is None:
            generate_guidance = 0.0 if self.is_distilled else 4.5
        self.generate_steps = generate_steps
        self.generate_guidance = generate_guidance

        if lora_rank is None:
            lora_rank = 16
            print("krea backend is LoRA-only; defaulting to rank 16")
        self.lora_rank = lora_rank
        self.compute_dtype = torch.bfloat16
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank,
            target_modules=_LORA_TARGETS,
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
        self.pipe.transformer.to(self.device)
        for p in self.pipe.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.transformer = self.pipe.transformer
        self.transformer.eval()
        base = self.transformer.get_base_model()
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()
        self.frozen = None
        # VAE is only needed for generate / pixel.
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

        self.patch_size = self.pipe.patch_size
        self.vae_scale_factor = self.pipe.vae_scale_factor
        # Spatial VAE latents (C, H, W) before packing. Packed working shape
        # is (seq, C·p·p) = transformer.in_channels on the last dim.
        spatial = resolution // self.vae_scale_factor
        packed = spatial // self.patch_size
        self.spatial_hw = (spatial, spatial)
        self.grid_hw = (packed, packed)
        self.latent_channels = base.config.in_channels // (self.patch_size ** 2)
        self.latent_shape = (packed * packed, base.config.in_channels)
        self._text_cache: OrderedDict[tuple[str, bool], TextEmbeds] = OrderedDict()
        self.encoder_lora = False
        self.max_sequence_length = 512

    def training_defaults(self) -> dict:
        if self.is_distilled:
            return {"sample_steps": 8, "sample_guidance": 0.0}
        return {"sample_steps": 14, "sample_guidance": 4.5}

    # ---------------- text ----------------

    def _park_text_encoder(self):
        """Drop the 4B Qwen3-VL off GPU before the 12B DiT backward."""
        if self.encoder_lora:
            return
        if next(self.pipe.text_encoder.parameters()).device.type == "cuda":
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        self.pipe.text_encoder.to(self.device)
        embeds, mask = self.pipe.get_text_hidden_states(
            prompt, self.max_sequence_length, self.device,
        )
        return TextEmbeds(embeds, mask)

    def _remember_text(self, key: tuple[str, bool], text: TextEmbeds) -> None:
        self._text_cache[key] = TextEmbeds(
            text.embeds.detach().to("cpu"),
            None if text.mask is None else text.mask.detach().to("cpu"),
        )
        self._text_cache.move_to_end(key)
        while len(self._text_cache) > _TEXT_CACHE_MAX:
            self._text_cache.popitem(last=False)

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        key = (prompt, frozen)
        if key not in self._text_cache:
            if self.encoder_lora and frozen:
                with self.pipe.text_encoder.disable_adapter():
                    raw = self._encode_raw(prompt)
            else:
                raw = self._encode_raw(prompt)
            self._remember_text(key, raw)
        else:
            self._text_cache.move_to_end(key)
        text = self._text_cache[key]
        return TextEmbeds(
            text.embeds.to(self.device),
            None if text.mask is None else text.mask.to(self.device),
        )

    def encode_text_grad(self, prompt: str) -> TextEmbeds:
        return self._encode_raw(prompt)

    def attach_encoder_lora(self, rank: int = 8):
        from peft import LoraConfig, get_peft_model

        assert not self.encoder_lora, "encoder LoRA already attached"
        config = LoraConfig(
            r=rank, lora_alpha=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, config)
        self.pipe.text_encoder.to(self.device)
        for n, p in self.pipe.text_encoder.named_parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.pipe.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _position_ids(self, text_seq_len: int):
        gh, gw = self.grid_hw
        return self.pipe.prepare_position_ids(text_seq_len, gh, gw, self.device)

    def _forward(self, model, z, timestep, text: TextEmbeds):
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        pos = self._position_ids(text.embeds.shape[1])
        mask = None if text.mask is None else text.mask.to(self.device)
        out = model(
            hidden_states=z.to(device=self.device, dtype=dtype),
            encoder_hidden_states=text.embeds.to(device=self.device, dtype=dtype),
            timestep=t,
            position_ids=pos,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        return out.float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self.transformer.disable_adapter():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    # ---------------- sampling ----------------

    def _fresh_scheduler(self, num_steps):
        import numpy as np

        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = self.latent_shape[0]
        if self.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                sched.config.get("base_image_seq_len", 256),
                sched.config.get("max_image_seq_len", 6400),
                sched.config.get("base_shift", 0.5),
                sched.config.get("max_shift", 1.15),
            )
        sched.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        if hasattr(sched, "set_begin_index"):
            sched.set_begin_index(0)
        return sched

    def _cfg(self, prompt, z, t, guidance, frozen):
        # Krea convention: cond + g*(cond - uncond), enabled when g > 0.
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance > 0 and prompt != "":
            v_u = self.predict_v("", z, t, frozen=frozen)
            v = v + guidance * (v - v_u)
        return v

    def _noise(self, generator):
        return torch.randn(
            (1, *self.latent_shape), generator=generator,
            device=self.device, dtype=torch.float32,
        )

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        sched = self._fresh_scheduler(num_steps)
        z = self._noise(generator)
        for i, t in enumerate(sched.timesteps):
            if i >= stop_index:
                return z, t
            v = self._cfg(prompt, z, t, guidance, frozen=False)
            z = sched.step(v, t, z, return_dict=False)[0]
        return z, sched.timesteps[-1]

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        sched = self._fresh_scheduler(num_steps)
        z = self._noise(generator)
        n = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self._cfg(prompt, z, t, guidance, frozen=frozen)
                z = sched.step(v, t, z, return_dict=False)[0]
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        self.pipe.vae.to(self.device)
        vae = self.pipe.vae
        with torch.set_grad_enabled(grad):
            latents = self.pipe._unpack_latents(
                z, self.resolution, self.resolution).to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean, device=latents.device,
                                dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device,
                                     dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            latents = latents / std + mean
            img = vae.decode(latents, return_dict=False)[0][:, :, 0]
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from PIL import Image

        num_steps = num_steps or self.generate_steps
        guidance = self.generate_guidance if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        # Encode on GPU, then park the 4B encoder so denoise + VAE decode
        # do not sit next to it. Official pipe() keeps all three resident.
        try:
            img = self.render(prompt, g, num_steps, guidance, grad_steps=0,
                              frozen=frozen)
        finally:
            self.pipe.vae.to("cpu")
            self._park_text_encoder()
            torch.cuda.empty_cache()
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        assert params
        if train_method not in (None, "lora"):
            print(f"krea backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
