"""Qwen-Image backend (20B MMDiT, Qwen2.5-VL, Qwen-Image VAE).

LoRA-only: a second copy of the transformer will not fit. The frozen
reference is the base model with the adapter disabled.

Qwen-Image-Edit / Edit-2509 use the same ``QwenImageTransformer2DModel``
PEFT layout (``transformer_blocks.N.attn.{to_q,to_k,to_v,to_out.0}`` plus
optional dual-stream ``add_*_proj``). Convert treats them as ``qwen``.
This backend trains text-to-image; pass ``Qwen/Qwen-Image`` (default).
Edit checkpoints load the transformer the same way when ``model_id``
contains ``edit``, but ``generate`` stays T2I (no reference image).

Working latents are packed the way the official pipeline keeps them
(seq, C·p·p). CFG is standard ``uncond + g*(cond - uncond)`` via
``true_cfg_scale`` (default 4.0, 50 sample steps).

Full train of the 20B DiT is not a VM smoke: use
``scripts/smoke_qwen.py`` on a GPU box with the hub checkpoint, or the
convert / mapper unit tests which do not load weights.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda

DEFAULT_MODEL = "Qwen/Qwen-Image"
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
_TEXT_CACHE_MAX = 16


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


def _load_pipeline(model_id: str):
    """Load T2I, or the Edit pipeline's shared transformer stack."""
    if "edit" in model_id.lower():
        from diffusers import QwenImageEditPipeline

        print(f"qwen: loading Edit pipeline {model_id} (T2I train path)")
        return QwenImageEditPipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    from diffusers import QwenImagePipeline

    return QwenImagePipeline.from_pretrained(model_id, dtype=torch.bfloat16)


class QwenBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 512, lora_rank: int | None = None,
                 generate_steps: int = 50, generate_guidance: float = 4.0):
        self.device = str(require_cuda(device))
        self.resolution = resolution
        self.generate_steps = generate_steps
        self.generate_guidance = generate_guidance
        self.pipe = _load_pipeline(model_id)
        # 20B DiT + 7B VL + VAE do not all fit next to training activations.
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to(self.device)
        self.pipe.transformer.to(self.device)
        print(f"qwen transformer+text_encoder on {self.device}; vae parked on cpu")
        self.pipe.set_progress_bar_config(disable=True)

        if lora_rank is None:
            lora_rank = 16
            print("qwen backend is LoRA-only; defaulting to rank 16")
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
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

        self.vae_scale_factor = self.pipe.vae_scale_factor
        spatial = 2 * (resolution // (self.vae_scale_factor * 2))
        packed = spatial // 2
        self.spatial_hw = (spatial, spatial)
        self.grid_hw = (packed, packed)
        self.latent_channels = base.config.in_channels // 4
        self.latent_shape = (packed * packed, base.config.in_channels)
        self._text_cache: OrderedDict[tuple[str, bool], TextEmbeds] = OrderedDict()
        self.encoder_lora = False
        self.max_sequence_length = 512

    def training_defaults(self) -> dict:
        return {"sample_steps": 14, "sample_guidance": 4.0}

    # ---------------- text ----------------

    def _park_text_encoder(self):
        if self.encoder_lora:
            return
        if next(self.pipe.text_encoder.parameters()).device.type == "cuda":
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        self.pipe.text_encoder.to(self.device)
        embeds, mask = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
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
        for _n, p in self.pipe.text_encoder.named_parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.pipe.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _img_shapes(self):
        gh, gw = self.grid_hw
        return [[(1, gh, gw)]]

    def _forward(self, model, z, timestep, text: TextEmbeds):
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        mask = None if text.mask is None else text.mask.to(self.device)
        out = model(
            hidden_states=z.to(device=self.device, dtype=dtype),
            encoder_hidden_states=text.embeds.to(device=self.device, dtype=dtype),
            timestep=t,
            encoder_hidden_states_mask=mask,
            img_shapes=self._img_shapes(),
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
        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        image_seq_len = self.latent_shape[0]
        mu = _calculate_shift(
            image_seq_len,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sched.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        if hasattr(sched, "set_begin_index"):
            sched.set_begin_index(0)
        return sched

    def _cfg(self, prompt, z, t, guidance, frozen):
        # Official Qwen true_cfg: uncond + g*(cond - uncond), g>1 enables it.
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance > 1.0 and prompt != "":
            v_u = self.predict_v("", z, t, frozen=frozen)
            v = v_u + guidance * (v - v_u)
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
                z, self.resolution, self.resolution, self.vae_scale_factor,
            ).to(vae.dtype)
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
            print(f"qwen backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
