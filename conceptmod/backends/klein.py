"""Flux.2 Klein backend (4B/9B MMDiT, Qwen3, Flux2 VAE).

LoRA-only: a second copy of the transformer will not fit. The frozen
reference is the base model with the adapter disabled.

Product story matches Krea Raw/Turbo: train on **Base** (undistilled),
intended to run on Distilled.

Default hub id is the official 4B Base checkpoint
``black-forest-labs/FLUX.2-klein-base-4B`` (Apache 2.0, verified on the
Hub). 9B Base is optional via ``--model-id 9b-base`` /
``black-forest-labs/FLUX.2-klein-base-9B`` (gated, non-commercial).

Base generate: 50 steps, CFG 4.0. Distilled (``is_distilled`` or a
non-``base`` Klein id): 4 steps, CFG off. Official CFG is
``uncond + g*(cond - uncond)``.

Working latents are packed ``(seq, C)`` the way Flux2KleinPipeline keeps
them. Full train is not a VM smoke: use ``scripts/smoke_klein.py`` on a
GPU box.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
KLEIN_9B_BASE = "black-forest-labs/FLUX.2-klein-base-9B"
ALIASES = {
    "4b-base": DEFAULT_MODEL,
    "4b": "black-forest-labs/FLUX.2-klein-4B",
    "9b-base": KLEIN_9B_BASE,
    "9b": "black-forest-labs/FLUX.2-klein-9B",
}
# Double-stream attn (separate QKV) + single-stream fused QKV+MLP.
_LORA_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "to_qkv_mlp_proj", "to_out",
]
_TEXT_CACHE_MAX = 16


def resolve_model_id(model_id: str) -> str:
    """Expand ``4b-base`` / ``9b-base`` aliases; pass Hub ids through."""
    return ALIASES.get(model_id.lower(), model_id)


def looks_distilled(model_id: str) -> bool:
    """Official distilled ids omit ``base`` (``FLUX.2-klein-4B`` vs ``-base-4B``)."""
    name = resolve_model_id(model_id).rsplit("/", 1)[-1].lower()
    return "klein" in name and "base" not in name


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    try:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu)
        return float(compute_empirical_mu(image_seq_len, num_steps))
    except Exception:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        return float(a * num_steps + (m_200 - 200.0 * a))


class KleinBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 512, lora_rank: int | None = None,
                 generate_steps: int | None = None,
                 generate_guidance: float | None = None):
        self.device = str(require_cuda(device))
        self.resolution = resolution
        model_id = resolve_model_id(model_id)
        self.model_id = model_id
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16)
        # 4B DiT + ~4B Qwen3 + VAE: park the VAE; encoder is staged.
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to(self.device)
        self.pipe.transformer.to(self.device)
        print(f"klein transformer+text_encoder on {self.device}; vae parked on cpu")
        self.pipe.set_progress_bar_config(disable=True)

        self.is_distilled = bool(getattr(self.pipe.config, "is_distilled", False))
        if not self.is_distilled and looks_distilled(model_id):
            self.is_distilled = True
            print("klein: model id looks distilled; is_distilled=True")
        if generate_steps is None:
            generate_steps = 4 if self.is_distilled else 50
        if generate_guidance is None:
            generate_guidance = 0.0 if self.is_distilled else 4.0
        self.generate_steps = generate_steps
        self.generate_guidance = generate_guidance

        if lora_rank is None:
            lora_rank = 16
            print("klein backend is LoRA-only; defaulting to rank 16")
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
        if self.is_distilled:
            return {"sample_steps": 4, "sample_guidance": 0.0}
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
        embeds, _text_ids = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
        )
        return TextEmbeds(embeds, None)

    def _remember_text(self, key: tuple[str, bool], text: TextEmbeds) -> None:
        self._text_cache[key] = TextEmbeds(
            text.embeds.detach().to("cpu"), None)
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
        return TextEmbeds(self._text_cache[key].embeds.to(self.device), None)

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

    def _img_ids(self, batch: int):
        ph, pw = self.grid_hw
        dummy = torch.zeros(
            1, self.latent_channels, ph, pw, device=self.device)
        ids = self.pipe._prepare_latent_ids(dummy).to(self.device)
        if batch > 1:
            ids = ids.expand(batch, -1, -1)
        return ids

    def _forward(self, model, z, timestep, text: TextEmbeds):
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        txt_ids = self.pipe._prepare_text_ids(text.embeds).to(self.device)
        img_ids = self._img_ids(z.shape[0])
        out = model(
            hidden_states=z.to(device=self.device, dtype=dtype),
            encoder_hidden_states=text.embeds.to(device=self.device, dtype=dtype),
            timestep=t,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=None,
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
        mu = _compute_empirical_mu(self.latent_shape[0], num_steps)
        sched.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        if hasattr(sched, "set_begin_index"):
            sched.set_begin_index(0)
        return sched

    def _cfg(self, prompt, z, t, guidance, frozen):
        # Official Klein CFG: uncond + g*(cond - uncond); distilled ignores g.
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if (not self.is_distilled) and guidance and guidance > 1.0 and prompt != "":
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
        ph, pw = self.grid_hw
        with torch.set_grad_enabled(grad):
            ids = self._img_ids(z.shape[0])
            latents = self.pipe._unpack_latents_with_ids(z, ids, ph, pw)
            latents = latents.to(vae.dtype)
            mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * std + mean
            latents = self.pipe._unpatchify_latents(latents)
            img = vae.decode(latents, return_dict=False)[0]
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
            print(f"klein backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
