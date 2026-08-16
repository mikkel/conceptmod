"""Z-Image Turbo backend (6B flow-matching S3-DiT, Qwen text encoder).

LoRA-only: a full fp32 second copy of the 6B transformer is not worth the
VRAM; the frozen reference is the base model with the adapter disabled.

API quirks vs Sana: prompt embeddings are *unpadded variable-length lists*
(one tensor per sample, chat-template encoded, hidden_states[-2]); the
transformer takes lists of (C, 1, H, W) latents; its time input is the
inverted normalized timestep (1000 - t) / 1000; CFG is
``pos + g * (pos - neg)``.
"""

from __future__ import annotations

import torch
from diffusers import ZImagePipeline

from conceptmod.backends.base import Backend, TextEmbeds

DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"


class ZImageBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 768, lora_rank: int | None = None,
                 generate_steps: int = 9, generate_guidance: float = 0.0):
        self.device = device
        self.resolution = resolution
        self.generate_steps = generate_steps
        self.generate_guidance = generate_guidance
        self.pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        if lora_rank is None:
            lora_rank = 16
            print("zimage backend is LoRA-only; defaulting to rank 16")
        self.lora_rank = lora_rank
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
        for p in self.pipe.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.transformer = self.pipe.transformer
        # 6B backward at bf16 needs activation checkpointing to fit 48GB
        self.transformer.get_base_model().enable_gradient_checkpointing()
        self.frozen = None

        base_cfg = self.transformer.get_base_model().config
        self.latent_channels = base_cfg.in_channels
        h = 2 * (resolution // (self.pipe.vae_scale_factor * 2))
        self.latent_shape = (self.latent_channels, h, h)
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}
        self.encoder_lora = False

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        embeds_list = self.pipe._encode_prompt(prompt, device=self.device)
        return TextEmbeds(embeds_list, None)  # unpadded list, no mask

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        key = (prompt, frozen)
        if key not in self._text_cache:
            if self.encoder_lora and frozen:
                with self.pipe.text_encoder.disable_adapter():
                    self._text_cache[key] = self._encode_raw(prompt)
            else:
                self._text_cache[key] = self._encode_raw(prompt)
        return self._text_cache[key]

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
        for p in self.pipe.text_encoder.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.pipe.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _forward(self, model, z, timestep, text: TextEmbeds):
        dtype = next(model.parameters()).dtype
        t = timestep.expand(z.shape[0]).to(self.device).float()
        t_model = (1000.0 - t) / 1000.0
        latents = list(z.to(dtype).unsqueeze(2).unbind(dim=0))
        embeds = [e.to(dtype) for e in text.embeds]
        out = model(latents, t_model, embeds, return_dict=False)[0]
        # Z-Image predicts the negated flow velocity; flip so our v matches
        # the scheduler's convention (same as the pipeline's `-noise_pred`)
        return -torch.stack([o.squeeze(1) for o in out], dim=0).float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self.transformer.disable_adapter():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    # ---------------- sampling ----------------

    def _fresh_scheduler(self, num_steps):
        from diffusers.pipelines.z_image.pipeline_z_image import (
            calculate_shift, get_default_z_image_sigmas)

        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        image_seq_len = (self.latent_shape[1] // 2) * (self.latent_shape[2] // 2)
        mu = calculate_shift(
            image_seq_len,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sigmas = get_default_z_image_sigmas(num_steps)
        sched.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        sched.set_begin_index(0)
        return sched

    def _cfg(self, prompt, z, t, guidance, frozen):
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance > 0 and prompt != "":
            v_u = self.predict_v("", z, t, frozen=frozen)
            v = v + guidance * (v - v_u)
        return v

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        sched = self._fresh_scheduler(num_steps)
        z = torch.randn((1, *self.latent_shape), generator=generator,
                        device=self.device, dtype=torch.float32)
        for i, t in enumerate(sched.timesteps):
            if i >= stop_index:
                return z, t
            v = self._cfg(prompt, z, t, guidance, frozen=False)
            z = sched.step(v, t, z, return_dict=False)[0]
        return z, sched.timesteps[-1]

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        sched = self._fresh_scheduler(num_steps)
        z = torch.randn((1, *self.latent_shape), generator=generator,
                        device=self.device, dtype=torch.float32)
        n = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self._cfg(prompt, z, t, guidance, frozen=frozen)
                z = sched.step(v, t, z, return_dict=False)[0]
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        vae = self.pipe.vae
        with torch.set_grad_enabled(grad):
            z = z.to(vae.dtype)
            z = z / vae.config.scaling_factor + vae.config.shift_factor
            img = vae.decode(z, return_dict=False)[0]
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from PIL import Image

        num_steps = num_steps or self.generate_steps
        guidance = self.generate_guidance if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        img = self.render(prompt, g, num_steps, guidance, grad_steps=0,
                          frozen=frozen)
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        assert params
        if train_method not in (None, "lora"):
            print(f"zimage backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
