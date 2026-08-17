"""SANA backend (0.6B flow-matching linear DiT, Gemma-2 text encoder, DC-AE).

The trained transformer is kept in fp32 (bf16 master weights + Adam is
unstable); the frozen reference copy stays in bf16. Latents are tiny
(32x16x16 at 512px) so fp32 forward passes are cheap.
"""

from __future__ import annotations

import copy

import torch
from diffusers import SanaPipeline

from conceptmod.backends.base import Backend, TextEmbeds, pin_modules, require_cuda

DEFAULT_MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"


class SanaBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 512, lora_rank: int | None = None):
        self.device = str(require_cuda(device))
        self.resolution = resolution
        self.pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pin_modules(self.pipe, self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.lora_rank = lora_rank
        if lora_rank is None:
            # frozen copy in bf16, trained copy promoted to fp32
            self.frozen = copy.deepcopy(self.pipe.transformer)
            self.pipe.transformer.to(torch.float32)
        else:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
            self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
            self.pipe.transformer.to(torch.float32)
            self.frozen = None  # frozen = adapter disabled
        self.transformer = self.pipe.transformer
        if self.frozen is not None:
            self.frozen.requires_grad_(False)
            self.frozen.eval()

        cfg = self._base_config()
        self.latent_channels = cfg.in_channels
        scale = self.pipe.vae_scale_factor * cfg.patch_size
        self.latent_shape = (cfg.in_channels, resolution // scale, resolution // scale)
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}
        self.encoder_lora = False

    def _base_config(self):
        t = self.transformer
        return t.get_base_model().config if self.lora_rank is not None else t.config

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        embeds, mask, _, _ = self.pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device,
            clean_caption=False,
            complex_human_instruction=None,
        )
        return TextEmbeds(embeds, mask)

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        """Cached prompt encoding. Once an encoder LoRA is attached,
        frozen=True encodes with the adapter disabled (original encoder)."""
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
        """Uncached encoding with gradients through the encoder LoRA."""
        return self._encode_raw(prompt)

    def attach_encoder_lora(self, rank: int = 8):
        from peft import LoraConfig, get_peft_model

        assert not self.encoder_lora, "encoder LoRA already attached"
        config = LoraConfig(
            r=rank, lora_alpha=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, config)
        # LoRA weights in fp32 for stable training; base stays bf16
        for n, p in self.pipe.text_encoder.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.pipe.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _forward(self, model, z, timestep, text: TextEmbeds):
        dtype = next(model.parameters()).dtype
        t = timestep.expand(z.shape[0]).to(self.device)
        t = t * self._base_config().timestep_scale
        v = model(
            z.to(dtype),
            encoder_hidden_states=text.embeds.to(dtype),
            encoder_attention_mask=text.mask,
            timestep=t,
            return_dict=False,
        )[0]
        if self._base_config().out_channels // 2 == self.latent_channels:
            v = v.chunk(2, dim=1)[0]
        return v.float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            if self.frozen is not None:
                with torch.no_grad():
                    return self._forward(self.frozen, z, timestep, text)
            with torch.no_grad(), self._adapter_disabled():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    def _adapter_disabled(self):
        return self.transformer.disable_adapter()

    # ---------------- sampling ----------------

    def _fresh_scheduler(self, num_steps):
        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        sched.set_timesteps(num_steps, device=self.device)
        return sched

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        sched = self._fresh_scheduler(num_steps)
        z = torch.randn((1, *self.latent_shape), generator=generator,
                        device=self.device, dtype=torch.float32)
        for i, t in enumerate(sched.timesteps):
            if i >= stop_index:
                return z, t
            v = self.predict_v(prompt, z, t, frozen=False)
            if guidance != 1.0 and prompt != "":
                v_u = self.predict_v("", z, t, frozen=False)
                v = v_u + guidance * (v - v_u)
            z = sched.step(v, t, z, return_dict=False)[0]
        return z, sched.timesteps[-1]

    def render(self, prompt: str, generator: torch.Generator, num_steps: int,
               guidance: float, grad_steps: int = 0, frozen: bool = False):
        """Full denoise -> decoded image tensor in [-1, 1]. Gradients flow
        through the last ``grad_steps`` Euler steps and the VAE decode."""
        sched = self._fresh_scheduler(num_steps)
        z = torch.randn((1, *self.latent_shape), generator=generator,
                        device=self.device, dtype=torch.float32)
        n = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self.predict_v(prompt, z, t, frozen=frozen)
                if guidance != 1.0 and prompt != "":
                    v_u = self.predict_v("" , z, t, frozen=frozen)
                    v = v_u + guidance * (v - v_u)
                z = sched.step(v, t, z, return_dict=False)[0]
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        vae = self.pipe.vae
        with torch.set_grad_enabled(grad):
            img = vae.decode(z.to(vae.dtype) / vae.config.scaling_factor,
                             return_dict=False)[0]
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from PIL import Image
        import numpy as np

        num_steps = num_steps or 20
        guidance = 4.5 if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        img = self.render(prompt, g, num_steps, guidance, grad_steps=0, frozen=frozen)
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        model = self.transformer
        model.train()
        if self.lora_rank is not None:
            params = [p for p in model.parameters() if p.requires_grad]
            assert params, "peft returned no trainable params"
            return params
        model.requires_grad_(False)
        params = []
        for name, param in model.named_parameters():
            keep = False
            if train_method == "xattn":
                keep = "attn2" in name
            elif train_method == "selfattn":
                keep = "attn1" in name
            elif train_method == "attn":
                keep = "attn1" in name or "attn2" in name
            elif train_method == "full":
                keep = True
            elif train_method == "noxattn":
                keep = not ("attn2" in name or "time_embed" in name
                            or name.startswith("proj_out"))
            else:
                raise ValueError(f"unknown train_method {train_method!r}")
            if keep:
                param.requires_grad_(True)
                params.append(param)
        assert params, f"train_method {train_method!r} selected no parameters"
        return params

    def save_trained(self, path: str):
        if self.lora_rank is not None:
            self.transformer.save_pretrained(path)
        else:
            from safetensors.torch import save_file

            sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
                  for k, v in self.transformer.state_dict().items()}
            save_file(sd, path)
