"""Anima backend (2B Cosmos-Predict2 DiT, Qwen3 + T5 conditioner, Qwen-Image VAE).

Anima ships as a ModularPipeline. We load that to get the components, then
drive encode / velocity / sampling ourselves so the DSL can train.

LoRA-only: same reason as Z-Image — a second fp32 copy is wasteful, and
CircleStone tells you not to train the LLM adapter (``text_conditioner``).
The frozen reference is the base transformer with the adapter disabled.

Official Base (this default) is a regular CFG flow-matching model; Turbo is
CFG-distilled. Pass ``model_id`` to switch.
"""

from __future__ import annotations

import torch
from diffusers import ModularPipeline

from conceptmod.backends.base import Backend, TextEmbeds, pin_modules, require_cuda

DEFAULT_MODEL = "circlestone-labs/Anima-Base-v1.0-Diffusers"
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
# CircleStone defaults: 30-50 steps, CFG 4-5, plus their recommended negative.
DEFAULT_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, "
    "artist name, blurry, jpeg artifacts, chromatic aberration, "
    "nsfw, explicit, nude, nipples"
)


class AnimaBackend(Backend):
    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 768, lora_rank: int | None = None,
                 generate_steps: int = 40, generate_guidance: float = 4.0):
        self.device = str(require_cuda(device))
        self.resolution = resolution
        self.generate_steps = generate_steps
        self.generate_guidance = generate_guidance
        self.default_negative = DEFAULT_NEGATIVE
        self.pipe = ModularPipeline.from_pretrained(model_id)
        self.pipe.load_components(dtype=torch.bfloat16)
        moved = pin_modules(self.pipe, self.device)
        print(f"anima modules on {self.device}: {', '.join(moved)}")
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

        if lora_rank is None:
            lora_rank = 16
            print("anima backend is LoRA-only; defaulting to rank 16")
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

        self.latent_channels = base.config.in_channels
        scale = self.pipe.vae_scale_factor
        # Cosmos image latents keep a singleton time axis.
        h = resolution // scale
        self.latent_shape = (self.latent_channels, 1, h, h)
        # Official Cosmo mask is pixel-sized zeros; the transformer resizes it.
        self._padding_mask = torch.zeros(
            1, 1, resolution, resolution, device=self.device, dtype=torch.bfloat16,
        )
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}
        self.encoder_lora = False
        self.max_sequence_length = 512

    def training_defaults(self) -> dict:
        return {"sample_steps": 14, "sample_guidance": 4.5}

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        prompts = [prompt]
        device = self.device
        cond_dtype = self.pipe.text_conditioner.dtype
        tok = self.pipe.tokenizer(
            prompts, padding="longest", max_length=self.max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        ids = tok.input_ids.to(device)
        mask = tok.attention_mask.to(device)
        if ids.shape[-1] == 0:
            ids = ids.new_zeros((ids.shape[0], 1))
            mask = mask.new_zeros((mask.shape[0], 1))
        qwen = self.pipe.text_encoder(
            input_ids=ids, attention_mask=mask, output_hidden_states=False,
        ).last_hidden_state
        qwen = qwen * mask.to(qwen.dtype).unsqueeze(-1)

        t5 = self.pipe.t5_tokenizer(
            prompts, padding="longest", max_length=self.max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        cond = self.pipe.text_conditioner(
            source_hidden_states=qwen.to(device=device, dtype=cond_dtype),
            target_input_ids=t5.input_ids.to(device),
            target_attention_mask=t5.attention_mask.to(device),
            source_attention_mask=mask,
        )
        return TextEmbeds(cond.float().to(device), None)

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
        # CircleStone: do not train the LLM adapter (text_conditioner).
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

    def _forward(self, model, z, timestep, text: TextEmbeds):
        # LoRA params are fp32; the DiT must stay bf16. Using
        # next(parameters()).dtype here would promote the whole forward to fp32.
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        hidden = z.to(device=self.device, dtype=dtype)
        if hidden.ndim == 4:
            hidden = hidden.unsqueeze(2)
        out = model(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=text.embeds.to(device=self.device, dtype=dtype),
            padding_mask=self._padding_mask.to(device=self.device, dtype=dtype),
            return_dict=False,
        )[0]
        # Keep the singleton time axis so scheduler.step sees the same
        # (B, C, 1, H, W) layout as the official Cosmo loop. Squeezing it
        # made Euler broadcast a 4D velocity against 5D latents and smear.
        if out.ndim == 4:
            out = out.unsqueeze(2)
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
        sched.set_timesteps(sigmas=sigmas, device=self.device)
        if hasattr(sched, "set_begin_index"):
            sched.set_begin_index(0)
        return sched

    def _cfg(self, prompt, z, t, guidance, frozen, negative=""):
        # Anima's guider is standard CFG (uncond + g*(cond-uncond)); g=1 is off.
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance != 1.0 and prompt != negative:
            v_u = self.predict_v(negative, z, t, frozen=frozen)
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
               frozen=False, negative=""):
        sched = self._fresh_scheduler(num_steps)
        z = self._noise(generator)
        n = len(sched.timesteps)
        for i, t in enumerate(sched.timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self._cfg(prompt, z, t, guidance, frozen=frozen,
                              negative=negative)
                z = sched.step(v, t, z, return_dict=False)[0]
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        vae = self.pipe.vae
        with torch.set_grad_enabled(grad):
            latents = z.to(vae.dtype)
            if latents.ndim == 4:
                latents = latents.unsqueeze(2)
            mean = torch.tensor(vae.config.latents_mean, device=latents.device,
                                dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device,
                                     dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
            latents = latents / std + mean
            img = vae.decode(latents, return_dict=False)[0][:, :, 0]
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from contextlib import nullcontext

        num_steps = num_steps or self.generate_steps
        guidance = self.generate_guidance if guidance is None else guidance
        if hasattr(self.pipe, "guider") and self.pipe.guider is not None:
            self.pipe.guider.guidance_scale = guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        ctx = self.transformer.disable_adapter() if frozen else nullcontext()
        with ctx:
            images = self.pipe(
                prompt=prompt,
                negative_prompt=self.default_negative,
                height=self.resolution,
                width=self.resolution,
                num_inference_steps=num_steps,
                generator=g,
                output="images",
            )
        if isinstance(images, (list, tuple)):
            return images[0]
        return images.images[0]

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        assert params
        if train_method not in (None, "lora"):
            print(f"anima backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
