"""SenseNova-U1.5-8B-MoT backend (NEO-unify, 17.5B, pixel-space flow matching).

This is not a diffusers pipeline. ``NEOChatModel`` is a ``transformers``
any-to-any LLM: one Qwen3-8B token stream where every one of the 42 decoder
layers carries a *second* parallel set of attention+MLP weights suffixed
``_mot_gen`` (Mixture-of-Transformers). Text tokens route through the base
weights, image tokens through the ``_mot_gen`` weights, and the two streams
meet in one shared joint self-attention. So the "text encoder" and the
"denoiser" are the same module, told apart only by the weight suffix.

Consequences for the conceptmod contract:

* **No VAE.** Flow matching runs directly on pixels. ``decode`` is
  ``unpatchify`` and nothing else. Working latents are the patch-32 packed
  image, ``(tokens, 3*32*32)``, exactly the ``z`` the official sampler steps.
* **No diffusers scheduler.** The sampler is hand-rolled on the model:
  ``_apply_time_schedule`` for the resolution-aware shift and a plain Euler
  step. Timesteps run **0 -> 1 with 0 = pure noise** (the opposite of the
  diffusers sigma convention) and the step is ``z + (t_next - t) * v``.
* **Velocity is not O(1) across t.** The head predicts the clean image and
  the wrapper divides: ``v = (x_pred - z) / (1 - t)``. ``|v|`` therefore grows
  several-fold toward the image end of the schedule, which an unweighted
  velocity MSE reads as "these timesteps matter tens of times more". The
  backend reports the divisor through ``velocity_loss_scale`` so ``ops``
  compares ``x_pred`` instead; without it the LoRA learns a global,
  prompt-independent DC offset (a relight) rather than a concept edit.
* **Noise is not unit variance.** The official sampler seeds with
  ``noise_scale * randn``, where ``noise_scale`` grows as
  ``sqrt(tokens / 64)``. Seeding at std 1 lands off-distribution and renders
  mush, so ``_noise`` reproduces the scaling.
* **Text conditioning is a KV cache**, not an embedding tensor. Each prompt
  is templated into ChatML, run through the base (understanding) branch once,
  and the resulting ``past_key_values`` prefix is what the image tokens
  attend to. ``encode_text`` caches that prefix; ``TextEmbeds.embeds`` carries
  the prefix hidden states so the encoder-stage pooling helpers still see a
  ``(B, L, D)`` tensor.
* **CFG is uncond-anchored** (``v_u + g * (v - v_u)``, off at ``g <= 1``),
  matching ``t2i_generate``. Same family as sana/anima, not krea/zimage. Note
  what that expands to: ``g * v_c - (g - 1) * v_u``, so drift in the negative
  reaches every rendered prompt at 3x the default guidance. ``#`` anchors it
  through :meth:`SenseNovaBackend.cfg_negative_prompt`.
* **The DSL loss is not flat across ``t``** even after ``velocity_loss_scale``,
  because a prompt can barely move the prediction once the image is nearly
  resolved. 96% of the ``=`` loss sits in the first three of the 16 training
  timesteps, so the backend declares ``accumulation_steps`` -- see
  ``training_defaults``.
* **The CFG negative is not the empty prompt.** ``t2i_generate`` builds its
  conditional from ``_build_t2i_query(prompt, system_message=
  SYSTEM_MESSAGE_FOR_GEN, append_text=<think></think><img>)`` for *any*
  ``prompt``, ``''`` included (251 tokens), and separately builds the
  unconditional as a bare ``_build_t2i_query('', append_text=<img>)`` with no
  system message and no think block (9 tokens). Those are two different
  things, and this backend keeps them apart: ``encode_text('')`` is the
  empty-text *conditional*, and the 9-token negative lives behind
  ``CFG_UNCOND`` where only ``_cfg`` reaches it. Conflating them made every
  DSL direction ``v(p) - v('')`` a 9-vs-251-token *template* difference
  stacked on the concept difference -- measured on the real checkpoint,
  ``v(empty-conditional) - v(uncond)`` has std 0.011-0.025 against a
  human->robot direction of std 0.022-0.040, i.e. the contaminant is the same
  size as the signal and points elsewhere (cos -0.12 to -0.20). It also made
  ``#`` (``FREEZE('', '')``) anchor a 9-token prompt that image generation
  never visits, which is why it logged 0.0000 all run while every
  conditional prompt drifted together.

LoRA-only, on the generation branch only, the way the official 8-step LoRA
does it: a second copy of 17.5B is not remotely affordable, and the frozen
reference is the same weights with the adapter disabled.

Do **not** call ``prepare_flash_kv_cache``: its optimised path copies the
current K/V into a preallocated ``torch.empty`` buffer in place, which
detaches the graph and silently kills gradients to ``k_proj_mot_gen`` and
``v_proj_mot_gen``. Leaving the cache unprepared takes the ``torch.cat``
fallback, which keeps the graph intact. The prefix is ~250 tokens against 256
image tokens at 512px, so the re-concat is cheap next to the attention itself.
"""

from __future__ import annotations

import math
import types

import torch

from conceptmod.backends.base import Backend, TextEmbeds, pin_modules, require_cuda

DEFAULT_MODEL = "sensenova/SenseNova-U1.5-8B-MoT"

# The generation branch only -- the base (understanding) projections keep the
# checkpoint's text behaviour intact. peft matches these as name suffixes, and
# the ``_mot_gen`` suffix is what keeps them off the base branch. The MLP
# targets need the parent in the pattern, because a bare ``gate_proj`` would
# also match the base branch's ``mlp.gate_proj``.
_LORA_TARGETS = [
    "q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen",
]
_LORA_TARGETS_MLP = [
    "mlp_mot_gen.gate_proj", "mlp_mot_gen.up_proj", "mlp_mot_gen.down_proj",
]

IMG_START_TOKEN = "<img>"
# Non-think mode: t2i_generate appends a pre-closed think block then <img>.
_THINK_CONTENT = "<think>\n\n</think>\n\n" + IMG_START_TOKEN

# The CFG negative. Not a prompt a caller can type: it is a sentinel that
# ``_encode_raw`` maps to the official unconditional query (no generation
# system message, no think block). ``''`` is a real, empty-text *prompt* and
# must not be routed here -- see the module docstring.
CFG_UNCOND = "\x00sensenova-uncond"


class _Prefix(TextEmbeds):
    """A prompt's conditioning: the base-branch KV prefix plus its geometry.

    ``embeds`` is the prefix's last hidden state so encoder_train's pooling
    helpers keep working; the denoiser never reads it -- it attends to
    ``cache``.
    """

    def __init__(self, embeds, mask, cache, indexes_image, text_len):
        super().__init__(embeds, mask)
        self.cache = cache
        self.indexes_image = indexes_image
        self.text_len = text_len


class SenseNovaBackend(Backend):
    # Prefix caches pin a KV tensor per prompt (~15-25 MiB at 42 layers), and
    # the op templates mint a lot of distinct prompts over a run. Cap it.
    _CACHE_LIMIT = 32

    def __init__(self, device: str, model_id: str = DEFAULT_MODEL,
                 resolution: int = 512, lora_rank: int | None = None,
                 generate_steps: int | None = None,
                 generate_guidance: float | None = None,
                 timestep_shift: float = 3.0, t_eps: float = 0.02,
                 train_mlp: bool = False, gradient_checkpointing: bool = False):
        dev = require_cuda(device)
        self.device = str(dev)
        # NEOChatModel's hand-written attention/index code allocates helper
        # tensors on the *current* CUDA device rather than the input's. On a
        # multi-GPU box that silently mixes cuda:0 scratch with cuda:1 weights
        # and dies with an illegal memory access inside the first RMSNorm, so
        # pin the process to the target device. One process, one backend.
        if dev.index is not None:
            torch.cuda.set_device(dev)
        self.resolution = resolution
        # Runtime recommendation from the model card; config.json says 1.0,
        # which the official script overrides.
        self.timestep_shift = timestep_shift

        import sensenova_u1  # noqa: F401  -- registers `neo_chat` with transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id)   # no trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # bf16 is mandatory: the _mot_gen branch ships as F32 (50 GB on disk)
        # and only comes down to ~35 GB cast. device_map pins the shards
        # straight onto the target GPU -- a plain CPU load would need 35 GB of
        # host RAM before the .to(), which this box does not reliably have.
        self.model = AutoModel.from_pretrained(
            model_id, config=config, dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        moved = pin_modules(types.SimpleNamespace(model=self.model), self.device)
        print(f"sensenova modules on {self.device}: {', '.join(moved)}")
        self.model.eval()
        # _t2i_predict_v divides by (1 - t).clamp_min(config.t_eps); normally
        # t2i_generate sets this from its own argument.
        self.model.config.t_eps = t_eps

        # Base checkpoint defaults. There is an official 8-step distillation
        # LoRA (cfg 1.0 / 8 steps); merging it is a separate step, so nothing
        # here auto-detects a distilled checkpoint.
        self.generate_steps = 50 if generate_steps is None else generate_steps
        self.generate_guidance = (
            4.0 if generate_guidance is None else generate_guidance)

        # ---- geometry -------------------------------------------------
        m = self.model
        self.merge_size = int(1 / m.downsample_ratio)          # 2
        self.pixel_patch = m.patch_size                        # 16
        self.token_patch = m.patch_size * self.merge_size      # 32 px per token
        if resolution % self.token_patch:
            raise ValueError(
                f"resolution {resolution} must be a multiple of "
                f"{self.token_patch} (patch_size * 1/downsample_ratio)")
        self.image_size = (resolution, resolution)             # (W, H)
        th = resolution // self.token_patch
        self.token_hw = (th, th)
        self.image_seq_len = th * th
        self.grid_hw = (resolution // self.pixel_patch,) * 2   # 16px patch grid
        # Packed working shape, the same z the official Euler loop steps.
        self.latent_shape = (self.image_seq_len, 3 * self.token_patch ** 2)
        self.noise_scale = self._noise_scale()

        # ---- LoRA -----------------------------------------------------
        if lora_rank is None:
            lora_rank = 16
            print("sensenova backend is LoRA-only; defaulting to rank 16")
        self.lora_rank = lora_rank
        self.compute_dtype = torch.bfloat16
        targets = list(_LORA_TARGETS)
        if train_mlp:
            targets += _LORA_TARGETS_MLP
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_rank,
                         target_modules=targets)
        # peft injects the adapters in place, so calling self.model directly
        # still routes through them; the wrapper is what gives us
        # disable_adapter() / save_pretrained().
        self.transformer = get_peft_model(self.model, cfg)
        self.transformer.to(self.device)
        for p in self.transformer.parameters():
            if p.requires_grad:
                # AdamW on bf16 LoRA params is unstable; the base stays bf16
                # and _forward pins the compute dtype so these never promote
                # the whole 17.5B forward to fp32.
                p.data = p.data.float().to(self.device)
        self.transformer.eval()
        if gradient_checkpointing:
            inner = self.model.language_model.model
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()
        self.frozen = None

        self._text_cache: dict[tuple[str, bool], _Prefix] = {}
        self.encoder_lora = False
        torch.cuda.empty_cache()

    def _noise_scale(self) -> float:
        """Reproduce t2i_generate's resolution-aware noise scaling."""
        m = self.model
        scale = float(m.noise_scale)
        if m.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            gh, gw = self.grid_hw
            ratio = math.sqrt(
                (gh * gw) / (self.merge_size ** 2)
                / float(m.noise_scale_base_image_seq_len))
            scale = ratio * float(m.noise_scale)
            if m.noise_scale_mode == "dynamic_sqrt":
                scale = math.sqrt(scale)
        return min(scale, float(m.noise_scale_max_value))

    def training_defaults(self) -> dict:
        # The base checkpoint is a 50-step / cfg 4.0 model. Sampling z_ctx at
        # full length would dominate every training step (CFG doubles it), so
        # shorten the schedule the way krea does for its raw checkpoint.
        # 16 (not 10) keeps the ratio to generate_steps in the same range as
        # the backends whose proofs landed -- at 10 the trainer only ever saw
        # 10 timesteps out of the 50 the grid renders at, which let a badly
        # scaled loss cook the training schedule while the verification grid
        # looked merely dim.
        #
        # write_context="source": flow matching here runs on pixels, and the
        # prompt-difference of the velocity field is local in z. Measured on
        # this checkpoint, the frozen "human"->"robot" direction evaluated at
        # the robot trajectory has cos +0.36 / +0.03 / -0.01 with the same
        # direction at the human trajectory (t=0.045/0.167/0.423), even though
        # those latents differ by only 0.7% / 2.8% / 10.5% in norm. Drawing
        # z_ctx from the target's trajectory therefore trains the edit at
        # latents the sampler never reaches when it renders the source prompt;
        # the only component that survives to inference is the
        # prompt-independent one, which shows up as a global relight and moves
        # the control as much as the targets.
        #
        # accumulation_steps=8: the DSL losses are not remotely flat across
        # this backend's schedule. Measured on the real checkpoint with the
        # composite proof's own phrase, the `=` loss over the 16 training
        # indices is
        #     idx  0 (t=0.000)  0.1958      idx  8 (t=0.250)  0.00085
        #     idx  1 (t=0.022)  0.0124      idx 11 (t=0.423)  0.00027
        #     idx  2 (t=0.046)  0.0026      idx 15 (t=0.833)  0.00005
        # -- a 3704x range, with 96% of the total in the first three indices
        # (at idx 0 nothing has been denoised yet, so the prompt alone has to
        # produce the image and the required push is 25% of |v|; by idx 3 it is
        # 2%). That concentration is real and is where the edit lives. It only
        # becomes fatal one draw at a time: AdamW is scale invariant, so a
        # 5e-5 draw walks the parameters as far as a 0.196 one, and at
        # accumulation_steps=1 roughly 13 of every 16 updates were full-size
        # steps along sampling noise. The 250-step round that failed shows
        # exactly that -- |delta| reached 34-47% of the required magnitude at
        # cos +0.06..+0.37 with the objective, sign-flipping across templates.
        # Summing the gradient over a stratified window first (see
        # model_train.stop_index_for) puts the loss magnitude back in charge.
        return {"sample_steps": 16, "sample_guidance": 4.0,
                "write_context": "source", "accumulation_steps": 8}

    def cfg_negative_prompt(self) -> str:
        """``CFG_UNCOND``, not ``''``.

        ``_cfg`` guides every prompt against the 9-token bare query, so that
        is the velocity a render leans on at every step -- with weight
        ``-(g - 1) = -3`` at the default guidance 4.0. Measured on the failed
        250-step adapter, it drifted ``|d|/|v|`` 0.0075 / 0.0045 / 0.0035 at
        t = 0.045 / 0.167 / 0.423 and contributed 23% / 72% / 71% of the
        *rendered* velocity change on a control prompt nothing else touched.
        The bare ``#`` was anchoring ``''`` instead, which no render ever
        evaluates -- an anchor on nothing.
        """
        return CFG_UNCOND

    def velocity_loss_scale(self, timestep: torch.Tensor) -> float:
        """``(1 - t)``: the divisor ``_t2i_predict_v`` applies to the head.

        The network predicts ``x_pred`` (the clean image) and the wrapper
        returns ``v = (x_pred - z) / (1 - t).clamp_min(t_eps)``. ``x_pred - z``
        is O(1) in pixel space for every ``t``, so all of the schedule
        dependence in ``|v|`` is that one divisor: measured frozen velocity
        std runs 2.17 at t=0.077, 2.57 at t=0.250 and 4.37 at t=0.571, i.e.
        1.87-2.00 once multiplied by ``(1 - t)``. Unscaled, the top of the
        training schedule therefore carries ~16x the MSE weight of the
        bottom, and a plain uniform-in-t MSE learns the DC offset that cuts
        those samples rather than the concept direction.

        Multiplying both sides of an op's MSE by this makes the loss an MSE
        on ``x_pred``: ``w * v_i == x_i - z``, and the shared ``z`` cancels
        out of the difference, so ``MSE(w*v_t, w*target)`` is exactly
        ``MSE(x_t, x_target)`` with no scheduler bias left.
        """
        t = timestep.to(dtype=torch.float32).reshape(())
        return float((1.0 - t).clamp_min(self.model.config.t_eps))

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> _Prefix:
        from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

        m = self.model
        if prompt == CFG_UNCOND:
            # The official unconditional: no generation system message, no
            # think block. Reproduced verbatim -- this is what CFG anchors on,
            # and it is *not* what the empty prompt encodes to.
            query = m._build_t2i_query("", append_text=IMG_START_TOKEN)
        else:
            # Every user-visible prompt, '' included, takes t2i_generate's
            # conditional path.
            query = m._build_t2i_query(
                f"{prompt}", system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text=_THINK_CONTENT)
        input_ids, indexes, attn = m._build_t2i_text_inputs(self.tokenizer, query)
        cache, hidden = m._t2i_prefix_forward(input_ids, indexes, attn)
        text_len = indexes.shape[1]
        th, tw = self.token_hw
        indexes_image = m._build_t2i_image_indexes(
            th, tw, text_len, device=input_ids.device)
        mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        return _Prefix(hidden, mask, cache, indexes_image, text_len)

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        key = (prompt, frozen)
        if key not in self._text_cache:
            while len(self._text_cache) >= self._CACHE_LIMIT:
                self._text_cache.pop(next(iter(self._text_cache)))
            self._text_cache[key] = self._encode_raw(prompt)
        return self._text_cache[key]

    def encode_text_grad(self, prompt: str) -> TextEmbeds:
        return self._encode_raw(prompt)

    def attach_encoder_lora(self, rank: int = 8):
        raise NotImplementedError(
            "SenseNova has no separable text encoder: the understanding and "
            "generation branches are the same 42 layers, distinguished only "
            "by the _mot_gen weight suffix, and they share one joint "
            "attention. An 'encoder' adapter would live in the same peft "
            "model as the denoiser adapter, so disable_adapter() could not "
            "tell the two stages apart. Train this backend with --stage model."
        )

    # ---------------- velocity ----------------

    def _forward(self, z, timestep, text: _Prefix):
        """One denoiser call: patch-32 latents -> velocity, same shape."""
        m = self.model
        dtype = self.compute_dtype
        B = z.shape[0]
        th, tw = self.token_hw
        gh, gw = self.grid_hw
        n_tok = self.image_seq_len
        # t stays fp32 like the official sampler's `torch.linspace`: the head
        # divides by (1 - t).clamp_min(t_eps), and near t=1 a bf16 (1 - t)
        # would carry percent-level error into the velocity. The embedder
        # casts to its own dtype internally, so fp32 is safe to hand it.
        t = timestep.to(device=self.device, dtype=torch.float32).reshape(())
        z = z.to(device=self.device, dtype=dtype)

        # Pixels are the latents here: unpack to (B,3,H,W), then re-patchify
        # at the 16px vision patch to feed the (VAE-replacing) patchifier.
        img = m.unpatchify(z, self.token_patch, self.image_size[1], self.image_size[0])
        image_input = m.patchify(img, self.pixel_patch, channel_first=True)
        grid_hw = torch.tensor([[gh, gw]] * B, device=self.device)
        embeds = m.extract_feature(
            image_input.reshape(B * gh * gw, -1), gen_model=True, grid_hw=grid_hw,
        ).view(B, n_tok, -1)

        t_expanded = t.expand(B * n_tok)
        t_emb = m.fm_modules["timestep_embedder"](t_expanded).view(B, n_tok, -1)
        if m.add_noise_scale_embedding:
            ns = torch.full_like(
                t_expanded, self.noise_scale / m.noise_scale_max_value)
            t_emb = t_emb + m.fm_modules["noise_scale_embedder"](ns).view(B, n_tok, -1)
        embeds = embeds + t_emb

        # attention_mask None -> the joint block where image tokens attend
        # bidirectionally over [text prefix + all image tokens].
        v = m._t2i_predict_v(
            embeds, text.indexes_image, {"full_attention": None}, text.cache,
            t, z, image_token_num=n_tok, timestep_embeddings=t_emb,
            image_size=self.image_size,
        )
        return v.float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self.transformer.disable_adapter():
                return self._forward(z, timestep, text)
        return self._forward(z, timestep, text)

    # ---------------- sampling ----------------

    def _timesteps(self, num_steps: int) -> torch.Tensor:
        """num_steps+1 points running 0 (pure noise) -> 1 (image)."""
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        return self.model._apply_time_schedule(
            ts, self.image_seq_len, self.timestep_shift)

    def _cfg(self, prompt, z, t, guidance, frozen):
        # SenseNova convention: v_uncond + g*(v_cond - v_uncond), and
        # t2i_generate gates it on `cfg_scale > 1`, not > 0. The negative is
        # CFG_UNCOND (the 9-token bare query), never '': t2i_generate guides
        # an empty user prompt against it just like any other prompt.
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance > 1 and prompt != CFG_UNCOND:
            v_u = self.predict_v(CFG_UNCOND, z, t, frozen=frozen)
            v = v_u + guidance * (v - v_u)
        return v

    def _noise(self, generator):
        # Seeded in image space at the model's own noise_scale, then packed --
        # unit-variance noise is off-distribution for this checkpoint.
        img = torch.randn(
            (1, 3, self.image_size[1], self.image_size[0]), generator=generator,
            device=self.device, dtype=torch.float32,
        )
        return self.model.patchify(self.noise_scale * img, self.token_patch)

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        ts = self._timesteps(num_steps)
        z = self._noise(generator)
        for i in range(num_steps):
            if i >= stop_index:
                return z, ts[i]
            v = self._cfg(prompt, z, ts[i], guidance, frozen=False)
            z = z + (ts[i + 1] - ts[i]) * v
        return z, ts[num_steps - 1]

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        ts = self._timesteps(num_steps)
        z = self._noise(generator)
        for i in range(num_steps):
            with torch.set_grad_enabled(not frozen and i >= num_steps - grad_steps):
                v = self._cfg(prompt, z, ts[i], guidance, frozen=frozen)
                z = z + (ts[i + 1] - ts[i]) * v
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        """No VAE: the latents *are* the image, normalised to mean=std=0.5."""
        with torch.set_grad_enabled(grad):
            img = self.model.unpatchify(
                z, self.token_patch, self.image_size[1], self.image_size[0])
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        from PIL import Image

        num_steps = num_steps or self.generate_steps
        guidance = self.generate_guidance if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(seed)
        # frozen is threaded down to predict_v rather than wrapped here:
        # peft's disable_adapter() does not nest, so an outer context would be
        # cancelled by the first inner __exit__ and quietly re-enable the LoRA.
        img = self.render(prompt, g, num_steps, guidance, grad_steps=0,
                          frozen=frozen)
        arr = (img * 0.5 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0)
        return Image.fromarray((arr * 255).round().byte().cpu().numpy())

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        assert params
        if train_method not in (None, "lora"):
            print("sensenova backend is LoRA-only; "
                  f"ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str):
        self.transformer.save_pretrained(path)
