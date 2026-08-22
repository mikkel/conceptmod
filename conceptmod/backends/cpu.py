"""CPU / dummy backend: a tiny flow-matching stand-in for the test cycle.

No Hub download, no GPU, no 20B weights. A two-class text table plus two
Linear layers (LoRA attaches to the class path) expose the same
velocity-space geometry the real backends feed into ``dsl`` / ``ops``:

    v(z, t, c) − v(z, t, '')

The known-answer sample is ``red=blue``: embeddings are opposite on a line
(``red = +e``, ``blue = −e``, empty = 0), so a write must rotate the LoRA
until the trained red-direction aligns with the frozen blue-direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.backends.base import Backend, TextEmbeds

LATENT_CHANNELS = 4
LATENT_HW = 8  # 8px "latent"
TEXT_DIM = 8
DEFAULT_LORA_RANK = 4
# Class path only. LoRA on linear1 can fit a z-local offset that does not
# generalize to the probe; linear2 is the 2-class remap the sample asserts.
_LORA_TARGETS = ["linear2"]

# 2-class sample: write red so it behaves like blue.
SAMPLE_PHRASE = "red=blue"
SAMPLE_SRC = "red"
SAMPLE_DST = "blue"
SAMPLE_SEED = 0
SAMPLE_STEPS = 80
SAMPLE_LR = 1e-2
SAMPLE_LORA = DEFAULT_LORA_RANK
COSINE_THRESHOLD = 0.45

_CLASS_RED = 1
_CLASS_BLUE = 2
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def resolve_device(device: str) -> str:
    """CPU is the point of this backend; CUDA is optional, never required."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if torch.cuda.is_available():
            return str(dev)
        print(f"cpu backend: {device!r} requested but CUDA is unavailable; using cpu")
        return "cpu"
    return "cpu"


def prompt_class(prompt: str) -> int:
    """Map a prompt onto the 2-class table (plus empty/other = 0)."""
    tokens = set(_TOKEN_RE.findall((prompt or "").lower()))
    has_red = SAMPLE_SRC in tokens
    has_blue = SAMPLE_DST in tokens
    if has_red and not has_blue:
        return _CLASS_RED
    if has_blue and not has_red:
        return _CLASS_BLUE
    return 0


class TinyTextEncoder(nn.Module):
    """Fixed 2-class table + identity Linear stack (encoder-LoRA hooks)."""

    def __init__(self, dim: int = TEXT_DIM):
        super().__init__()
        table = torch.zeros(3, dim)
        table[_CLASS_RED, 0] = 1.0
        table[_CLASS_BLUE, 0] = -1.0
        self.register_buffer("class_table", table)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        e = self.class_table[class_ids]
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(e))))


class TinyFlowDiT(nn.Module):
    """Affine flow head: ``v = linear1(z, t) + linear2(c)``.

    Two Linear layers so LoRA can attach. ``linear2`` is the class path —
    ``red = +e`` and ``blue = −e`` start opposite, and a write must flip
    the trained red column toward blue. ``linear1`` is the shared (z, t)
    path (CFG geometry still lives in ``v(c) − v('')``).
    """

    def __init__(self, channels: int = LATENT_CHANNELS, spatial: int = LATENT_HW,
                 text_dim: int = TEXT_DIM):
        super().__init__()
        self.channels = channels
        self.spatial = spatial
        self.text_dim = text_dim
        z_dim = channels * spatial * spatial
        self.linear1 = nn.Linear(z_dim + 1, z_dim)
        self.linear2 = nn.Linear(text_dim, z_dim, bias=False)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.zeros_(self.linear1.bias)
        nn.init.normal_(self.linear2.weight, std=0.8)

    def forward(self, z: torch.Tensor, timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z.shape
        z_flat = z.reshape(b, -1)
        t = timestep.to(dtype=z.dtype, device=z.device).reshape(b, 1) / 1000.0
        cond = encoder_hidden_states.to(dtype=z.dtype, device=z.device)
        if cond.ndim == 3:
            cond = cond.mean(dim=1)
        v = self.linear1(torch.cat([z_flat, t], dim=-1)) + self.linear2(cond)
        return v.reshape(b, c, h, w)


class CpuBackend(Backend):
    """In-repo tiny DiT. ``--backend cpu`` (alias: ``dummy``)."""

    def __init__(self, device: str = "cpu", model_id: str | None = None,
                 resolution: int | None = None, lora_rank: int | None = None,
                 seed: int = SAMPLE_SEED):
        del model_id, resolution  # no hub id / pixel resolution
        self.device = resolve_device(device)
        if lora_rank is None:
            lora_rank = DEFAULT_LORA_RANK
            print(f"cpu backend is LoRA-only; defaulting to rank {lora_rank}")
        self.lora_rank = lora_rank
        self.latent_channels = LATENT_CHANNELS
        self.latent_shape = (LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        self.generate_steps = 4
        self.generate_guidance = 1.0

        torch.manual_seed(seed)
        self.text_encoder = TinyTextEncoder().to(self.device)
        transformer = TinyFlowDiT().to(self.device)

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank,
            target_modules=_LORA_TARGETS,
        )
        self.transformer = get_peft_model(transformer, config)
        self.transformer.to(self.device)
        for p in self.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.transformer.eval()
        self.frozen = None
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}
        self.encoder_lora = False

    def training_defaults(self) -> dict:
        return {"sample_steps": 4, "sample_guidance": 1.0}

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        cid = torch.tensor([prompt_class(prompt)], device=self.device)
        embeds = self.text_encoder(cid).unsqueeze(1)
        mask = torch.ones(1, 1, device=self.device, dtype=torch.bool)
        return TextEmbeds(embeds, mask)

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        key = (prompt, frozen)
        if key not in self._text_cache:
            if self.encoder_lora and frozen:
                with self.text_encoder.disable_adapter():
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
        self.text_encoder = get_peft_model(self.text_encoder, config)
        self.text_encoder.to(self.device)
        for _n, p in self.text_encoder.named_parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _forward(self, model, z, timestep, text: TextEmbeds):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device)
        t = timestep.reshape(-1).to(device=self.device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.expand(z.shape[0])
        return model(z.to(self.device), t, text.embeds.to(self.device)).float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self.transformer.disable_adapter():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    # ---------------- sampling ----------------

    def _schedule(self, num_steps: int):
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        timesteps = 1000.0 * sigmas[:-1]
        return timesteps, sigmas

    def _cfg(self, prompt, z, t, guidance, frozen):
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance != 1.0 and prompt != "":
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
        timesteps, sigmas = self._schedule(num_steps)
        z = self._noise(generator)
        for i, t in enumerate(timesteps):
            if i >= stop_index:
                return z, t
            v = self._cfg(prompt, z, t, guidance, frozen=False)
            z = z + (sigmas[i + 1] - sigmas[i]) * v
        return z, timesteps[-1]

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        timesteps, sigmas = self._schedule(num_steps)
        z = self._noise(generator)
        n = len(timesteps)
        for i, t in enumerate(timesteps):
            with torch.set_grad_enabled(not frozen and i >= n - grad_steps):
                v = self._cfg(prompt, z, t, guidance, frozen=frozen)
                z = z + (sigmas[i + 1] - sigmas[i]) * v
        return self.decode(z, grad=not frozen and grad_steps > 0)

    def decode(self, z, grad=False):
        # First 3 latent channels as an 8×8 RGB stand-in (no VAE).
        with torch.set_grad_enabled(grad):
            return torch.tanh(z[:, :3])

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
        assert params, "peft returned no trainable params"
        if train_method not in (None, "lora", "xattn", "full"):
            print(f"cpu backend is LoRA-only; ignoring train_method={train_method!r}")
        return params

    def save_trained(self, path: str) -> None:
        self.transformer.save_pretrained(path)


def write_alignment_cosine(backend, src: str = SAMPLE_SRC, dst: str = SAMPLE_DST,
                           seed: int = SAMPLE_SEED) -> float:
    """Cosine of trained ``v(src)−v('')`` vs frozen ``v(dst)−v('')``."""
    g = torch.Generator(device=backend.device).manual_seed(seed + 17)
    z = torch.randn((1, *backend.latent_shape), generator=g,
                    device=backend.device, dtype=torch.float32)
    t = torch.tensor([500.0], device=backend.device)
    with torch.no_grad():
        v_src = backend.predict_v(src, z, t, frozen=False)
        v_dst = backend.predict_v(dst, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
    # Frozen uncond so a shared LoRA drift on linear1 cannot hide the remap.
    d_src = (v_src - v_null).flatten().unsqueeze(0)
    d_dst = (v_dst - v_null).flatten().unsqueeze(0)
    return F.cosine_similarity(d_src, d_dst, dim=1, eps=1e-6).item()


def lora_B_norm(backend) -> float:
    """``lora_B`` starts at 0, so this is the learned LoRA delta size."""
    total = 0.0
    for name, param in backend.transformer.named_parameters():
        if "lora_B" in name:
            total += param.detach().float().pow(2).sum().item()
    return total ** 0.5


@dataclass
class SampleResult:
    phrase: str
    cosine_before: float
    cosine_after: float
    lora_delta_norm: float
    elapsed_s: float
    history: list[float]


def run_sample_problem(
    device: str = "cpu",
    phrase: str = SAMPLE_PHRASE,
    iterations: int = SAMPLE_STEPS,
    lr: float = SAMPLE_LR,
    lora_rank: int = SAMPLE_LORA,
    seed: int = SAMPLE_SEED,
) -> SampleResult:
    """Train ``red=blue`` through the real ``train_model`` → dsl/ops path."""
    from conceptmod import ops
    from conceptmod.model_train import train_model

    torch.manual_seed(seed)
    t0 = time.time()
    backend = CpuBackend(device=device, lora_rank=lora_rank, seed=seed)
    cosine_before = write_alignment_cosine(backend, seed=seed)
    cfg = ops.OpDefaults(sample_steps=4, sample_guidance=1.0, write_guidance=1.0)
    history = train_model(
        backend,
        phrase,
        iterations=iterations,
        lr=lr,
        train_method="lora",
        seed=seed,
        op_defaults=cfg,
        log_every=0,
        sample_prompt=None,
    )
    cosine_after = write_alignment_cosine(backend, seed=seed)
    return SampleResult(
        phrase=phrase,
        cosine_before=cosine_before,
        cosine_after=cosine_after,
        lora_delta_norm=lora_B_norm(backend),
        elapsed_s=time.time() - t0,
        history=list(history),
    )
