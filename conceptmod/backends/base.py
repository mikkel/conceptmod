"""Backend protocol: what a diffusion model must expose for conceptmod.

All models are flow-matching: the network predicts a velocity v(z, t, text)
and an Euler step is ``z_next = z + (sigma_next - sigma) * v``. Everything the
DSL does is built from velocity predictions of a *trained* and a *frozen*
copy of the same network.
"""

from __future__ import annotations

import abc

import torch


def require_cuda(device: str) -> torch.device:
    """All training / sampling runs on one CUDA device. CPU is not a fallback."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"backends require a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return dev


def pin_modules(root, device: str) -> list[str]:
    """Move every nn.Module hanging off ``root`` onto ``device`` and check it stuck."""
    dev = require_cuda(device)
    moved = []
    for name in dir(root):
        if name.startswith("_"):
            continue
        try:
            mod = getattr(root, name)
        except Exception:
            continue
        if isinstance(mod, torch.nn.Module):
            mod.to(dev)
            _assert_cuda(mod, dev, name)
            moved.append(name)
    return moved


def _assert_cuda(module: torch.nn.Module, device: torch.device, name: str) -> None:
    want = device.index
    for n, p in module.named_parameters():
        if p.device.type != "cuda" or (want is not None and p.device.index != want):
            raise RuntimeError(f"{name}.{n} is on {p.device}, expected {device}")
    for n, b in module.named_buffers():
        if b.device.type != "cuda" or (want is not None and b.device.index != want):
            raise RuntimeError(f"{name}.{n} buffer is on {b.device}, expected {device}")


class TextEmbeds:
    """Prompt embeddings plus attention mask (mask may be None)."""

    def __init__(self, embeds: torch.Tensor, mask: torch.Tensor | None):
        self.embeds = embeds
        self.mask = mask


class Backend(abc.ABC):
    device: str
    latent_shape: tuple  # (C, H, W) for one sample

    @abc.abstractmethod
    def encode_text(self, prompt: str) -> TextEmbeds:
        """Encode a prompt (cached). '' is the unconditional prompt."""

    @abc.abstractmethod
    def predict_v(self, prompt: str, z: torch.Tensor, timestep: torch.Tensor,
                  frozen: bool) -> torch.Tensor:
        """Velocity prediction for latents z at a scheduler timestep."""

    @abc.abstractmethod
    def partial_denoise(self, prompt: str, stop_index: int, num_steps: int,
                        guidance: float, generator: torch.Generator
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Euler steps from pure noise down to ``timesteps[stop_index]``
        using the trained model (no grad). Returns (z_t, timestep_t)."""

    @abc.abstractmethod
    def generate(self, prompt: str, seed: int, num_steps: int, guidance: float,
                 frozen: bool = False):
        """Full generation -> PIL.Image. ``frozen=True`` uses the untouched
        reference model (the 'before' image)."""

    @abc.abstractmethod
    def trainable_parameters(self, train_method: str) -> list[torch.nn.Parameter]:
        """Select the parameter group to finetune (xattn/selfattn/full/...)."""

    @abc.abstractmethod
    def save_trained(self, path: str) -> None:
        """Save the trained weights (full state dict or LoRA adapter)."""

    def training_defaults(self) -> dict:
        """Backend-specific OpDefaults overrides (sample_steps / sample_guidance)."""
        return {}
