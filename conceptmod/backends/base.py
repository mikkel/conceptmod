"""Backend protocol: what a diffusion model must expose for conceptmod.

All models are flow-matching: the network predicts a velocity v(z, t, text)
and an Euler step is ``z_next = z + (sigma_next - sigma) * v``. Everything the
DSL does is built from velocity predictions of a *trained* and a *frozen*
copy of the same network.
"""

from __future__ import annotations

import abc

import torch


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
