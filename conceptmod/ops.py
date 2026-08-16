"""Per-op training losses on velocity predictions.

Notation: v_f(p) / v_t(p) are the frozen / trained model's velocity
predictions for prompt p, evaluated at the same partially-denoised latents
z_ctx and timestep. z_ctx is sampled by partially denoising from pure noise
under a *context prompt* that depends on the op (like the original's
``quick_sample_till_t``). The classifier-free-guidance direction
``v(p) - v('')`` is the concept direction every op manipulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from conceptmod import dsl

WRITE_TEMPLATES = [
    "{}",
    "{}",
    "a photo of {}",
    "a photo of a {}",
    "a painting of a {}",
    "a portrait of a {}",
    "a {} in a landscape",
    "a close-up of a {}",
]


@dataclass
class OpDefaults:
    exaggerate_guidance: float = 3.0   # g in v* = v0 + g (vc - v0)
    erase_guidance: float = 1.0        # g in v* = v0 - g (vc - v0)
    write_guidance: float = 2.0        # g=1: a behaves exactly like b;
    #                                    >1 overshoots for a stronger write
    sample_guidance: float = 4.5       # CFG while sampling z_ctx
    sample_steps: int = 14             # schedule length for z_ctx sampling
    pixel_render_steps: int = 10       # full-render steps for the ^ op
    pixel_grad_steps: int = 1          # how many final steps carry gradient
    orthogonal_scale: float = 0.01    # parity with the original % loss scale
    probe_p: float = 0.7               # fraction of ++ steps trained on a
    #                                    random probe prompt (global effect)
    #                                    instead of the concept prompt itself


class StepContext:
    """One training step: a shared start noise + timestep index, with caches
    for partial-denoise latents (per context prompt) and velocity
    predictions (per model/prompt/context)."""

    def __init__(self, backend, stop_index: int, seed: int, cfg: OpDefaults,
                 probe: str | None = None):
        self.backend = backend
        self.stop_index = stop_index
        self.seed = seed
        self.cfg = cfg
        self.probe = probe  # random prompt for globalized ++ (None = classic)
        self._z: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._v: dict[tuple, torch.Tensor] = {}

    def z_for(self, context_prompt: str):
        if context_prompt not in self._z:
            g = torch.Generator(device=self.backend.device).manual_seed(self.seed)
            self._z[context_prompt] = self.backend.partial_denoise(
                context_prompt, self.stop_index, self.cfg.sample_steps,
                self.cfg.sample_guidance, g)
        return self._z[context_prompt]

    def v(self, prompt: str, context_prompt: str, frozen: bool,
          grad: bool = False) -> torch.Tensor:
        key = (prompt, context_prompt, frozen, grad)
        if key not in self._v:
            z, t = self.z_for(context_prompt)
            if grad:
                self._v[key] = self.backend.predict_v(prompt, z, t, frozen=frozen)
            else:
                with torch.no_grad():
                    self._v[key] = self.backend.predict_v(prompt, z, t, frozen=frozen)
        return self._v[key]


def rule_loss(rule: dsl.Rule, ctx: StepContext) -> torch.Tensor:
    """Unscaled loss for one rule (the trainer applies rule.alpha)."""
    cfg = ctx.cfg

    if rule.op == dsl.EXAGGERATE:
        g = rule.options.get("guidance", cfg.exaggerate_guidance)
        if ctx.probe:
            # globalized: make an arbitrary probe prompt p produce more of
            # the concept, by amplifying the direction toward "p, concept"
            # (concept-sliders style; generalizes to all prompts)
            p, comb = ctx.probe, f"{ctx.probe}, {rule.a}"
            v0 = ctx.v(p, p, frozen=True)
            vc = ctx.v(comb, p, frozen=True)
            target = v0 + g * (vc - v0)
            vt = ctx.v(p, p, frozen=False, grad=True)
            return F.mse_loss(vt, target)
        # classic: amplify the concept prompt's own direction past the
        # frozen model's (trained prompt = the concept itself)
        v0 = ctx.v("", "", frozen=True)
        vc = ctx.v(rule.a, "", frozen=True)
        target = v0 + g * (vc - v0)
        vt = ctx.v(rule.a, "", frozen=False, grad=True)
        return F.mse_loss(vt, target)

    if rule.op == dsl.ERASE:
        # true ESD: sample in the concept's own context, push its velocity
        # to the negatively-guided target
        g = rule.options.get("guidance", cfg.erase_guidance)
        v0 = ctx.v("", rule.a, frozen=True)
        vc = ctx.v(rule.a, rule.a, frozen=True)
        target = v0 - g * (vc - v0)
        vt = ctx.v(rule.a, rule.a, frozen=False, grad=True)
        return F.mse_loss(vt, target)

    if rule.op == dsl.WRITE:
        # train prompt a (possibly '' = unconditional) to behave like b,
        # in b's sampling context. For non-empty a, vary a shared template
        # around both concepts so the write generalizes beyond the bare
        # word ("a photo of a cat" -> dog, not just "cat" -> dog).
        g = rule.options.get("guidance", cfg.write_guidance)
        a, b = rule.a, rule.b
        if a:
            import random as _random

            tmpl = _random.Random(ctx.seed).choice(WRITE_TEMPLATES)
            a, b = tmpl.format(a), tmpl.format(b)
        v0 = ctx.v("", b, frozen=True)
        vb = ctx.v(b, b, frozen=True)
        target = v0 + g * (vb - v0)
        vt = ctx.v(a, b, frozen=False, grad=True)
        return F.mse_loss(vt, target)

    if rule.op == dsl.FREEZE:
        target = ctx.v(rule.b, rule.b, frozen=True)
        vt = ctx.v(rule.a, rule.b, frozen=False, grad=True)
        return F.mse_loss(vt, target)

    if rule.op == dsl.ORTHOGONAL:
        # decorrelate trained b-direction from frozen a-direction;
        # negative rule.alpha flips the sign -> alignment (blend)
        da = ctx.v(rule.a, "", frozen=True) - ctx.v("", "", frozen=True)
        db = (ctx.v(rule.b, "", frozen=False, grad=True)
              - ctx.v("", "", frozen=False))
        cos = F.cosine_similarity(da.flatten(1), db.flatten(1), dim=1, eps=1e-6)
        return cfg.orthogonal_scale * cos.abs().mean()

    if rule.op == dsl.PIXEL:
        # render both prompts from the same start noise; L2 in pixel space,
        # gradient through the final Euler step(s) + VAE decode
        g_a = torch.Generator(device=ctx.backend.device).manual_seed(ctx.seed)
        img_a = ctx.backend.render(rule.a, g_a, cfg.pixel_render_steps,
                                   cfg.sample_guidance,
                                   grad_steps=cfg.pixel_grad_steps)
        g_b = torch.Generator(device=ctx.backend.device).manual_seed(ctx.seed)
        with torch.no_grad():
            img_b = ctx.backend.render(rule.b, g_b, cfg.pixel_render_steps,
                                       cfg.sample_guidance, frozen=True)
        return F.mse_loss(img_a, img_b)

    if rule.op == dsl.REWARD:
        raise NotImplementedError(
            "the ';' (ImageReward) op is not implemented in conceptmod 2.x")

    raise ValueError(f"unknown op {rule.op!r}")
