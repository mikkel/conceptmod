"""Per-op training losses on velocity predictions.

Notation: v_f(p) / v_t(p) are the frozen / trained model's velocity
predictions for prompt p, evaluated at the same partially-denoised latents
z_ctx and timestep. z_ctx is sampled by partially denoising from pure noise
under a *context prompt* that depends on the op (like the original's
``quick_sample_till_t``). The classifier-free-guidance direction
``v(p) - v('')`` is the concept direction every op manipulates.

An op may only train a prompt's velocity where that velocity is evaluated, so
the context prompt should be the prompt whose velocity carries the gradient.
ERASE, FREEZE(a == b) and the globalized EXAGGERATE already sample that way;
WRITE's context is selectable through ``OpDefaults.write_context`` because the
historical default samples in the *target's* context instead (see the WRITE
branch of :func:`rule_loss`).

Every MSE here goes through ``StepContext.vmse``, which applies the backend's
``velocity_loss_scale``. The trainer draws one uniformly random timestep per
step, so an MSE on a velocity whose magnitude depends strongly on ``t`` is not
a uniform objective: the loudest timesteps set the gradient, and the cheapest
way to satisfy them is a prompt-independent DC offset rather than the edit.

``velocity_loss_scale`` only removes the part of that ``t``-dependence that
comes from the *parameterisation* of ``v``. What remains is real: how much a
prompt can still change the prediction at a given ``t``. On a pixel-space
backend that residual is brutally peaked -- measured on SenseNova-U1.5, the
``=`` loss runs 4.0e-2 at the noise end of the 16-point training schedule and
1e-4 by a quarter of the way in, a ~400x range, with ~90% of the total sitting
in the first three indices. That is not a bug to weight away; it is where the
edit lives. It only becomes fatal because the trainer hands *one* draw at a
time to AdamW, which is scale invariant: a draw carrying 1e-4 of signal moves
the parameters exactly as far as one carrying 4e-2. See
``OpDefaults.accumulation_steps``.
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
    # scene/action templates so the edit survives contextual prompts
    # ("a cat sitting on a windowsill" resisted the plain-template pool)
    "a {} sitting on a windowsill",
    "a {} sitting on a chair",
    "a {} in a garden",
    "a {} sleeping on a bed",
    "a {} next to a person",
    "a {} outdoors, golden hour",
    "a {} perched by a window",
    "a {} indoors, natural light",
    "a side view of a {} sitting",
    "a {} walking in a city",
]

ERASE_TEMPLATES = [
    "{}",
    "{}",
    "a {} photograph",
    "a photo of {}",
    "a {} image of a scene",
    "{} style",
    "{}, detailed picture",
    "an image of {}",
]


@dataclass
class OpDefaults:
    exaggerate_guidance: float = 3.0   # g in v* = v0 + g (vc - v0)
    erase_guidance: float = 2.0        # g in v* = v0 - g (vc - v0)
    write_guidance: float = 2.0        # g=1: a behaves exactly like b;
    #                                    >1 overshoots for a stronger write
    sample_guidance: float = 4.5       # CFG while sampling z_ctx
    sample_steps: int = 14             # schedule length for z_ctx sampling
    pixel_render_steps: int = 10       # full-render steps for the ^ op
    pixel_grad_steps: int = 1          # how many final steps carry gradient
    pixel_anchor_weight: float = 20.0  # freeze-anchor on the ^ op's b side
    #                                    (pixel L2 is small vs velocity MSE)
    orthogonal_scale: float = 0.01    # parity with the original % loss scale
    probe_p: float = 0.7               # fraction of ++ steps trained on a
    #                                    random probe prompt (global effect)
    #                                    instead of the concept prompt itself
    write_context: str = "target"      # which trajectory '=' draws z_ctx from:
    #                                    "target" = b's (historical), "source"
    #                                    = a's, the prompt being trained.
    #                                    See rule_loss's WRITE branch.
    accumulation_steps: int = 1        # micro-steps averaged per optimizer
    #                                    step. Not a batch-size/speed knob
    #                                    here: it is what makes the
    #                                    t-marginalised objective well posed.
    #                                    Each micro-step draws one timestep,
    #                                    and the loss at the informative end of
    #                                    the schedule can be hundreds of times
    #                                    larger than at the other end. Summing
    #                                    the gradients first lets that
    #                                    magnitude act as the importance
    #                                    weight it is; stepping on each draw
    #                                    separately lets AdamW normalise it
    #                                    away and turns the quiet draws into
    #                                    full-size random walk. Backends whose
    #                                    velocity is strongly t-dependent
    #                                    should raise it in
    #                                    ``training_defaults``.


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
        # The prompt CFG anchors on. '' on every diffusers backend; SenseNova
        # keeps a separate bare query for it. See the FREEZE branch.
        self.uncond = backend.cfg_negative_prompt()
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

    def vmse(self, pred: torch.Tensor, target: torch.Tensor,
             context_prompt: str) -> torch.Tensor:
        """MSE between two velocity fields, rescaled so the loss is
        commensurate across the schedule.

        Every op target is an affine combination of velocities evaluated at
        the *same* ``(z, t)`` as ``pred``, so scaling both sides by the
        backend's ``velocity_loss_scale`` is exactly a change of variables --
        no op semantics change, and backends with an O(1) velocity (all the
        diffusers ones) get the identity. See ``Backend.velocity_loss_scale``
        for why an unscaled MSE learns a global DC offset on SenseNova.
        """
        w = float(self.backend.velocity_loss_scale(self.z_for(context_prompt)[1]))
        loss = F.mse_loss(pred, target)
        # w**2 * MSE(a, b) == MSE(w*a, w*b) exactly, without allocating two
        # more full-size tensors into the graph on a 17.5B backward.
        return loss if w == 1.0 else (w * w) * loss


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
            return ctx.vmse(vt, target, p)
        # classic: amplify the concept prompt's own direction past the
        # frozen model's (trained prompt = the concept itself)
        v0 = ctx.v("", "", frozen=True)
        vc = ctx.v(rule.a, "", frozen=True)
        target = v0 + g * (vc - v0)
        vt = ctx.v(rule.a, "", frozen=False, grad=True)
        return ctx.vmse(vt, target, "")

    if rule.op == dsl.ERASE:
        # true ESD: sample in the concept's own context, push its velocity
        # to the negatively-guided target. The context is drawn from varied
        # templates so the erase covers scene modes, not just the bare
        # phrase (a bare-phrase erase left "a monochrome photograph of a
        # city street" untouched while flipping its synonyms).
        import random as _random

        g = rule.options.get("guidance", cfg.erase_guidance)
        c = _random.Random(ctx.seed).choice(ERASE_TEMPLATES).format(rule.a)
        v0 = ctx.v("", c, frozen=True)
        vc = ctx.v(c, c, frozen=True)
        target = v0 - g * (vc - v0)
        vt = ctx.v(c, c, frozen=False, grad=True)
        return ctx.vmse(vt, target, c)

    if rule.op == dsl.WRITE:
        # train prompt a (possibly '' = the empty prompt) to behave like b.
        # For non-empty a, vary a shared template around both concepts so the
        # write generalizes beyond the bare word ("a photo of a cat" -> dog,
        # not just "cat" -> dog).
        #
        # ``write_context`` picks the trajectory z_ctx is drawn from. The loss
        # only moves v(a) at the latents it is evaluated on, and the latents
        # that matter at inference are the ones the sampler visits while
        # rendering a -- so "source" (a's own trajectory, the ESD convention
        # the ERASE branch below already follows) is the objective that means
        # what the op says. "target" (b's trajectory) is the historical
        # behaviour and only works when the a->b direction is smooth enough in
        # z to carry from one trajectory to the other.
        #
        # It does not carry on a pixel-space backend. Measured on
        # SenseNova-U1.5 with the composite proof's own prompts, the frozen
        # human->robot direction at the robot trajectory versus at the human
        # trajectory has cos +0.36 / +0.03 / -0.01 at t=0.045/0.167/0.423,
        # while the two latents differ by only 0.7% / 2.8% / 10.5% in norm --
        # the direction decorrelates faster than the trajectories separate.
        # Training under "target" there fits a target that is orthogonal to
        # the one needed where the sampler actually goes, so the run learns
        # only the prompt-independent remainder (a global relight) and the
        # grid shows no edit. Backends declare their choice through
        # ``training_defaults``.
        g = rule.options.get("guidance", cfg.write_guidance)
        a, b = rule.a, rule.b
        if a:
            import random as _random

            tmpl = _random.Random(ctx.seed).choice(WRITE_TEMPLATES)
            a, b = tmpl.format(a), tmpl.format(b)
        if cfg.write_context not in ("target", "source"):
            raise ValueError(
                f"write_context must be 'target' or 'source', "
                f"got {cfg.write_context!r}")
        # a is '' for the "=b" form; the empty prompt is a fine trajectory to
        # sample, but b stays the context there so the historical behaviour of
        # writing a concept into the empty prompt is unchanged.
        zctx = a if (cfg.write_context == "source" and a) else b
        v0 = ctx.v("", zctx, frozen=True)
        vb = ctx.v(b, zctx, frozen=True)
        target = v0 + g * (vb - v0)
        vt = ctx.v(a, zctx, frozen=False, grad=True)
        return ctx.vmse(vt, target, zctx)

    if rule.op == dsl.FREEZE:
        a, b = rule.a, rule.b
        if not a and not b:
            # The bare '#' means "freeze the unconditional prompt": the one
            # the sampler evaluates on every CFG step, whose drift therefore
            # lands on *every* rendered prompt. Under v_u + g(v_c - v_u) that
            # drift enters each render with weight -(g - 1), so at g=4 an
            # unanchored uncond is a 3x-amplified, prompt-independent
            # contaminant -- which is what a global relight looks like.
            # On the diffusers backends that prompt is '' and this is a no-op;
            # on SenseNova the CFG negative is a separate bare query and ''
            # is a real prompt the sampler never visits, so anchoring '' would
            # constrain nothing that reaches an image.
            a = b = ctx.uncond
        target = ctx.v(b, b, frozen=True)
        vt = ctx.v(a, b, frozen=False, grad=True)
        return ctx.vmse(vt, target, b)

    if rule.op == dsl.ORTHOGONAL:
        # decorrelate trained b-direction from frozen a-direction;
        # negative rule.alpha flips the sign -> alignment (blend).
        # No vmse here: cosine is scale-invariant, and both directions are
        # velocities at the same timestep, so velocity_loss_scale cancels.
        da = ctx.v(rule.a, "", frozen=True) - ctx.v("", "", frozen=True)
        db = (ctx.v(rule.b, "", frozen=False, grad=True)
              - ctx.v("", "", frozen=False))
        cos = F.cosine_similarity(da.flatten(1), db.flatten(1), dim=1, eps=1e-6)
        return cfg.orthogonal_scale * cos.abs().mean()

    if rule.op == dsl.PIXEL:
        # render both prompts from the same start noise; L2 in pixel space,
        # gradient through the final Euler step(s) + VAE decode. The b side
        # is meant to stay fixed, but weight updates leak through shared
        # tokens, so anchor v(b) to the frozen model.
        g_a = torch.Generator(device=ctx.backend.device).manual_seed(ctx.seed)
        img_a = ctx.backend.render(rule.a, g_a, cfg.pixel_render_steps,
                                   cfg.sample_guidance,
                                   grad_steps=cfg.pixel_grad_steps)
        g_b = torch.Generator(device=ctx.backend.device).manual_seed(ctx.seed)
        with torch.no_grad():
            img_b = ctx.backend.render(rule.b, g_b, cfg.pixel_render_steps,
                                       cfg.sample_guidance, frozen=True)
        anchor = ctx.vmse(ctx.v(rule.b, rule.b, frozen=False, grad=True),
                          ctx.v(rule.b, rule.b, frozen=True), rule.b)
        return F.mse_loss(img_a, img_b) + cfg.pixel_anchor_weight * anchor

    if rule.op == dsl.REWARD:
        raise NotImplementedError(
            "the ';' (ImageReward) op is not implemented in conceptmod 2.x")

    raise ValueError(f"unknown op {rule.op!r}")
