"""Optional ``c--`` erase modes (ESD / GEM / EraseAnything-inspired).

The live trainer still goes through :func:`conceptmod.ops.rule_loss`, which is
the original ESD velocity target. That path is what the SANA proofs used.
This module is the hook for DiT/RF-era variants so they can be tested on CPU
without rewriting ``ops.py`` or ``train.py`` (those files are also being
touched by parallel ``--backend cpu`` / Klein work).

Modes
-----
``esd``
    Current negatively-guided target
    ``v* = v('') - g (v(c) - v(''))``. Byte-for-byte the ESD branch in
    ``ops.rule_loss``.

``gem``
    GEM-inspired (Grebe et al., ICML'26, arXiv:2606.00140, Eqs. 13–14):
    contrastive velocity matching. Attract the trained erase-prompt
    velocity toward a *safe* field and repel it from the frozen erase
    field. ``L = relu(||v_t(c) - v_safe|| - η ||v_t(c) - v_f(c)||)``.
    Paper ``ĉ`` is a harmless rewording of ``c``, not an orthogonal keep
    concept — this hook builds ``v_safe`` with ESD reverse-CFG
    (paper Eq. 2 / uncond). ``erase_keep`` is a retain term only.
    Not a full GEM port: no multi-timestep trajectory window, no LoRA-on-
    Flux dual-stream Q/K.

``ea``
    EraseAnything-inspired (Gao et al., ICML'25,
    https://arxiv.org/abs/2412.20413): ESD plus a retain regularizer on a
    keep-concept probe. Velocity-space stand-in for EA's upper-level
    "do not harm ``D_ir``" term. Not a full EA port: no attention-map
    regularizer, no bi-level LoRA, no LLM-sampled ``D_ir`` / RSC features.

CLI hook (unwired on purpose; default remains ESD)::

    # train.py
    p.add_argument("--erase-mode", choices=list(ERASE_MODES), default="esd")
    # cfg.erase_mode = args.erase_mode
    # cfg.erase_keep = args.erase_keep   # optional keep / anchor prompt
    #
    # ops.rule_loss ERASE branch:
    #     from conceptmod.ops_erase import erase_loss
    #     return erase_loss(rule, ctx)
"""

from __future__ import annotations

import random as _random

import torch
import torch.nn.functional as F

from conceptmod import dsl, ops

ERASE_MODES = ("esd", "gem", "ea")

# GEM paper default for the repulsive scale η.
_GEM_ETA = 1.0
# EA retain weight when a keep prompt is set.
_EA_RETAIN = 1.0


def resolve_erase_mode(ctx: ops.StepContext | None = None,
                       mode: str | None = None) -> str:
    """``mode`` argument, else ``ctx.cfg.erase_mode``, else ``esd``."""
    if mode is None:
        mode = getattr(getattr(ctx, "cfg", None), "erase_mode", "esd")
    mode = str(mode).lower()
    if mode not in ERASE_MODES:
        raise ValueError(
            f"unknown erase mode {mode!r}; expected one of {ERASE_MODES}")
    return mode


def formatted_erase_prompt(rule: dsl.Rule, ctx: ops.StepContext) -> str:
    """Same template draw as ``ops.rule_loss`` (seeded by ``ctx.seed``)."""
    return _random.Random(ctx.seed).choice(ops.ERASE_TEMPLATES).format(rule.a)


def esd_target(v0: torch.Tensor, vc: torch.Tensor,
               guidance: float) -> torch.Tensor:
    """Negatively-guided ESD / EraseAnything lower-level target."""
    return v0 - guidance * (vc - v0)


def esd_loss(rule: dsl.Rule, ctx: ops.StepContext) -> torch.Tensor:
    """ESD velocity MSE. Must stay aligned with ``ops.rule_loss`` (ERASE)."""
    g = rule.options.get("guidance", ctx.cfg.erase_guidance)
    c = formatted_erase_prompt(rule, ctx)
    v0 = ctx.v("", c, frozen=True)
    vc = ctx.v(c, c, frozen=True)
    vt = ctx.v(c, c, frozen=False, grad=True)
    return F.mse_loss(vt, esd_target(v0, vc, g))


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).flatten(1).norm(dim=1).mean()


def gem_loss(rule: dsl.Rule, ctx: ops.StepContext) -> torch.Tensor:
    """GEM hinge (paper Eq. 14): attract to a *safe* field, repel from erase.

    Grebe et al. ICML'26 Eqs. 13–14:

        d_pos = ||v_t(c) − v_f(ĉ)||     # pull toward the teacher's safe anchor
        d_neg = ||v_t(c) − v_f(c)||     # push off the teacher's erase field
        L     = relu(d_pos − η d_neg)

    ``ĉ`` is a harmless rewording of the erase prompt (or ESD reverse-CFG
    when no explicit anchor exists — paper Eq. 2). It is **not** an
    orthogonal keep concept. The previous hook used ``erase_keep`` as
    ``ĉ``, which converted the erase prompt into the keep concept on the
    2-D fixture (leak ≈ +2.86). Keep, if set, is a retain MSE only —
    same role as EA / ``#``.

    Still missing vs the paper: the ``t ∈ {0..t_stop}`` trajectory window
    and LoRA on Flux dual-stream Q/K. This is the smallest velocity-space
    hinge that is geometrically an erase, not a convert-to-keep.
    """
    eta = getattr(ctx.cfg, "gem_eta", None)
    if eta is None:
        eta = rule.options.get("eta", _GEM_ETA)
    eta = float(eta)
    g = rule.options.get("guidance", ctx.cfg.erase_guidance)
    c = formatted_erase_prompt(rule, ctx)
    vt = ctx.v(c, c, frozen=False, grad=True)
    v0 = ctx.v("", c, frozen=True)
    vc = ctx.v(c, c, frozen=True)
    # Safe attractor = ESD reverse-CFG of uncond, never v(keep).
    v_safe = esd_target(v0, vc, float(g))
    loss = F.relu(_l2(vt, v_safe) - eta * _l2(vt, vc))
    keep = getattr(ctx.cfg, "erase_keep", "") or ""
    if keep:
        lam = float(getattr(ctx.cfg, "erase_retain", _EA_RETAIN))
        vf = ctx.v(keep, keep, frozen=True)
        vk = ctx.v(keep, keep, frozen=False, grad=True)
        loss = loss + lam * F.mse_loss(vk, vf)
    return loss


def ea_loss(rule: dsl.Rule, ctx: ops.StepContext) -> torch.Tensor:
    """ESD + keep-concept retain (EraseAnything-inspired, velocity only)."""
    loss = esd_loss(rule, ctx)
    keep = getattr(ctx.cfg, "erase_keep", "") or ""
    if not keep:
        return loss
    lam = float(getattr(ctx.cfg, "erase_retain", _EA_RETAIN))
    vf = ctx.v(keep, keep, frozen=True)
    vt = ctx.v(keep, keep, frozen=False, grad=True)
    return loss + lam * F.mse_loss(vt, vf)


def erase_loss(rule: dsl.Rule, ctx: ops.StepContext,
               mode: str | None = None) -> torch.Tensor:
    """Dispatch ``esd`` / ``gem`` / ``ea``. Default is ESD."""
    if rule.op != dsl.ERASE:
        raise ValueError(f"erase_loss expects an erase rule, got {rule.op!r}")
    mode = resolve_erase_mode(ctx, mode)
    if mode == "esd":
        return esd_loss(rule, ctx)
    if mode == "gem":
        return gem_loss(rule, ctx)
    return ea_loss(rule, ctx)


@torch.no_grad()
def concept_probe(backend, prompt: str, direction: torch.Tensor,
                  z: torch.Tensor, t: torch.Tensor,
                  frozen: bool = False) -> float:
    """How much of ``direction`` sits in the CFG velocity ``v(p) - v('')``.

    This is the CPU stand-in for an erase/keep probe on a real DiT: a drop
    on the erase concept with a smaller drop on the keep concept is the
    disentangle / preserve signal EraseAnything and GEM optimize for.
    """
    v = backend.predict_v(prompt, z, t, frozen=frozen)
    v0 = backend.predict_v("", z, t, frozen=frozen)
    delta = (v - v0).reshape(-1).float()
    d = direction.reshape(-1).float()
    denom = d.norm().clamp(min=1e-8)
    return torch.dot(delta, d / denom).item()
