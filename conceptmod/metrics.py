"""Fixed-probe edit metrics: when is a phrase actually done?

Training loss is the wrong stop signal. It is computed on a fresh random
``z_ctx`` every step, starts near zero, and stays there whether or not the
edit has taken. These scores reuse the same velocity geometry as the losses
but on a *fixed* set of (seed, stop-index) probes so they are comparable
across steps.

Headline number for a write / composite (``a=b``):

    swap_ratio = ||v_t(a) − v*(b)|| / ||v_f(a) − v*(b)||

1.0 at init (trained == frozen), 0.0 when the write target is hit.
Stop when it drops below ``swap_stop`` and the freeze *hold* has not blown up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from conceptmod import dsl, ops


@dataclass
class RuleScore:
    raw: str
    op: str
    name: str
    value: float
    done: bool


@dataclass
class EditReport:
    swap_ratio: float | None
    hold: float | None
    remain: float | None
    scores: list[RuleScore] = field(default_factory=list)
    done: bool = False

    def summary(self) -> str:
        parts = []
        if self.swap_ratio is not None:
            parts.append(f"swap_ratio={self.swap_ratio:.3f}")
        if self.hold is not None:
            parts.append(f"hold={self.hold:.3f}")
        if self.remain is not None:
            parts.append(f"remain={self.remain:.3f}")
        for s in self.scores:
            if s.name not in ("swap_ratio", "hold", "remain"):
                parts.append(f"{s.raw}:{s.name}={s.value:.3f}")
        parts.append("DONE" if self.done else "train")
        return " ".join(parts)


def _mse(a, b) -> float:
    return F.mse_loss(a, b).item()


@torch.no_grad()
def _probes(backend, cfg: ops.OpDefaults, seeds, stop_fracs):
    steps = max(int(cfg.sample_steps), 2)
    stops = sorted({max(1, min(steps - 1, int(steps * f))) for f in stop_fracs})
    out = []
    for seed in seeds:
        for stop in stops:
            g = torch.Generator(device=backend.device).manual_seed(int(seed) + 17 * stop)
            z, t = backend.partial_denoise("", stop, steps, 1.0, g)
            out.append((z, t))
    return out


@torch.no_grad()
def eval_phrase(
    backend,
    phrase: str,
    cfg: ops.OpDefaults | None = None,
    seeds=(0, 1),
    stop_fracs=(0.35, 0.7),
    swap_stop: float = 0.35,
    hold_max: float = 0.25,
    remain_stop: float = 0.40,
) -> EditReport:
    """Score every rule in ``phrase`` on a frozen probe set."""
    cfg = cfg or ops.OpDefaults()
    rules = [r for r in dsl.parse_phrase(phrase)
             if r.op != dsl.REWARD and not r.needs_random_prompt]
    probes = _probes(backend, cfg, seeds, stop_fracs)
    scores: list[RuleScore] = []

    was_training = False
    if hasattr(backend, "transformer"):
        was_training = bool(backend.transformer.training)
        backend.transformer.eval()

    try:
        for rule in rules:
            scores.append(_score_rule(backend, rule, cfg, probes,
                                      swap_stop, hold_max, remain_stop))
    finally:
        if was_training and hasattr(backend, "transformer"):
            backend.transformer.train()

    swaps = [s.value for s in scores if s.name == "swap_ratio"]
    holds = [s.value for s in scores if s.name == "hold"]
    remains = [s.value for s in scores if s.name == "remain"]
    report = EditReport(
        swap_ratio=max(swaps) if swaps else None,
        hold=max(holds) if holds else None,
        remain=max(remains) if remains else None,
        scores=scores,
    )
    stoppable = [s for s in scores if s.name in ("swap_ratio", "remain")]
    if stoppable:
        report.done = all(s.done for s in stoppable) and all(
            s.done for s in scores if s.name == "hold")
    return report


def _score_rule(backend, rule, cfg, probes, swap_stop, hold_max, remain_stop) -> RuleScore:
    if rule.op == dsl.WRITE:
        g = float(rule.options.get("guidance", cfg.write_guidance))
        nums, dens = [], []
        for z, t in probes:
            v0 = backend.predict_v("", z, t, frozen=True)
            vb = backend.predict_v(rule.b, z, t, frozen=True)
            target = v0 + g * (vb - v0)
            vt = backend.predict_v(rule.a, z, t, frozen=False)
            vf = backend.predict_v(rule.a, z, t, frozen=True)
            nums.append(_mse(vt, target))
            dens.append(_mse(vf, target))
        value = sum(nums) / max(sum(dens), 1e-8)
        return RuleScore(rule.raw, rule.op, "swap_ratio", value, value <= swap_stop)

    if rule.op == dsl.FREEZE:
        holds = []
        for z, t in probes:
            target = backend.predict_v(rule.b, z, t, frozen=True)
            vt = backend.predict_v(rule.a, z, t, frozen=False)
            scale = target.pow(2).mean().clamp(min=1e-8).item()
            holds.append(_mse(vt, target) / scale)
        value = sum(holds) / len(holds)
        return RuleScore(rule.raw, rule.op, "hold", value, value <= hold_max)

    if rule.op == dsl.ERASE:
        g = float(rule.options.get("guidance", cfg.erase_guidance))
        nums, dens = [], []
        for z, t in probes:
            v0 = backend.predict_v("", z, t, frozen=True)
            vc = backend.predict_v(rule.a, z, t, frozen=True)
            target = v0 - g * (vc - v0)
            vt = backend.predict_v(rule.a, z, t, frozen=False)
            nums.append(_mse(vt, target))
            dens.append(_mse(vc, target))
        value = sum(nums) / max(sum(dens), 1e-8)
        return RuleScore(rule.raw, rule.op, "remain", value, value <= remain_stop)

    if rule.op == dsl.EXAGGERATE:
        g = float(rule.options.get("guidance", cfg.exaggerate_guidance))
        aligns = []
        for z, t in probes:
            v0 = backend.predict_v("", z, t, frozen=True)
            vc = backend.predict_v(rule.a, z, t, frozen=True)
            vt = backend.predict_v(rule.a, z, t, frozen=False)
            direction = (vc - v0).flatten()
            moved = (vt - vc).flatten()
            denom = direction.norm().clamp(min=1e-8)
            aligns.append(torch.dot(moved, direction / denom).item() / (
                denom.item() * max(g, 1e-8)))
        value = sum(aligns) / len(aligns)
        # not a stop signal: more boost is not "done"
        return RuleScore(rule.raw, rule.op, "boost", value, False)

    if rule.op == dsl.ORTHOGONAL:
        cosines = []
        for z, t in probes:
            da = (backend.predict_v(rule.a, z, t, frozen=True)
                  - backend.predict_v("", z, t, frozen=True)).flatten(1)
            db = (backend.predict_v(rule.b, z, t, frozen=False)
                  - backend.predict_v("", z, t, frozen=False)).flatten(1)
            cosines.append(F.cosine_similarity(da, db, dim=1, eps=1e-6).abs().mean().item())
        value = sum(cosines) / len(cosines)
        return RuleScore(rule.raw, rule.op, "cosine", value, False)

    return RuleScore(rule.raw, rule.op, "skip", 0.0, True)
