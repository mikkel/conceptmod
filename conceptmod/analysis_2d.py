"""CPU 2-D concept field for scoring DSL / erase geometry.

The 1-D ``cpu`` sample (``red=blue``) and the prompt-gated
``tests/test_erase_cpu.py`` fixture are scalar checks. This module is a
two-axis stand-in so a write or erase on one concept can be scored on an
orthogonal keep axis.

Axes
----
* color (x): ``red = +e_x``, ``blue = −e_x``
* pattern (y): ``stripe = +e_y``, ``dot = −e_y``

Embeddings are prompt-gated (a prompt mentioning only ``red`` does not
activate the pattern axis). The trainable residual is a *shared* rank-r
update on the class path — the same shape as LoRA on ``CpuBackend.linear2``.
Because the residual is shared, a method that looks correct on the target
cosine can still leak onto the keep axis. That is the point of the fixture:
``TwoConceptVelocity`` in ``test_erase_cpu`` cannot show this leak.

Losses are the live ones: :func:`conceptmod.ops.rule_loss` for DSL ops and
ESD, :func:`conceptmod.ops_erase.erase_loss` for the GEM / EA hooks. This
is not a second trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from conceptmod import dsl, ops
from conceptmod.ops_erase import erase_loss

LATENT_SHAPE = (4, 4, 4)
TEXT_DIM = 4
LORA_RANK = 2
COLOR = "red"
COLOR_OPP = "blue"
KEEP = "stripe"
KEEP_OPP = "dot"
PROBE_PROMPTS = (COLOR, COLOR_OPP, KEEP, KEEP_OPP, f"{COLOR} {KEEP}", "")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Bare-word templates so the 2-D story is the axes, not template coverage.
# Same pin as tests/test_erase_cpu.py.
_BARE_TEMPLATES = ["{}"]

# Cheap CPU budget: a few dozen Adam steps on a 64-d linear field.
DEFAULT_STEPS = 40
DEFAULT_LR = 8e-2
DEFAULT_SEED = 0
DEFAULT_GUIDANCE = 1.0
EXAGGERATE_GUIDANCE = 3.0


def prompt_axes(prompt: str) -> tuple[float, float]:
    """Map a prompt onto the (color, pattern) plane. Mixed / empty → 0."""
    tokens = set(_TOKEN_RE.findall((prompt or "").lower()))
    color = 0.0
    if COLOR in tokens and COLOR_OPP not in tokens:
        color = 1.0
    elif COLOR_OPP in tokens and COLOR not in tokens:
        color = -1.0
    pattern = 0.0
    if KEEP in tokens and KEEP_OPP not in tokens:
        pattern = 1.0
    elif KEEP_OPP in tokens and KEEP not in tokens:
        pattern = -1.0
    return color, pattern


class TwoAxisBackend:
    """Frozen orthonormal class path + shared LoRA residual.

    ``v(z, t, c) = 0.1 z + W e(c) + B A e(c)`` (trained). ``B`` starts at 0
    so trained == frozen at step 0. ``A`` is small Gaussian, so a nonzero
    ``B`` leaks onto every concept whose embedding has support on ``A``.
    """

    device = "cpu"
    latent_shape = LATENT_SHAPE

    def __init__(self, seed: int = DEFAULT_SEED, rank: int = LORA_RANK):
        torch.manual_seed(seed)
        z_dim = int(torch.tensor(LATENT_SHAPE).prod().item())
        self.d_color = torch.zeros(1, *LATENT_SHAPE)
        self.d_color[0, 0, 0, 0] = 1.0
        self.d_pattern = torch.zeros(1, *LATENT_SHAPE)
        self.d_pattern[0, 1, 0, 0] = 1.0
        w = torch.zeros(z_dim, TEXT_DIM)
        w[:, 0] = self.d_color.reshape(-1)
        w[:, 1] = self.d_pattern.reshape(-1)
        self.W = w  # frozen class path
        self.lora_A = torch.nn.Parameter(torch.randn(rank, TEXT_DIM) * 0.05)
        self.lora_B = torch.nn.Parameter(torch.zeros(z_dim, rank))

    def _embed(self, prompt: str) -> torch.Tensor:
        color, pattern = prompt_axes(prompt)
        e = torch.zeros(TEXT_DIM)
        e[0] = color
        e[1] = pattern
        return e

    def _class_v(self, prompt: str, trained: bool) -> torch.Tensor:
        e = self._embed(prompt)
        v = self.W @ e
        if trained:
            v = v + self.lora_B @ (self.lora_A @ e)
        return v.view(1, *LATENT_SHAPE)

    def predict_v(self, prompt, z, timestep, frozen):
        del timestep
        return 0.1 * z + self._class_v(prompt, trained=not frozen)

    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        del prompt, num_steps, guidance
        z = torch.randn(1, *self.latent_shape, generator=generator)
        return z, torch.tensor(float(stop_index))

    def trainable_parameters(self, train_method: str = "lora"):
        del train_method
        return [self.lora_A, self.lora_B]


def _probe_zt(backend: TwoAxisBackend, seed: int = DEFAULT_SEED):
    g = torch.Generator(device="cpu").manual_seed(seed + 17)
    z = torch.randn(1, *backend.latent_shape, generator=g)
    return z, torch.tensor(2.0)


@torch.no_grad()
def cfg_delta(backend: TwoAxisBackend, prompt: str, z, t, frozen: bool) -> torch.Tensor:
    """CFG direction ``v(p) − v('')`` with frozen uncond (same as cpu sample)."""
    v = backend.predict_v(prompt, z, t, frozen=frozen)
    v0 = backend.predict_v("", z, t, frozen=True)
    return v - v0


@torch.no_grad()
def project_axes(delta: torch.Tensor, backend: TwoAxisBackend) -> tuple[float, float]:
    d = delta.reshape(-1).float()
    return (
        torch.dot(d, backend.d_color.reshape(-1)).item(),
        torch.dot(d, backend.d_pattern.reshape(-1)).item(),
    )


@torch.no_grad()
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=1, eps=1e-6,
    ).item()


@dataclass
class ProbeSnapshot:
    step: int
    color_on_red: float
    pattern_on_stripe: float
    color_on_stripe: float
    pattern_on_red: float
    write_cosine: float
    stripe_hold: float
    red_norm: float
    stripe_norm: float


@dataclass
class MethodResult:
    name: str
    phrase: str
    erase_mode: str | None
    verdict: str
    note: str
    before: ProbeSnapshot
    after: ProbeSnapshot
    history: list[ProbeSnapshot] = field(default_factory=list)
    elapsed_s: float = 0.0
    points_before: dict[str, tuple[float, float]] = field(default_factory=dict)
    points_after: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        # Keep the JSON reviewable: first, last, and every 5th step.
        hist = [h.__dict__ for i, h in enumerate(self.history)
                if i == 0 or i == len(self.history) - 1 or h.step % 5 == 0]
        return {
            "name": self.name,
            "phrase": self.phrase,
            "erase_mode": self.erase_mode,
            "verdict": self.verdict,
            "note": self.note,
            "elapsed_s": self.elapsed_s,
            "before": self.before.__dict__,
            "after": self.after.__dict__,
            "history": hist,
            "points_before": {k: list(v) for k, v in self.points_before.items()},
            "points_after": {k: list(v) for k, v in self.points_after.items()},
        }


def _cfg(guidance: float = DEFAULT_GUIDANCE, mode: str = "esd",
         keep: str = KEEP, exaggerate: float | None = None) -> ops.OpDefaults:
    cfg = ops.OpDefaults(
        erase_guidance=guidance,
        write_guidance=guidance,
        exaggerate_guidance=exaggerate if exaggerate is not None else EXAGGERATE_GUIDANCE,
        sample_steps=4,
        sample_guidance=1.0,
        orthogonal_scale=1.0,
    )
    cfg.erase_mode = mode
    cfg.erase_keep = keep
    cfg.gem_eta = 1.0
    cfg.erase_retain = 1.0
    return cfg


def snapshot(backend: TwoAxisBackend, z, t, step: int) -> ProbeSnapshot:
    d_red = cfg_delta(backend, COLOR, z, t, frozen=False)
    d_blue_f = cfg_delta(backend, COLOR_OPP, z, t, frozen=True)
    d_stripe = cfg_delta(backend, KEEP, z, t, frozen=False)
    d_stripe_f = cfg_delta(backend, KEEP, z, t, frozen=True)
    cr, pr = project_axes(d_red, backend)
    cs, ps = project_axes(d_stripe, backend)
    return ProbeSnapshot(
        step=step,
        color_on_red=cr,
        pattern_on_stripe=ps,
        color_on_stripe=cs,
        pattern_on_red=pr,
        write_cosine=cosine(d_red, d_blue_f),
        stripe_hold=cosine(d_stripe, d_stripe_f),
        red_norm=d_red.norm().item(),
        stripe_norm=d_stripe.norm().item(),
    )


def plane_points(backend: TwoAxisBackend, z, t, frozen: bool) -> dict[str, tuple[float, float]]:
    out = {}
    for prompt in PROBE_PROMPTS:
        label = prompt if prompt else "∅"
        out[label] = project_axes(cfg_delta(backend, prompt, z, t, frozen), backend)
    return out


def _pin_bare_templates(monkey_ops=ops):
    """Return a restore callable. Used by the suite and by tests."""
    old_e, old_w = monkey_ops.ERASE_TEMPLATES, monkey_ops.WRITE_TEMPLATES
    monkey_ops.ERASE_TEMPLATES = list(_BARE_TEMPLATES)
    monkey_ops.WRITE_TEMPLATES = list(_BARE_TEMPLATES)

    def restore():
        monkey_ops.ERASE_TEMPLATES = old_e
        monkey_ops.WRITE_TEMPLATES = old_w

    return restore


def train_one(
    phrase: str,
    *,
    erase_mode: str | None = None,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    guidance: float = DEFAULT_GUIDANCE,
    keep: str = KEEP,
) -> tuple[TwoAxisBackend, list[ProbeSnapshot], float]:
    """SGD/Adam on the live ``rule_loss`` / ``erase_loss`` path."""
    restore = _pin_bare_templates()
    try:
        backend = TwoAxisBackend(seed=seed)
        z, t = _probe_zt(backend, seed)
        rules = dsl.parse_phrase(phrase)
        mode = erase_mode or "esd"
        exaggerate = EXAGGERATE_GUIDANCE if any(r.op == dsl.EXAGGERATE for r in rules) else None
        cfg = _cfg(guidance=guidance, mode=mode, keep=keep, exaggerate=exaggerate)
        opt = torch.optim.Adam(backend.trainable_parameters(), lr=lr)
        history = [snapshot(backend, z, t, 0)]
        t0 = time.time()
        for i in range(steps):
            ctx = ops.StepContext(backend, stop_index=2, seed=seed + i, cfg=cfg)
            for rule in rules:
                if erase_mode in ("gem", "ea") and rule.op == dsl.ERASE:
                    loss = rule.alpha * erase_loss(rule, ctx, mode=erase_mode)
                else:
                    loss = rule.alpha * ops.rule_loss(rule, ctx)
                loss.backward()
                ctx._v = {k: v for k, v in ctx._v.items() if not k[3]}
            torch.nn.utils.clip_grad_norm_(backend.trainable_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            history.append(snapshot(backend, z, t, i + 1))
        return backend, history, time.time() - t0
    finally:
        restore()


def _verdict_for(name: str, before: ProbeSnapshot, after: ProbeSnapshot) -> tuple[str, str]:
    """Geometric verdict from probes. Thresholds are the claims tests gate."""
    color_move = after.color_on_red - before.color_on_red
    keep_drop = before.pattern_on_stripe - after.pattern_on_stripe
    keep_ok = after.stripe_hold > 0.85 and abs(after.color_on_stripe) < 0.25
    leak_on_red = abs(after.pattern_on_red) > 0.25

    if name == "write":
        if after.write_cosine > 0.7 and keep_ok and not leak_on_red:
            return "right", (
                "Trained red-direction aligns with frozen blue; stripe hold stays high."
            )
        if after.write_cosine > 0.7 and (not keep_ok or leak_on_red):
            return "needs help", (
                "Write hits the color target but leaks onto the pattern axis."
            )
        return "needs help", "Write failed to align red with blue."

    if name == "exaggerate":
        if color_move > 1.0 and keep_ok and not leak_on_red:
            return "right", (
                "Classic ++ stretches the color axis past the frozen red "
                "direction without dragging stripe."
            )
        if color_move > 1.0:
            return "needs help", "++ grew red but leaked onto the keep axis."
        return "needs help", "++ failed to amplify the color probe."

    if name in ("erase_esd", "erase_ea", "erase_esd_freeze"):
        erased = after.color_on_red < 0.2 and color_move < -0.5
        if name == "erase_esd":
            if erased and keep_ok and not leak_on_red:
                return "right", (
                    "Live ESD flips the red CFG toward the negatively-guided "
                    "target; stripe hold stays high on this LoRA."
                )
            if erased and (not keep_ok or leak_on_red):
                return "needs help", (
                    "ESD moves red but the shared LoRA leaks onto stripe. "
                    "A keep anchor (# or EA retain) is the next knob."
                )
            return "needs help", "ESD failed to move the red probe."
        if name == "erase_ea":
            if erased and keep_ok and not leak_on_red:
                better = after.stripe_hold >= before.pattern_on_stripe - 0.05
                extra = (
                    " Retain matches a `#` freeze on this velocity-only hook."
                    if better else ""
                )
                return "right", (
                    "EA = ESD + keep retain: red is erased and stripe stays put."
                    + extra
                )
            if erased:
                return "needs help", (
                    "EA erased red but the retain term was too thin to hold stripe."
                )
            return "needs help", "EA failed to erase red."
        # erase_esd_freeze
        if erased and keep_ok:
            return "right", (
                "ESD plus `stripe#stripe` is the live-DSL way to pin the keep "
                "axis. Same geometry as the EA retain hook on this fixture."
            )
        if erased:
            return "needs help", "Freeze did not hold the stripe axis."
        return "needs help", "Protected erase failed to move red."

    if name == "erase_gem":
        pulled_to_keep = after.pattern_on_red > 0.4 and after.color_on_red < 0.6
        if pulled_to_keep:
            return "needs help", (
                "GEM still converts red into stripe — the safe attractor "
                "is wrong (keep must not be ĉ)."
            )
        if after.color_on_red < 0.2 and keep_ok and not leak_on_red:
            return "right", (
                "GEM hinge attracts toward the ESD safe field (uncond "
                "reverse-CFG), not toward stripe. Keep is a retain term. "
                "Erase-axis drops; stripe hold stays high."
            )
        if after.color_on_red < 0.2:
            return "needs help", "GEM erased red but leaked onto the keep axis."
        return "needs help", (
            "GEM did not produce a clean erase; the contrastive hinge is "
            "under-specified for this field."
        )

    return "needs help", "Unknown method."


def run_method(name: str, phrase: str, erase_mode: str | None = None,
               **kwargs) -> MethodResult:
    backend, history, elapsed = train_one(phrase, erase_mode=erase_mode, **kwargs)
    z, t = _probe_zt(backend, kwargs.get("seed", DEFAULT_SEED))
    before, after = history[0], history[-1]
    verdict, note = _verdict_for(name, before, after)
    return MethodResult(
        name=name,
        phrase=phrase,
        erase_mode=erase_mode,
        verdict=verdict,
        note=note,
        before=before,
        after=after,
        history=history,
        elapsed_s=elapsed,
        points_before=plane_points(TwoAxisBackend(seed=kwargs.get("seed", DEFAULT_SEED)), z, t, True),
        points_after=plane_points(backend, z, t, False),
    )


def method_specs() -> list[dict]:
    return [
        {"name": "write", "phrase": f"{COLOR}={COLOR_OPP}", "erase_mode": None},
        {"name": "erase_esd", "phrase": f"{COLOR}--", "erase_mode": None},
        {"name": "erase_gem", "phrase": f"{COLOR}--", "erase_mode": "gem"},
        {"name": "erase_ea", "phrase": f"{COLOR}--", "erase_mode": "ea"},
        {"name": "exaggerate", "phrase": f"{COLOR}++", "erase_mode": None},
        {"name": "erase_esd_freeze", "phrase": f"{COLOR}--|{KEEP}#{KEEP}",
         "erase_mode": None},
    ]


def run_suite(steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
              seed: int = DEFAULT_SEED) -> list[MethodResult]:
    results = []
    for spec in method_specs():
        results.append(run_method(steps=steps, lr=lr, seed=seed, **spec))
    return results


# ---------------------------------------------------------------------------
# plots + report artifacts
# ---------------------------------------------------------------------------

METHOD_TITLES = {
    "write": "write  red=blue",
    "erase_esd": "erase ESD  red--",
    "erase_gem": "erase GEM  red--  keep=stripe",
    "erase_ea": "erase EA  red--  keep=stripe",
    "exaggerate": "exaggerate  red++",
    "erase_esd_freeze": "erase ESD + freeze  red--|stripe#stripe",
}

METHOD_COLORS = {
    "write": "#1f77b4",
    "erase_esd": "#d62728",
    "erase_gem": "#ff7f0e",
    "erase_ea": "#2ca02c",
    "exaggerate": "#9467bd",
    "erase_esd_freeze": "#17becf",
}


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to write 2-D analysis plots"
        ) from exc
    return plt


def plot_quiver(results: list[MethodResult], path: Path) -> None:
    plt = _require_matplotlib()
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes = axes.ravel()
    prompt_colors = {
        COLOR: "#c0392b",
        COLOR_OPP: "#2980b9",
        KEEP: "#8e44ad",
        KEEP_OPP: "#16a085",
        f"{COLOR} {KEEP}": "#e67e22",
        "∅": "#7f8c8d",
    }
    for ax, result in zip(axes, results):
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.axvline(0, color="#cccccc", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        for prompt, (x0, y0) in result.points_before.items():
            x1, y1 = result.points_after[prompt]
            color = prompt_colors.get(prompt, "#333333")
            ax.scatter([x0], [y0], c=color, s=28, zorder=3, marker="o")
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
            )
            ax.scatter([x1], [y1], c=color, s=36, zorder=4, marker="s")
            ax.text(x1 + 0.08, y1 + 0.08, prompt, fontsize=8, color=color)
        ax.set_xlabel("color  (red → +)")
        ax.set_ylabel("pattern  (stripe → +)")
        badge = "right" if result.verdict == "right" else "needs help"
        ax.set_title(f"{METHOD_TITLES[result.name]}\n{badge}", fontsize=10)
        lim = 4.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(
        "Frozen ○ → trained □  CFG deltas in the 2-D concept plane",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_trajectories(results: list[MethodResult], path: Path) -> None:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for result in results:
        steps = [h.step for h in result.history]
        c = METHOD_COLORS[result.name]
        label = METHOD_TITLES[result.name]
        axes[0].plot(steps, [h.color_on_red for h in result.history],
                     color=c, label=label, lw=1.8)
        axes[1].plot(steps, [h.pattern_on_stripe for h in result.history],
                     color=c, label=label, lw=1.8)
        axes[2].plot(steps, [h.pattern_on_red for h in result.history],
                     color=c, label=label, lw=1.8)
    axes[0].axhline(0.0, color="#999999", ls="--", lw=0.8)
    axes[0].set_title("target  ⟨CFG(red), e_color⟩")
    axes[0].set_ylabel("color probe on red")
    axes[1].axhline(1.0, color="#999999", ls="--", lw=0.8)
    axes[1].set_title("keep prompt  ⟨CFG(stripe), e_pattern⟩")
    axes[1].set_ylabel("pattern probe on stripe")
    axes[2].axhline(0.0, color="#999999", ls="--", lw=0.8)
    axes[2].set_title("leak  ⟨CFG(red), e_pattern⟩")
    axes[2].set_ylabel("pattern component of red")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Probe trajectories  (shared LoRA residual — leakage is possible)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_table(results: list[MethodResult], path: Path) -> None:
    plt = _require_matplotlib()
    cols = [
        "method", "verdict", "red color", "stripe hold",
        "red→blue cos", "leak ⟨red, e_y⟩",
    ]
    cells = []
    colors = []
    for r in results:
        cells.append([
            METHOD_TITLES[r.name],
            r.verdict,
            f"{r.after.color_on_red:+.3f}",
            f"{r.after.stripe_hold:+.3f}",
            f"{r.after.write_cosine:+.3f}",
            f"{r.after.pattern_on_red:+.3f}",
        ])
        colors.append(
            "#d5f5e3" if r.verdict == "right" else "#fadbd8"
        )
    fig, ax = plt.subplots(figsize=(11.2, 0.55 * (len(results) + 2)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif col == 1:
            cell.set_facecolor(colors[row - 1])
    ax.set_title("2-D CPU geometry  (after training)", pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_metrics(results: list[MethodResult], path: Path) -> None:
    payload = {
        "fixture": {
            "color_axis": f"{COLOR}=+e_x / {COLOR_OPP}=−e_x",
            "pattern_axis": f"{KEEP}=+e_y / {KEEP_OPP}=−e_y",
            "residual": "shared rank-2 LoRA on the class path",
            "losses": "ops.rule_loss (DSL + ESD) / ops_erase.erase_loss (GEM, EA)",
            "steps": results[0].history[-1].step if results else 0,
        },
        "methods": [r.as_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


DEFAULT_OUT = Path("outputs/2d_analysis")


def write_artifacts(results: list[MethodResult], out: Path = DEFAULT_OUT) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "quiver": out / "quiver.png",
        "trajectories": out / "trajectories.png",
        "table": out / "table.png",
        "metrics": out / "metrics.json",
    }
    plot_quiver(results, paths["quiver"])
    plot_trajectories(results, paths["trajectories"])
    plot_table(results, paths["table"])
    write_metrics(results, paths["metrics"])
    return paths
