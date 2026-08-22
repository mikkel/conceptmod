"""2-D scores for phrase-DSL *jobs* (recipes vs first-class ops).

The erase/GEM suite in :mod:`conceptmod.analysis_2d` asks whether each
*method* moves the target axis. This module asks a different question:
which useful geometric jobs can a user already state, and which would
need a new operator?

Every job here goes through live :func:`conceptmod.ops.rule_loss`. No
``ops_erase`` import — GEM/EA stay on the other suite. Pixel ``^`` is
not expressible (the fixture has no renderer). ``;`` stays unimplemented.
"""

from __future__ import annotations

from pathlib import Path

from conceptmod.analysis_2d import (
    COLOR,
    COLOR_OPP,
    DEFAULT_LR,
    DEFAULT_OUT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    KEEP,
    MethodResult,
    plot_quiver,
    plot_table,
    run_method,
    write_metrics,
)


# ---------------------------------------------------------------------------
# job specs — phrases only, no erase_mode hooks
# ---------------------------------------------------------------------------

def job_specs() -> list[dict]:
    """Jobs with a clean 2-D story (or a documented no-op / fixture fight)."""
    return [
        {"name": "neutralize", "phrase": f"{COLOR}--:guidance=0"},
        {"name": "mix_write", "phrase": f"{COLOR}={COLOR} {KEEP}"},
        {"name": "isolate_write", "phrase": f"{COLOR} {KEEP}={KEEP}"},
        {"name": "orthogonal_noop", "phrase": f"{COLOR}%{KEEP}"},
        {"name": "blend_noop", "phrase": f"{COLOR}%{KEEP}:-1|{KEEP}%{COLOR}:-1"},
        {"name": "preserve", "phrase": f"{COLOR}--|#|{KEEP}#{KEEP}"},
        {"name": "replace_macro", "phrase": f"{COLOR}~{COLOR_OPP}"},
    ]


JOB_TITLES = {
    "neutralize": "neutralize  red--:guidance=0",
    "mix_write": "mix  red=red stripe",
    "isolate_write": "isolate  red stripe=stripe",
    "orthogonal_noop": "orthogonal  red%stripe",
    "blend_noop": "blend  red%stripe:-1|stripe%red:-1",
    "preserve": "preserve-except  red--|#|stripe#stripe",
    "replace_macro": "replace macro  red~blue",
}

JOB_COLORS = {
    "neutralize": "#1abc9c",
    "mix_write": "#e67e22",
    "isolate_write": "#8e44ad",
    "orthogonal_noop": "#7f8c8d",
    "blend_noop": "#95a5a6",
    "preserve": "#17becf",
    "replace_macro": "#c0392b",
}


def _job_verdict(name: str, before, after, points_after: dict) -> tuple[str, str]:
    keep_ok = after.stripe_hold > 0.85 and abs(after.color_on_stripe) < 0.25
    leak_on_red = abs(after.pattern_on_red) > 0.25
    unchanged = (
        abs(after.color_on_red - before.color_on_red) < 0.05
        and abs(after.pattern_on_stripe - before.pattern_on_stripe) < 0.05
        and abs(after.pattern_on_red - before.pattern_on_red) < 0.05
        and abs(after.color_on_stripe - before.color_on_stripe) < 0.05
    )

    if name == "neutralize":
        at_origin = abs(after.color_on_red) < 0.2
        not_antipode = after.write_cosine < 0.3
        if at_origin and not_antipode and keep_ok and not leak_on_red:
            return "right", (
                "ESD guidance 0 maps red to uncond (origin), not to blue. "
                "Already a phrase recipe — not a new operator."
            )
        return "needs help", "guidance=0 did not neutralize red to the origin."

    if name == "mix_write":
        mixed = after.color_on_red > 0.7 and after.pattern_on_red > 0.7
        if mixed and after.stripe_hold > 0.85:
            return "right", (
                "Write red toward the mixed embedding: red keeps color and "
                "picks up stripe. First-class `+` would be a synonym here."
            )
        return "needs help", "Write-to-concat did not add the pattern axis."

    if name == "isolate_write":
        mx, my = points_after.get(f"{COLOR} {KEEP}", (None, None))
        isolated = mx is not None and abs(mx) < 0.25 and my > 0.7
        if isolated:
            return "right", (
                "Write the mix onto stripe: `red stripe` lands on e_y. "
                "Standalone red/stripe drift is the linear LoRA class, "
                "same caveat as the antipodal write swap."
            )
        return "needs help", "Write-to-keep did not strip color from the mix."

    if name in ("orthogonal_noop", "blend_noop"):
        if unchanged:
            extra = (
                "Axes already start orthogonal, so |cos| is 0 and the "
                "gradient vanishes. Negative % cannot *create* a mix "
                "from a perpendicular pair — use `red=red stripe`."
                if name == "blend_noop"
                else "Axes already start orthogonal, so the |cos| loss is 0."
            )
            return "right", extra
        return "needs help", "% moved a pair that should have been a no-op."

    if name == "preserve":
        erased = after.color_on_red < 0.2
        if erased and keep_ok and not leak_on_red:
            return "right", (
                "Erase + freeze uncond + freeze keep. `#` on ∅ is a no-op "
                "on this e=0 uncond; `stripe#stripe` is the live retain. "
                "Same keep geometry as `red--|stripe#stripe`."
            )
        if erased:
            return "needs help", "Preserve recipe erased red but missed the keep pin."
        return "needs help", "Preserve recipe failed to erase red."

    if name == "replace_macro":
        # On a linear map of antipodes, blue++ and red=blue fight.
        return "recipe", (
            "`~` expands to `blue++ | red=blue | blue%red:-λ`. On this "
            "linear antipodal LoRA the ++ and = terms oppose each other, "
            "so the macro is not an independent geometric job. Use `=` "
            "or `++` alone when you want a clean 2-D story."
        )

    return "needs help", "Unknown DSL job."


def run_job(name: str, phrase: str, **kwargs) -> MethodResult:
    """Train one phrase and attach a job-specific verdict."""
    # Isolate needs the mixed-prompt landing; run_method's generic
    # _verdict_for does not know these names, so we overwrite.
    result = run_method(name, phrase, **kwargs)
    verdict, note = _job_verdict(name, result.before, result.after, result.points_after)
    result.verdict = verdict
    result.note = note
    return result


def run_jobs(steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
             seed: int = DEFAULT_SEED) -> list[MethodResult]:
    return [run_job(steps=steps, lr=lr, seed=seed, **spec) for spec in job_specs()]


# Reuse the 2-D plotters by temporarily registering titles/colors.
def write_job_artifacts(results: list[MethodResult], out: Path = DEFAULT_OUT) -> dict[str, Path]:
    from conceptmod import analysis_2d as a2

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old_titles, old_colors = a2.METHOD_TITLES, a2.METHOD_COLORS
    a2.METHOD_TITLES = {**old_titles, **JOB_TITLES}
    a2.METHOD_COLORS = {**old_colors, **JOB_COLORS}
    try:
        paths = {
            "quiver": out / "dsl_jobs_quiver.png",
            "table": out / "dsl_jobs_table.png",
            "metrics": out / "dsl_jobs.json",
        }
        # `~` is a documented fixture fight, not a quiver panel.
        pictured = [r for r in results if r.name != "replace_macro"]
        plot_quiver(pictured, paths["quiver"])
        plot_table(pictured, paths["table"])
        write_metrics(results, paths["metrics"])
        return paths
    finally:
        a2.METHOD_TITLES = old_titles
        a2.METHOD_COLORS = old_colors


def dead_ops() -> dict[str, str]:
    """Ops that do not have a 2-D velocity-space story on this fixture."""
    return {
        "^": (
            "Pixel L2 between full renders. TwoAxisBackend has no `render`, "
            "and the fixture is a velocity field, not pixels."
        ),
        ";": "ImageReward. Parsed, not implemented. Needs a GPU scorer — leave it.",
        "@": "Deprecated. Stripped and ignored at parse time.",
        "=b / # on ∅": (
            "Uncond write/freeze. e('') is the zero vector, so the shared "
            "LoRA B A e cannot move uncond on this fixture."
        ),
    }
