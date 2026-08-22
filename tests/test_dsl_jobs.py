"""Geometric gates for phrase-DSL jobs on the CPU 2-D field.

These tests ask whether a *job* is already a phrase recipe (or a dead
op), not whether GEM/EA are wired right. Live ``ops.rule_loss`` only.
"""

from __future__ import annotations

import pytest

from conceptmod import dsl, ops
from conceptmod.analysis_2d import (
    COLOR,
    COLOR_OPP,
    KEEP,
    TwoAxisBackend,
    train_one,
    _probe_zt,
    plane_points,
)
from conceptmod.analysis_dsl import dead_ops, run_job, run_jobs
from conceptmod.backends import BACKENDS


def test_backends_tuple_unchanged():
    assert BACKENDS == ("sana", "zimage", "anima", "krea", "qwen", "cpu", "klein")


def test_neutralize_guidance_0_does_not_write_the_antipode():
    """ESD g=0 is 'erase without writing the opposite' — already a recipe."""
    result = run_job("neutralize", f"{COLOR}--:guidance=0")
    assert result.verdict == "right"
    assert abs(result.after.color_on_red) < 0.2
    assert result.after.write_cosine < 0.3
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25
    bx, by = result.points_after[COLOR_OPP]
    assert bx < 0.2
    assert abs(by) < 0.25


def test_mix_write_adds_pattern_without_a_plus_op():
    """`red=red stripe` is mix/add. A first-class `+` would be a synonym."""
    result = run_job("mix_write", f"{COLOR}={COLOR} {KEEP}")
    assert result.verdict == "right"
    assert result.after.color_on_red > 0.7
    assert result.after.pattern_on_red > 0.7
    assert result.after.stripe_hold > 0.85


def test_isolate_write_strips_color_from_the_mix():
    """`red stripe=stripe` isolates pattern. Not a new extract op."""
    result = run_job("isolate_write", f"{COLOR} {KEEP}={KEEP}")
    assert result.verdict == "right"
    mx, my = result.points_after[f"{COLOR} {KEEP}"]
    assert abs(mx) < 0.25
    assert my > 0.7


def test_orthogonal_on_already_perp_axes_is_noop():
    result = run_job("orthogonal_noop", f"{COLOR}%{KEEP}")
    assert result.verdict == "right"
    assert result.after.color_on_red == pytest.approx(result.before.color_on_red, abs=0.05)
    assert result.after.pattern_on_stripe == pytest.approx(
        result.before.pattern_on_stripe, abs=0.05,
    )


def test_negative_percent_cannot_blend_from_orthogonal():
    """|cos| at 0 has no gradient — blend from ⊥ is not a missing `+`."""
    result = run_job("blend_noop", f"{COLOR}%{KEEP}:-1|{KEEP}%{COLOR}:-1")
    assert result.verdict == "right"
    assert abs(result.after.pattern_on_red) < 0.05
    assert abs(result.after.color_on_stripe) < 0.05


def test_preserve_everything_except_is_erase_plus_freeze():
    result = run_job("preserve", f"{COLOR}--|#|{KEEP}#{KEEP}")
    assert result.verdict == "right"
    assert result.after.color_on_red < 0.2
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25


def test_bipolar_slider_is_exaggerate_on_antipodes():
    """`red++` stretches +color and, on this LoRA, −color with it."""
    backend, history, _ = train_one(f"{COLOR}++")
    z, t = _probe_zt(backend)
    after = history[-1]
    pts = plane_points(backend, z, t, False)
    assert after.color_on_red > 2.0
    bx, by = pts[COLOR_OPP]
    assert bx < -2.0
    assert abs(by) < 0.25
    assert after.stripe_hold > 0.85


def test_keep_plus_erase_already_exists():
    """`c--|k#k` is the composed keep+erase token the prompt asked about."""
    rules = dsl.parse_phrase(f"{COLOR}--|{KEEP}#{KEEP}")
    assert [r.op for r in rules] == [dsl.ERASE, dsl.FREEZE]
    assert (rules[1].a, rules[1].b) == (KEEP, KEEP)


def test_replace_is_a_macro_and_fights_on_antipodes():
    rules = dsl.parse_phrase(f"{COLOR}~{COLOR_OPP}")
    assert [r.op for r in rules] == [dsl.EXAGGERATE, dsl.WRITE, dsl.ORTHOGONAL]
    result = run_job("replace_macro", f"{COLOR}~{COLOR_OPP}")
    assert result.verdict == "recipe"
    # ++ wants to stretch blue; = wants to flip red onto blue. Linear
    # antipodes make those one motion. The landing is not a clean write.
    assert result.after.write_cosine < 0.7


def test_pixel_and_reward_are_dead_here():
    dead = dead_ops()
    assert "^" in dead and ";" in dead and "@" in dead
    assert not hasattr(TwoAxisBackend, "render")
    backend = TwoAxisBackend()
    ctx = ops.StepContext(backend, stop_index=2, seed=0, cfg=ops.OpDefaults())
    with pytest.raises(NotImplementedError):
        ops.rule_loss(dsl.parse_phrase(";ignored")[0], ctx)


def test_job_suite_verdicts_and_budget():
    results = run_jobs()
    by_name = {r.name: r for r in results}
    assert by_name["neutralize"].verdict == "right"
    assert by_name["mix_write"].verdict == "right"
    assert by_name["isolate_write"].verdict == "right"
    assert by_name["orthogonal_noop"].verdict == "right"
    assert by_name["blend_noop"].verdict == "right"
    assert by_name["preserve"].verdict == "right"
    assert by_name["replace_macro"].verdict == "recipe"
    assert sum(r.elapsed_s for r in results) < 15.0
