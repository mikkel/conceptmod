"""Geometric gates for the CPU 2-D analysis suite.

Pixels are not asserted — probes and cosines are. A claimed ``right``
method that leaks onto stripe or fails to move red must fail here.
"""

from __future__ import annotations

import pytest

from conceptmod.analysis_2d import (
    COLOR,
    COLOR_OPP,
    KEEP,
    KEEP_OPP,
    TwoAxisBackend,
    cfg_delta,
    cosine,
    plane_points,
    project_axes,
    prompt_axes,
    run_method,
    run_suite,
)
from conceptmod.backends import BACKENDS


def _zt(backend, seed=0):
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed + 17)
    z = torch.randn(1, *backend.latent_shape, generator=g)
    return z, torch.tensor(2.0)


def test_backends_tuple_unchanged():
    assert BACKENDS == ("sana", "zimage", "anima", "krea", "qwen", "cpu", "klein")


def test_prompt_axes_are_gated():
    assert prompt_axes("red") == (1.0, 0.0)
    assert prompt_axes("a photo of a blue") == (-1.0, 0.0)
    assert prompt_axes("stripe") == (0.0, 1.0)
    assert prompt_axes("dot style") == (0.0, -1.0)
    assert prompt_axes("red stripe") == (1.0, 1.0)
    assert prompt_axes("") == (0.0, 0.0)
    assert prompt_axes("a cat sitting on a chair") == (0.0, 0.0)
    assert prompt_axes("red blue") == (0.0, 0.0)


def test_frozen_axes_are_orthonormal():
    backend = TwoAxisBackend(seed=0)
    z, t = _zt(backend)
    d_red = cfg_delta(backend, COLOR, z, t, frozen=True)
    d_blue = cfg_delta(backend, COLOR_OPP, z, t, frozen=True)
    d_stripe = cfg_delta(backend, KEEP, z, t, frozen=True)
    d_dot = cfg_delta(backend, KEEP_OPP, z, t, frozen=True)
    assert cosine(d_red, d_stripe) == pytest.approx(0.0, abs=1e-5)
    assert cosine(d_red, d_blue) == pytest.approx(-1.0, abs=1e-5)
    assert cosine(d_stripe, d_dot) == pytest.approx(-1.0, abs=1e-5)
    cr, pr = project_axes(d_red, backend)
    cs, ps = project_axes(d_stripe, backend)
    assert (cr, pr) == pytest.approx((1.0, 0.0), abs=1e-5)
    assert (cs, ps) == pytest.approx((0.0, 1.0), abs=1e-5)
    points = plane_points(backend, z, t, frozen=True)
    assert points["red"] == pytest.approx((1.0, 0.0), abs=1e-5)
    assert points[KEEP] == pytest.approx((0.0, 1.0), abs=1e-5)


def test_write_aligns_red_with_blue_and_holds_stripe():
    result = run_method("write", f"{COLOR}={COLOR_OPP}")
    assert result.verdict == "right"
    assert result.after.write_cosine > 0.7
    assert result.after.color_on_red < 0.0
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25
    assert abs(result.after.color_on_stripe) < 0.25


def test_linear_lora_write_also_flips_the_antipode():
    """Linear class-path LoRA on red = −blue cannot remap without swapping.

    The write *loss* only trains prompt ``red``; the function class maps
    ``e_blue = −e_red`` so blue comes along. That is a fixture fact, not a
    pass for remap-only write on a real DiT.
    """
    result = run_method("write", f"{COLOR}={COLOR_OPP}")
    bx, by = result.points_after[COLOR_OPP]
    assert bx > 0.5
    assert abs(by) < 0.25


def test_esd_erases_red_without_killing_stripe():
    result = run_method("erase_esd", f"{COLOR}--")
    assert result.verdict == "right"
    assert result.after.color_on_red < 0.2
    assert result.before.color_on_red - result.after.color_on_red > 0.5
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25


def test_g1_esd_matches_write_target_on_antipodes():
    """ESD g=1 target is −CFG(red) = CFG(blue): same as ``red=blue`` here."""
    write = run_method("write", f"{COLOR}={COLOR_OPP}")
    esd = run_method("erase_esd", f"{COLOR}--")
    assert write.after.color_on_red == pytest.approx(esd.after.color_on_red, abs=1e-4)
    assert write.after.write_cosine == pytest.approx(esd.after.write_cosine, abs=1e-4)


def test_exaggerate_grows_color_not_pattern():
    result = run_method("exaggerate", f"{COLOR}++")
    assert result.verdict == "right"
    assert result.after.color_on_red > 2.0
    assert result.after.color_on_red > result.before.color_on_red + 1.0
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25


def test_ea_holds_keep_at_least_as_well_as_esd():
    esd = run_method("erase_esd", f"{COLOR}--")
    ea = run_method("erase_ea", f"{COLOR}--", erase_mode="ea")
    assert ea.verdict == "right"
    assert ea.after.color_on_red < 0.2
    assert ea.after.stripe_hold >= esd.after.stripe_hold - 1e-4
    assert abs(ea.after.color_on_stripe) <= abs(esd.after.color_on_stripe) + 0.05


def test_esd_plus_freeze_matches_ea_keep():
    freeze = run_method("erase_esd_freeze", f"{COLOR}--|{KEEP}#{KEEP}")
    ea = run_method("erase_ea", f"{COLOR}--", erase_mode="ea")
    assert freeze.verdict == "right"
    assert freeze.after.stripe_hold == pytest.approx(ea.after.stripe_hold, abs=5e-3)
    assert abs(freeze.after.pattern_on_red) < 0.25


def test_gem_erases_red_without_becoming_stripe():
    """GEM must drop the erase axis and not convert red into stripe."""
    result = run_method("erase_gem", f"{COLOR}--", erase_mode="gem")
    assert result.verdict == "right"
    assert result.after.color_on_red < 0.2
    assert result.before.color_on_red - result.after.color_on_red > 0.5
    assert result.after.stripe_hold > 0.85
    assert abs(result.after.pattern_on_red) < 0.25


def test_suite_verdicts_and_budget():
    results = run_suite()
    by_name = {r.name: r for r in results}
    assert by_name["write"].verdict == "right"
    assert by_name["erase_esd"].verdict == "right"
    assert by_name["erase_ea"].verdict == "right"
    assert by_name["exaggerate"].verdict == "right"
    assert by_name["erase_esd_freeze"].verdict == "right"
    assert by_name["erase_gem"].verdict == "right"
    assert sum(r.elapsed_s for r in results) < 30.0
