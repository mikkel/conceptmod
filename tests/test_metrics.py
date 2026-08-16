import torch
import pytest

from conceptmod import metrics, ops
from tests.test_ops import MockBackend


def _report(backend, phrase, **kw):
    cfg = ops.OpDefaults(sample_steps=4, write_guidance=2.0)
    return metrics.eval_phrase(
        backend, phrase, cfg, seeds=(0,), stop_fracs=(0.5,), **kw)


def test_swap_ratio_is_one_before_any_edit():
    report = _report(MockBackend(), "cat=dog")
    assert report.swap_ratio == pytest.approx(1.0, abs=1e-5)
    assert report.done is False


def test_swap_ratio_hits_zero_when_write_target_is_met():
    backend = MockBackend()
    cfg = ops.OpDefaults(sample_steps=4, write_guidance=2.0)
    stop = 2
    g = torch.Generator(device="cpu").manual_seed(0 + 17 * stop)
    z, t = backend.partial_denoise("", stop, 4, 1.0, g)
    v0 = backend.predict_v("", z, t, frozen=True)
    vb = backend.predict_v("dog", z, t, frozen=True)
    vf = backend.predict_v("cat", z, t, frozen=True)
    target = v0 + 2.0 * (vb - v0)
    backend.delta.data = (target - vf)
    report = metrics.eval_phrase(
        backend, "cat=dog", cfg, seeds=(0,), stop_fracs=(0.5,), swap_stop=0.05)
    assert report.swap_ratio == pytest.approx(0.0, abs=1e-5)
    assert report.done is True


def test_freeze_hold_is_zero_at_init():
    report = _report(MockBackend(), "#")
    assert report.hold == pytest.approx(0.0, abs=1e-6)
    assert report.done is False  # freeze-only phrase is not a stoppable edit


def test_composite_done_needs_write_and_hold():
    report = _report(MockBackend(), "#:0.4|human=robot:0.8|robot%human:-0.1")
    assert report.swap_ratio == pytest.approx(1.0, abs=1e-5)
    assert report.hold == pytest.approx(0.0, abs=1e-6)
    assert report.done is False
    assert "swap_ratio" in report.summary()
