"""CPU fixture for the ``c--`` erase path (keep vs erase disentangle).

``--backend cpu`` may land in a parallel PR. This file does not wait for
it: the two-concept velocity below is self-contained and is *not* a 20B
train. It is a tiny rectified-flow stand-in so ESD (and the optional
GEM / EraseAnything hooks in ``conceptmod.ops_erase``) can be scored on
CPU in well under 30s.

Cite: EraseAnything (Gao et al., ICML'25) https://arxiv.org/abs/2412.20413
GEM is *not* fully ported — ``ops_erase`` is a velocity-space hook.
"""

from __future__ import annotations

import time

import pytest
import torch

from conceptmod import dsl, ops
from conceptmod.ops_erase import (
    ERASE_MODES,
    concept_probe,
    erase_loss,
    esd_loss,
)

# Dummy concepts: orthogonal directions in velocity space.
ERASE = "stripe"
KEEP = "dot"


class TwoConceptVelocity:
    """Frozen field is a linear mix of two orthonormal concept directions.

    Trained residual is *prompt-gated* (LoRA-shaped): only the matching
    concept's delta is added. That is the smallest model in which
    ``erase--`` can reduce the erase probe without dragging the keep
    probe — a global additive delta (see ``tests/test_ops.MockBackend``)
    cannot show disentangle.
    """

    device = "cpu"
    latent_shape = (4, 4, 4)

    def __init__(self):
        self.d_erase = torch.zeros(1, *self.latent_shape)
        self.d_erase[0, 0, 0, 0] = 1.0
        self.d_keep = torch.zeros(1, *self.latent_shape)
        self.d_keep[0, 1, 0, 0] = 1.0
        self.delta_e = torch.nn.Parameter(torch.zeros(1, *self.latent_shape))
        self.delta_k = torch.nn.Parameter(torch.zeros(1, *self.latent_shape))

    def _gates(self, prompt: str) -> tuple[float, float]:
        p = prompt.lower()
        return (1.0 if ERASE in p else 0.0), (1.0 if KEEP in p else 0.0)

    def predict_v(self, prompt, z, timestep, frozen):
        ge, gk = self._gates(prompt)
        v = 0.1 * z
        v = v + ge * self.d_erase + gk * self.d_keep
        if not frozen:
            v = v + ge * self.delta_e + gk * self.delta_k
        return v

    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        z = torch.randn(1, *self.latent_shape, generator=generator)
        return z, torch.tensor(float(stop_index))

    def trainable_parameters(self, train_method="full"):
        return [self.delta_e, self.delta_k]


@pytest.fixture
def bare_templates(monkeypatch):
    """Pin the ESD template so the phrase is the bare concept word."""
    monkeypatch.setattr(ops, "ERASE_TEMPLATES", ["{}"])


def _cfg(mode="esd"):
    cfg = ops.OpDefaults(erase_guidance=1.0, sample_steps=4)
    cfg.erase_mode = mode
    cfg.erase_keep = KEEP
    cfg.gem_eta = 1.0
    cfg.erase_retain = 1.0
    return cfg


def _probe_z(backend, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(1, *backend.latent_shape, generator=g)
    return z, torch.tensor(2.0)


def _probes(backend, z, t):
    return (
        concept_probe(backend, ERASE, backend.d_erase, z, t),
        concept_probe(backend, KEEP, backend.d_keep, z, t),
    )


def _step_erase(backend, mode, steps=16, lr=0.5, seed=0):
    """SGD on ``ERASE--``. ESD uses the live ``ops.rule_loss`` path."""
    rule = dsl.parse_phrase(f"{ERASE}--")[0]
    cfg = _cfg(mode)
    opt = torch.optim.SGD(backend.trainable_parameters(), lr=lr)
    for i in range(steps):
        ctx = ops.StepContext(backend, stop_index=2, seed=seed + i, cfg=cfg)
        if mode == "esd":
            loss = ops.rule_loss(rule, ctx)
        else:
            loss = erase_loss(rule, ctx, mode=mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return rule, cfg


def test_esd_erase_reduces_erase_probe_more_than_keep(bare_templates):
    """Success criterion: ``stripe--`` disentangles from ``dot``.

    After ESD on the live ``ops.rule_loss`` path, the erase-concept CFG
    probe drops more than the keep-concept probe (preserve / disentangle).
    """
    t0 = time.time()
    backend = TwoConceptVelocity()
    z, t = _probe_z(backend)
    erase0, keep0 = _probes(backend, z, t)
    assert erase0 == pytest.approx(1.0, abs=1e-5)
    assert keep0 == pytest.approx(1.0, abs=1e-5)

    _step_erase(backend, "esd")
    erase1, keep1 = _probes(backend, z, t)

    erase_drop = erase0 - erase1
    keep_drop = keep0 - keep1
    assert erase_drop > 0.0
    assert erase_drop > keep_drop
    # prompt-gated residual: keep delta is never in the ESD graph
    assert abs(keep_drop) < 1e-6
    assert time.time() - t0 < 30.0


def test_ops_erase_esd_matches_live_rule_loss(bare_templates):
    """``ops_erase.esd`` is the same tensor as ``ops.rule_loss`` (ERASE)."""
    backend = TwoConceptVelocity()
    rule = dsl.parse_phrase(f"{ERASE}--")[0]
    cfg = _cfg("esd")
    a = ops.rule_loss(rule, ops.StepContext(backend, 2, seed=7, cfg=cfg))
    b = esd_loss(rule, ops.StepContext(backend, 2, seed=7, cfg=cfg))
    assert torch.allclose(a, b)


def test_ea_retain_is_esd_plus_keep_anchor(bare_templates):
    """EA hook = ESD + retain; keep probe stays put, erase still drops.

    Not a full EraseAnything port (no attention regularizer / RSC / bi-level).
    https://arxiv.org/abs/2412.20413
    """
    backend = TwoConceptVelocity()
    z, t = _probe_z(backend)
    erase0, keep0 = _probes(backend, z, t)
    _step_erase(backend, "ea")
    erase1, keep1 = _probes(backend, z, t)
    assert erase0 - erase1 > keep0 - keep1
    assert erase1 < erase0
    assert abs(keep1 - keep0) < 1e-6


def test_gem_contrastive_reduces_erase_probe(bare_templates):
    """GEM hinge attracts toward the keep/uncond field, repels erase.

    Not a full GEM port (no trajectory window). Default η = 1.
    """
    backend = TwoConceptVelocity()
    z, t = _probe_z(backend)
    erase0, keep0 = _probes(backend, z, t)
    _step_erase(backend, "gem", steps=24, lr=0.25)
    erase1, keep1 = _probes(backend, z, t)
    assert erase1 < erase0
    assert (erase0 - erase1) > (keep0 - keep1)


def test_unknown_erase_mode_rejected(bare_templates):
    backend = TwoConceptVelocity()
    rule = dsl.parse_phrase(f"{ERASE}--")[0]
    ctx = ops.StepContext(backend, 2, seed=0, cfg=_cfg())
    with pytest.raises(ValueError, match="unknown erase mode"):
        erase_loss(rule, ctx, mode="flux-only")
    assert ERASE_MODES == ("esd", "gem", "ea")


def test_erase_loss_rejects_non_erase_rule(bare_templates):
    backend = TwoConceptVelocity()
    rule = dsl.parse_phrase(f"{ERASE}++")[0]
    ctx = ops.StepContext(backend, 2, seed=0, cfg=_cfg())
    with pytest.raises(ValueError, match="erase rule"):
        erase_loss(rule, ctx)
