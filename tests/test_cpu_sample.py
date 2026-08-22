"""CPU sample problem: real dsl/ops path, no GPU, no Hub weights."""

from __future__ import annotations

import torch

from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.cpu import (
    COSINE_THRESHOLD,
    SAMPLE_DST,
    SAMPLE_PHRASE,
    SAMPLE_SRC,
    CpuBackend,
    lora_B_norm,
    prompt_class,
    run_sample_problem,
    write_alignment_cosine,
)


def test_cpu_is_registered():
    assert "cpu" in BACKENDS
    backend = load_backend("cpu", device="cpu", lora_rank=4, seed=0)
    assert isinstance(backend, CpuBackend)
    assert backend.device == "cpu"
    assert backend.latent_shape == (4, 8, 8)


def test_dummy_alias_loads_cpu_backend():
    backend = load_backend("dummy", device="cpu", lora_rank=4, seed=0)
    assert isinstance(backend, CpuBackend)


def test_prompt_class_is_two_class():
    assert prompt_class("") == 0
    assert prompt_class("a photo of a red") == 1
    assert prompt_class("a photo of a blue") == 2
    assert prompt_class("a red sitting on a windowsill") == 1


def test_velocity_geometry_is_opposite_before_train():
    backend = CpuBackend(device="cpu", lora_rank=4, seed=0)
    g = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    t = torch.tensor([500.0])
    v_null = backend.predict_v("", z, t, frozen=True)
    v_red = backend.predict_v(SAMPLE_SRC, z, t, frozen=True)
    v_blue = backend.predict_v(SAMPLE_DST, z, t, frozen=True)
    d_red = (v_red - v_null).flatten()
    d_blue = (v_blue - v_null).flatten()
    assert d_red.norm() > 0.1
    assert d_blue.norm() > 0.1
    cos = torch.nn.functional.cosine_similarity(
        d_red.unsqueeze(0), d_blue.unsqueeze(0), dim=1, eps=1e-6,
    ).item()
    assert cos < -0.7
    # LoRA B is zero-init, so trained == frozen at step 0
    assert write_alignment_cosine(backend, seed=0) < -0.7
    assert lora_B_norm(backend) == 0.0


def test_cpu_sample_write_improves_alignment():
    """After N CPU steps, trained red-direction aligns with frozen blue."""
    result = run_sample_problem(device="cpu")
    assert result.phrase == SAMPLE_PHRASE
    assert result.cosine_before < -0.5
    assert result.cosine_after > result.cosine_before
    assert result.cosine_after > COSINE_THRESHOLD
    assert result.lora_delta_norm > 0.0
    assert result.elapsed_s < 30.0
    print(
        f"cpu sample wall time: {result.elapsed_s:.2f}s "
        f"cosine {result.cosine_before:.3f} -> {result.cosine_after:.3f}"
    )


def test_lora_attaches_to_linear_layers():
    backend = CpuBackend(device="cpu", lora_rank=4, seed=0)
    base = backend.transformer.get_base_model()
    assert hasattr(base, "linear1") and hasattr(base, "linear2")
    names = [n for n, p in backend.transformer.named_parameters() if p.requires_grad]
    assert any("linear2" in n and "lora_" in n for n in names)
    params = backend.trainable_parameters("lora")
    assert params
    assert sum(p.numel() for p in params) < 50_000
