"""Klein backend registry and alias smoke (no 4B weight load)."""
from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.klein import (
    DEFAULT_MODEL, KLEIN_9B_BASE, _LORA_TARGETS, looks_distilled,
    resolve_model_id,
)


def test_klein_is_registered():
    assert "klein" in BACKENDS
    assert DEFAULT_MODEL == "black-forest-labs/FLUX.2-klein-base-4B"
    assert KLEIN_9B_BASE == "black-forest-labs/FLUX.2-klein-base-9B"
    assert "to_qkv_mlp_proj" in _LORA_TARGETS
    assert "to_q" in _LORA_TARGETS


def test_klein_aliases_and_distilled_heuristic():
    assert resolve_model_id("9b-base") == KLEIN_9B_BASE
    assert resolve_model_id("4b") == "black-forest-labs/FLUX.2-klein-4B"
    assert resolve_model_id(DEFAULT_MODEL) == DEFAULT_MODEL
    assert looks_distilled("black-forest-labs/FLUX.2-klein-4B")
    assert looks_distilled("9b")
    assert not looks_distilled(DEFAULT_MODEL)
    assert not looks_distilled("9b-base")


def test_unknown_backend_still_lists_klein():
    try:
        load_backend("not-a-backend", device="cuda:0")
    except ValueError as exc:
        assert "klein" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_klein_matches_trainer_protocol():
    from conceptmod.backends.klein import KleinBackend

    required = (
        "encode_text", "encode_text_grad", "predict_v", "partial_denoise",
        "render", "generate", "trainable_parameters", "save_trained",
        "training_defaults", "attach_encoder_lora",
    )
    for name in required:
        assert callable(getattr(KleinBackend, name)), f"KleinBackend.{name}"
