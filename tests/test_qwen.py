"""Qwen backend registry and convert-path smoke (no 20B weight load)."""
from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.qwen import DEFAULT_MODEL, _LORA_TARGETS


def test_qwen_is_registered():
    assert "qwen" in BACKENDS
    assert DEFAULT_MODEL == "Qwen/Qwen-Image"
    assert _LORA_TARGETS == ["to_q", "to_k", "to_v", "to_out.0"]


def test_unknown_backend_still_lists_qwen():
    try:
        load_backend("not-a-backend", device="cuda:0")
    except ValueError as exc:
        assert "qwen" in str(exc)
    else:
        raise AssertionError("expected ValueError")
