"""Tests for scripts/convert_lora_comfyui.py.

These encode the verification from ntc-ai/conceptmod#3: synthetic PEFT
adapters covering every mapped Anima/Krea module, droppable Anima
prefixes, alpha emission, and that KREA_BLOCK is the inverse of the
linear entries in krea_weights._BLOCK_SUFFIX.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_lora_comfyui.py"


def load_convert():
    spec = importlib.util.spec_from_file_location("convert_lora_comfyui", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cvt = load_convert()


def _peft(module: str, side: str) -> str:
    return f"base_model.model.{module}.{side}.weight"


def _write_adapter(path: Path, modules, rank=8, extra=None):
    tensors = {}
    for module in modules:
        tensors[_peft(module, "lora_A")] = torch.ones(rank, 4)
        tensors[_peft(module, "lora_B")] = torch.ones(4, rank)
    if extra:
        tensors.update(extra)
    save_file(tensors, str(path))
    return path


def test_anima_maps_every_block_module():
    for tail, native in cvt.ANIMA_BLOCK.items():
        assert cvt.map_anima(f"transformer_blocks.3.{tail}") == (
            f"blocks.3.{native}")


def test_anima_drops_non_block_prefixes():
    for prefix in cvt.ANIMA_DROP:
        assert cvt.map_anima(f"{prefix}weight") is cvt.DROP


def test_krea_maps_all_three_stems():
    for stem, dest in cvt.KREA_STEMS:
        for tail, native in cvt.KREA_BLOCK.items():
            assert cvt.map_krea(f"{stem}2.{tail}") == f"{dest}2.{native}"


def test_krea_block_is_inverse_of_loader_linears():
    from conceptmod.backends.krea_weights import _BLOCK_SUFFIX

    # LoRA only attaches to Linear weights. Norms / modulation tables stay out.
    linear = {}
    for comfy, diffusers in _BLOCK_SUFFIX.items():
        if not diffusers.endswith(".weight"):
            continue
        if any(part in diffusers for part in (".norm", "norm1", "norm2", "norm_q", "norm_k")):
            continue
        linear[comfy] = diffusers[: -len(".weight")]
    inverse = {diff: comfy for comfy, diff in linear.items()}
    assert inverse == cvt.KREA_BLOCK
    assert set(cvt.KREA_BLOCK.values()) == set(linear)


def test_detect_backend_from_config_and_keys():
    anima_cfg = {"auto_mapping": {"base_model_class": "CosmosTransformer3DModel"}}
    backend, why = cvt.detect_backend(anima_cfg, [])
    assert backend == "anima"
    assert "base_model_class" in why

    krea_keys = [_peft("text_fusion.layerwise_blocks.0.attn.to_q", "lora_A")]
    backend, why = cvt.detect_backend({}, krea_keys)
    assert backend == "krea"
    assert "text_fusion" in why

    krea_attn = [_peft("transformer_blocks.0.attn.to_q", "lora_A")]
    backend, why = cvt.detect_backend({}, krea_attn)
    assert backend == "krea"

    anima_keys = [_peft("transformer_blocks.0.attn1.to_q", "lora_A")]
    backend, why = cvt.detect_backend({}, anima_keys)
    assert backend is None
    assert "ambiguous" in why


def test_synthetic_anima_convert_and_drops(tmp_path):
    modules = [f"transformer_blocks.0.{tail}" for tail in cvt.ANIMA_BLOCK]
    modules += [f"transformer_blocks.1.{tail}" for tail in cvt.ANIMA_BLOCK]
    drop_modules = [f"{prefix}proj" for prefix in cvt.ANIMA_DROP]
    path = _write_adapter(tmp_path / "adapter_model.safetensors", modules + drop_modules)
    tensors, dropped, unmapped = cvt.convert(str(path), "anima", {"r": 8, "lora_alpha": 8})
    assert not unmapped
    assert len(dropped) == 2 * len(drop_modules)
    # 16 mapped modules × 2 blocks × 2 sides = 64 keys; 32 unique patterns.
    patterns = {cvt.pattern(k) for k in tensors}
    assert len(tensors) == 64
    assert len(patterns) == 32
    assert all(k.startswith("diffusion_model.blocks.") for k in tensors)
    assert all(k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight") for k in tensors)
    assert not any(k.endswith(".alpha") for k in tensors)


def test_synthetic_krea_convert_three_stems(tmp_path):
    modules = []
    for stem, _dest in cvt.KREA_STEMS:
        for tail in cvt.KREA_BLOCK:
            modules.append(f"{stem}0.{tail}")
    path = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    tensors, dropped, unmapped = cvt.convert(str(path), "krea", {"r": 8, "lora_alpha": 8})
    assert not dropped
    assert not unmapped
    assert len(tensors) == len(modules) * 2
    joined = " ".join(tensors)
    assert "diffusion_model.blocks.0." in joined
    assert "diffusion_model.txtfusion.layerwise_blocks.0." in joined
    assert "diffusion_model.txtfusion.refiner_blocks.0." in joined


def test_alpha_emitted_only_when_not_equal_rank(tmp_path):
    modules = ["transformer_blocks.0.attn.to_q"]
    path = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    same, _, _ = cvt.convert(str(path), "krea", {"r": 8, "lora_alpha": 8})
    assert not any(k.endswith(".alpha") for k in same)
    diff, _, _ = cvt.convert(str(path), "krea", {"r": 8, "lora_alpha": 16})
    alphas = [k for k in diff if k.endswith(".alpha")]
    assert alphas == ["diffusion_model.blocks.0.attn.wq.alpha"]
    assert diff[alphas[0]].item() == 16.0


def test_check_against_anima_reference(tmp_path):
    modules = [f"transformer_blocks.0.{tail}" for tail in cvt.ANIMA_BLOCK]
    src = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    tensors, dropped, unmapped = cvt.convert(str(src), "anima", {"r": 8, "lora_alpha": 8})
    assert not dropped and not unmapped
    ref_tensors = {k: torch.ones(1) for k in tensors}
    # Extra reference keys must not fail the converted subset.
    ref_tensors["diffusion_model.blocks.N.self_attn.q_proj.lora_A.weight"] = torch.ones(1)
    ref = tmp_path / "anima_masterpiece_example.safetensors"
    save_file(ref_tensors, str(ref))
    assert cvt.check_against(tensors, str(ref), "anima")
    # A converted key absent from the reference fails.
    tensors["diffusion_model.blocks.0.not_in_example.lora_A.weight"] = torch.ones(1)
    assert not cvt.check_against(tensors, str(ref), "anima")


def test_cli_skips_existing_output(tmp_path, capsys):
    modules = ["transformer_blocks.0.attn.to_q"]
    src = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    cfg = {
        "r": 8,
        "lora_alpha": 8,
        "auto_mapping": {"base_model_class": "Krea2Transformer2DModel"},
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    argv = ["convert_lora_comfyui.py", str(tmp_path)]
    old = sys.argv
    try:
        sys.argv = argv
        cvt.main()
        out = tmp_path / "adapter_model_comfyui.safetensors"
        assert out.is_file()
        first_mtime = out.stat().st_mtime
        sys.argv = argv
        cvt.main()
    finally:
        sys.argv = old
    captured = capsys.readouterr().out
    assert "exists, skipping" in captured
    assert out.stat().st_mtime == first_mtime
