"""CPU unit tests for the Flux.2 Klein convert mapper (synthetic PEFT keys)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from conceptmod import convert as cvt
from conceptmod import convert_klein as klein

ROOT = Path(__file__).resolve().parents[1]


def _peft(module: str, side: str) -> str:
    return f"base_model.model.{module}.{side}.weight"


def _write_adapter(path: Path, modules, rank=4, extra=None, dims=None):
    tensors = {}
    for module in modules:
        in_dim, out_dim = (dims or {}).get(module, (8, 8))
        tensors[_peft(module, "lora_A")] = torch.ones(rank, in_dim)
        tensors[_peft(module, "lora_B")] = torch.ones(out_dim, rank)
    if extra:
        tensors.update(extra)
    save_file(tensors, str(path))
    return path


def test_map_klein_double_and_single():
    assert klein.map_klein("transformer_blocks.2.attn.to_q") == (
        "double_blocks.2.img_attn.to_q")
    assert klein.map_klein("transformer_blocks.2.attn.to_out.0") == (
        "double_blocks.2.img_attn.proj")
    assert klein.map_klein("transformer_blocks.2.attn.add_q_proj") == (
        "double_blocks.2.txt_attn.to_q")
    assert klein.map_klein("transformer_blocks.2.attn.to_add_out") == (
        "double_blocks.2.txt_attn.proj")
    assert klein.map_klein("single_transformer_blocks.7.attn.to_qkv_mlp_proj") == (
        "single_blocks.7.linear1")
    assert klein.map_klein("single_transformer_blocks.7.attn.to_out") == (
        "single_blocks.7.linear2")
    assert klein.map_klein("single_transformer_blocks.7.attn.to_q") == (
        "single_blocks.7.linear1_to_q")
    assert klein.map_klein("not_a_real_module") is None


def test_klein_is_registered_unverified_not_unsupported():
    assert cvt.MAPPERS["klein"] is klein.map_klein
    assert cvt.MODEL_CLASSES["Flux2Transformer2DModel"] == "klein"
    assert "klein" in cvt.UNVERIFIED
    assert "klein" not in cvt.UNSUPPORTED
    assert "klein" in cvt.FUSED_QKV
    assert cvt.HOST["klein"] == "dit"


def test_detect_backend_klein_from_config_and_keys():
    cfg = {"auto_mapping": {"base_model_class": "Flux2Transformer2DModel"}}
    backend, why = cvt.detect_backend(cfg, [])
    assert backend == "klein"
    assert "base_model_class" in why

    keys = [_peft("single_transformer_blocks.0.attn.to_qkv_mlp_proj", "lora_A")]
    backend, why = cvt.detect_backend({}, keys)
    assert backend == "klein"
    assert "single_transformer_blocks" in why

    keys = [_peft("transformer_blocks.0.attn.to_qkv_mlp_proj", "lora_A")]
    backend, why = cvt.detect_backend({}, keys)
    assert backend == "klein"
    assert "to_qkv_mlp_proj" in why


def test_detect_klein_before_qwen_add_q_proj():
    # Klein double-stream also has add_q_proj; single-stream must win.
    keys = [
        _peft("transformer_blocks.0.attn.add_q_proj", "lora_A"),
        _peft("single_transformer_blocks.0.attn.to_qkv_mlp_proj", "lora_A"),
    ]
    backend, why = cvt.detect_backend({}, keys)
    assert backend == "klein"
    assert "single_transformer_blocks" in why


def test_synthetic_klein_fuses_img_attn_qkv(tmp_path):
    modules = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.add_q_proj",
        "transformer_blocks.0.attn.add_k_proj",
        "transformer_blocks.0.attn.add_v_proj",
        "transformer_blocks.0.attn.to_add_out",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        "single_transformer_blocks.0.attn.to_out",
    ]
    path = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    out, dropped, unmapped = cvt.convert(str(path), "klein", {"r": 4, "lora_alpha": 4})
    assert not dropped and not unmapped
    expected = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight",
        "diffusion_model.double_blocks.0.img_attn.qkv.alpha",
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
        "diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight",
        "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight",
        "diffusion_model.double_blocks.0.txt_attn.qkv.alpha",
        "diffusion_model.double_blocks.0.txt_attn.proj.lora_A.weight",
        "diffusion_model.double_blocks.0.txt_attn.proj.lora_B.weight",
        "diffusion_model.single_blocks.0.linear1.lora_A.weight",
        "diffusion_model.single_blocks.0.linear1.lora_B.weight",
        "diffusion_model.single_blocks.0.linear2.lora_A.weight",
        "diffusion_model.single_blocks.0.linear2.lora_B.weight",
    }
    assert set(out) == expected
    qkv_down = out["diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"]
    qkv_up = out["diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight"]
    assert tuple(qkv_down.shape) == (12, 8)
    assert tuple(qkv_up.shape) == (24, 12)
    assert not any(".to_q.lora_" in k or ".img_attn.to_k." in k for k in out)


def test_klein_qkv_fusion_preserves_delta(tmp_path):
    torch.manual_seed(0)
    rank, dim, alpha = 3, 8, 6.0
    q_down, k_down, v_down = (torch.randn(rank, dim) for _ in range(3))
    q_up, k_up, v_up = (torch.randn(dim, rank) for _ in range(3))
    tensors = {}
    for tail, down, up in (
        ("to_q", q_down, q_up),
        ("to_k", k_down, k_up),
        ("to_v", v_down, v_up),
    ):
        name = f"base_model.model.transformer_blocks.0.attn.{tail}"
        tensors[f"{name}.lora_A.weight"] = down
        tensors[f"{name}.lora_B.weight"] = up
    path = tmp_path / "qkv.safetensors"
    save_file(tensors, str(path))
    out, dropped, unmapped = cvt.convert(str(path), "klein", {"r": rank, "lora_alpha": alpha})
    assert not dropped and not unmapped
    fused_down, fused_up, fused_alpha = cvt._block_diag_up(
        [(q_down, q_up, alpha), (k_down, k_up, alpha), (v_down, v_up, alpha)])
    want = torch.cat(
        [
            q_up @ q_down * (alpha / rank),
            k_up @ k_down * (alpha / rank),
            v_up @ v_down * (alpha / rank),
        ],
        dim=0,
    )
    written_down = out[
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"].float()
    written_up = out[
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight"].float()
    written_alpha = float(out[
        "diffusion_model.double_blocks.0.img_attn.qkv.alpha"])
    written_delta = (written_up * (written_alpha / written_down.shape[0])) @ written_down
    assert torch.allclose(written_delta, want, atol=2e-2, rtol=1e-2)
    assert torch.allclose(fused_down, written_down.float(), atol=2e-2, rtol=1e-2)


def test_synthetic_klein_fuses_single_block_linear1(tmp_path):
    modules = [
        "single_transformer_blocks.1.attn.to_q",
        "single_transformer_blocks.1.attn.to_k",
        "single_transformer_blocks.1.attn.to_v",
        "single_transformer_blocks.1.attn.to_out",
    ]
    path = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    out, dropped, unmapped = cvt.convert(str(path), "klein", {"r": 4, "lora_alpha": 4})
    assert not dropped and not unmapped
    assert "diffusion_model.single_blocks.1.linear1.lora_A.weight" in out
    assert "diffusion_model.single_blocks.1.linear2.lora_A.weight" in out
    assert not any("linear1_to_q" in k for k in out)
    down = out["diffusion_model.single_blocks.1.linear1.lora_A.weight"]
    up = out["diffusion_model.single_blocks.1.linear1.lora_B.weight"]
    assert tuple(down.shape) == (12, 8)
    assert tuple(up.shape) == (24, 12)


def test_klein_sidecar_marks_fused_qkv(tmp_path):
    modules = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    ]
    src = _write_adapter(tmp_path / "adapter_model.safetensors", modules)
    cfg = {
        "r": 4,
        "lora_alpha": 4,
        "auto_mapping": {"base_model_class": "Flux2Transformer2DModel"},
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    argv = ["convert_lora_comfyui.py", str(tmp_path)]
    old = sys.argv
    try:
        sys.argv = argv
        cvt.main()
    finally:
        sys.argv = old
    side = json.loads((tmp_path / "adapter_model_comfyui.safetensors.json").read_text())
    assert side["arch"] == "klein"
    assert side["fused_qkv"] is True
    assert side["host"] == "dit"


def test_klein_incomplete_qkv_is_unmapped(tmp_path):
    modules = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
    ]
    path = _write_adapter(tmp_path / "partial.safetensors", modules)
    _out, _dropped, unmapped = cvt.convert(str(path), "klein", {"r": 4, "lora_alpha": 4})
    assert any("incomplete QKV" in k for k in unmapped)
