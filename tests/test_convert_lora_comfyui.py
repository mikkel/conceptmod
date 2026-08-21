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


# ---- Music 3 (LoRANetwork sliders, not PEFT) --------------------------------

HUB_REPO = "ntc-ai/minimax-music3-concept-sliders"
HUB_TF = "weights/energy-slider-v2/energy_unit_last.safetensors"
HUB_LM = "weights/gender-lm-v4/gender-lm-v4_last.safetensors"


def _tf_slider(layer: int, tail: str, rank=4, dim=16, alpha=8.0):
    name = f"lora_unet-transformer_blocks-{layer}-attn-{tail}"
    return {
        f"{name}.lora_down.weight": torch.randn(rank, dim),
        f"{name}.lora_up.weight": torch.randn(dim, rank),
        f"{name}.alpha": torch.tensor(alpha),
    }


def _lm_slider(layer: int, proj: str, rank=4, in_dim=32, out_dim=32, alpha=8.0):
    name = f"lora_te-model-layers-{layer}-self_attn-{proj}"
    return {
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
        f"{name}.alpha": torch.tensor(alpha),
    }


def _write_tensors(path: Path, tensors):
    save_file(tensors, str(path))
    return path


def test_music3_maps_loranetwork_attention():
    assert cvt.map_music3("lora_unet-transformer_blocks-3-attn-to_out-0") == (
        "diffusion_transformer.transformer.layers.3.self_attn.to_out")
    assert cvt.map_music3("lora_unet-transformer_blocks-3-attn-to_q") == (
        "diffusion_transformer.transformer.layers.3.self_attn.to_q")
    assert cvt.map_music3("lora_unet-proj_in") == (
        "diffusion_transformer.transformer.project_in")
    assert cvt.map_music3("lora_unet-transformer_blocks-1-ff_in") == (
        "diffusion_transformer.transformer.layers.1.ff.ff.0.proj")
    assert cvt.map_music3("lora_unet-not-a-real-module") is None


def test_music3_lm_maps_qwen3_clip_names():
    assert cvt.map_music3_lm("lora_te-model-layers-2-self_attn-q_proj") == (
        "model.layers.2.self_attn.q_proj")
    assert cvt.map_music3_lm("lora_te-model-layers-2-self_attn-o_proj") == (
        "model.layers.2.self_attn.o_proj")
    assert cvt.map_music3_lm("lora_te-model-layers-2-mlp-gate_proj") is None


def test_detect_backend_music3_from_loranetwork_keys():
    backend, why = cvt.detect_backend(
        {}, ["lora_unet-transformer_blocks-0-attn-to_q.lora_down.weight"])
    assert backend == "music3"
    assert "music3" in why

    backend, why = cvt.detect_backend(
        {}, ["lora_te-model-layers-0-self_attn-q_proj.lora_down.weight"])
    assert backend == "music3_lm"
    assert "music3" in why

    backend, why = cvt.detect_backend({"kind": "language_model"}, [])
    assert backend == "music3_lm"


def test_music3_fuses_qkv_and_keeps_to_out(tmp_path):
    tensors = {}
    for layer in (0, 1):
        for tail in ("to_q", "to_k", "to_v", "to_out-0"):
            tensors.update(_tf_slider(layer, tail, rank=4, dim=16, alpha=8.0))
    path = _write_tensors(tmp_path / "slider.safetensors", tensors)
    out, dropped, unmapped = cvt.convert(str(path), "music3", {})
    assert not dropped and not unmapped
    expected = {
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha",
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.lora_A.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.lora_B.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_out.alpha",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.lora_A.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.lora_B.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_qkv.alpha",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.lora_A.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.lora_B.weight",
        "diffusion_model.diffusion_transformer.transformer.layers.1.self_attn.to_out.alpha",
    }
    assert set(out) == expected
    qkv_down = out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight"]
    qkv_up = out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight"]
    assert tuple(qkv_down.shape) == (12, 16)
    assert tuple(qkv_up.shape) == (48, 12)
    assert float(out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha"
    ]) == 24.0


def test_music3_qkv_fusion_preserves_delta(tmp_path):
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
        name = f"lora_unet-transformer_blocks-0-attn-{tail}"
        tensors[f"{name}.lora_down.weight"] = down
        tensors[f"{name}.lora_up.weight"] = up
        tensors[f"{name}.alpha"] = torch.tensor(alpha)
    path = _write_tensors(tmp_path / "qkv.safetensors", tensors)
    out, dropped, unmapped = cvt.convert(str(path), "music3", {})
    assert not dropped and not unmapped
    fused_down, fused_up, fused_alpha = cvt._block_diag_up(
        [(q_down, q_up, alpha), (k_down, k_up, alpha), (v_down, v_up, alpha)])
    scale = fused_alpha / fused_down.shape[0]
    fused_delta = (fused_up * scale) @ fused_down
    want = torch.cat(
        [
            q_up @ q_down * (alpha / rank),
            k_up @ k_down * (alpha / rank),
            v_up @ v_down * (alpha / rank),
        ],
        dim=0,
    )
    assert torch.allclose(fused_delta, want, atol=1e-5)
    written_down = out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_A.weight"
    ].float()
    written_up = out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.lora_B.weight"
    ].float()
    written_alpha = float(out[
        "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv.alpha"])
    written_delta = (written_up * (written_alpha / written_down.shape[0])) @ written_down
    assert torch.allclose(written_delta, want, atol=2e-2, rtol=1e-2)


def test_music3_full_targets_root_modules(tmp_path):
    tensors = _tf_slider(0, "to_q")
    tensors.update(_tf_slider(0, "to_k"))
    tensors.update(_tf_slider(0, "to_v"))
    tensors.update({
        "lora_unet-proj_in.lora_down.weight": torch.randn(4, 32),
        "lora_unet-proj_in.lora_up.weight": torch.randn(16, 4),
        "lora_unet-proj_in.alpha": torch.tensor(4.0),
        "lora_unet-preprocess_conv.lora_down.weight": torch.randn(4, 32),
        "lora_unet-preprocess_conv.lora_up.weight": torch.randn(32, 4),
        "lora_unet-preprocess_conv.alpha": torch.tensor(4.0),
        "lora_unet-transformer_blocks-0-ff_in.lora_down.weight": torch.randn(4, 16),
        "lora_unet-transformer_blocks-0-ff_in.lora_up.weight": torch.randn(64, 4),
        "lora_unet-transformer_blocks-0-ff_in.alpha": torch.tensor(4.0),
    })
    path = _write_tensors(tmp_path / "full.safetensors", tensors)
    out, dropped, unmapped = cvt.convert(str(path), "music3", {})
    assert not dropped and not unmapped
    assert "diffusion_model.diffusion_transformer.transformer.project_in.lora_A.weight" in out
    assert "diffusion_model.diffusion_transformer.preprocess_conv.lora_A.weight" in out
    assert "diffusion_model.diffusion_transformer.transformer.layers.0.ff.ff.0.proj.lora_A.weight" in out


def test_music3_lm_keeps_gqa_shapes(tmp_path):
    tensors = {}
    tensors.update(_lm_slider(0, "q_proj", out_dim=32))
    tensors.update(_lm_slider(0, "k_proj", out_dim=8))
    tensors.update(_lm_slider(0, "v_proj", out_dim=8))
    tensors.update(_lm_slider(0, "o_proj", out_dim=32))
    path = _write_tensors(tmp_path / "lm.safetensors", tensors)
    out, dropped, unmapped = cvt.convert(str(path), "music3_lm", {})
    assert not dropped and not unmapped
    assert tuple(out["text_encoders.model.layers.0.self_attn.k_proj.lora_B.weight"].shape) == (8, 4)
    assert tuple(out["text_encoders.model.layers.0.self_attn.q_proj.lora_B.weight"].shape) == (32, 4)
    assert "text_encoders.model.layers.0.self_attn.o_proj.lora_A.weight" in out


def test_music3_unmapped_and_incomplete_qkv(tmp_path):
    tensors = _tf_slider(0, "to_q")
    tensors.update(_tf_slider(0, "to_k"))
    tensors.update(_tf_slider(0, "to_v"))
    tensors["lora_unet-mystery-linear.lora_down.weight"] = torch.randn(4, 8)
    tensors["lora_unet-mystery-linear.lora_up.weight"] = torch.randn(8, 4)
    path = _write_tensors(tmp_path / "bad.safetensors", tensors)
    _out, _dropped, unmapped = cvt.convert(str(path), "music3", {})
    assert any("mystery" in k for k in unmapped)

    partial = _tf_slider(0, "to_q")
    partial.update(_tf_slider(0, "to_k"))
    path = _write_tensors(tmp_path / "partial.safetensors", partial)
    _out, _dropped, unmapped = cvt.convert(str(path), "music3", {})
    assert any("incomplete QKV" in k for k in unmapped)


def _hub_slider(filename: str):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(HUB_REPO, filename)
    except Exception:
        return None


def test_hub_energy_transformer_slider():
    local = _hub_slider(HUB_TF)
    if local is None:
        pytest.skip(f"could not download {HUB_TF}")
    kind, keys = cvt.classify(local)
    assert kind == "lora"
    backend, _why = cvt.detect_backend(cvt.read_config(local), keys)
    assert backend == "music3"
    out, dropped, unmapped = cvt.convert(local, "music3", cvt.read_config(local))
    assert not dropped and not unmapped
    qkv = [k for k in out if k.endswith(".self_attn.to_qkv.lora_A.weight")]
    outs = [k for k in out if k.endswith(".self_attn.to_out.lora_A.weight")]
    assert len(qkv) == 36
    assert len(outs) == 36
    assert all(k.startswith("diffusion_model.diffusion_transformer.") for k in out)
    sample = out[qkv[0]]
    assert tuple(sample.shape) == (24, 2048)
    assert sample.dtype == torch.bfloat16
    assert not any(".to_q.lora_" in k or ".to_k.lora_" in k for k in out)


def test_hub_gender_lm_slider():
    local = _hub_slider(HUB_LM)
    if local is None:
        pytest.skip(f"could not download {HUB_LM}")
    kind, keys = cvt.classify(local)
    assert kind == "lora"
    backend, _why = cvt.detect_backend(cvt.read_config(local), keys)
    assert backend == "music3_lm"
    out, dropped, unmapped = cvt.convert(local, "music3_lm", cvt.read_config(local))
    assert not dropped and not unmapped
    q = [k for k in out if k.endswith(".self_attn.q_proj.lora_A.weight")]
    k = [k for k in out if k.endswith(".self_attn.k_proj.lora_A.weight")]
    o = [k for k in out if k.endswith(".self_attn.o_proj.lora_A.weight")]
    assert len(q) == 36 and len(k) == 36 and len(o) == 36
    assert all(name.startswith("text_encoders.model.layers.") for name in out)
    k_up = out["text_encoders.model.layers.0.self_attn.k_proj.lora_B.weight"]
    q_up = out["text_encoders.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert tuple(k_up.shape) == (1024, 8)
    assert tuple(q_up.shape) == (4096, 8)
