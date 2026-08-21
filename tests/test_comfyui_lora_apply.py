"""Apply converted Music 3 sliders through ComfyUI's real LoRA loader.

Uses ``comfy.lora.load_lora`` / ``model_lora_keys_unet`` /
``model_lora_keys_clip`` / ``calculate_weight`` against the MiniMax Music 3
module trees from ``comfy/ldm/minimax_music/dit.py`` and
``MiniMaxMusic3TEModel`` (meta device — no full checkpoint, no sampling).

Set ``COMFYUI_ROOT`` to a ComfyUI checkout, or the test shallow-clones
https://github.com/Comfy-Org/ComfyUI into ``/tmp/comfyui``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_lora_comfyui.py"
COMFY_REPO = "https://github.com/Comfy-Org/ComfyUI.git"
HUB_REPO = "ntc-ai/minimax-music3-concept-sliders"
HUB_TF = "weights/energy-slider-v2/energy_unit_last.safetensors"
HUB_LM = "weights/gender-lm-v4/gender-lm-v4_last.safetensors"


def load_convert():
    spec = importlib.util.spec_from_file_location("convert_lora_comfyui", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cvt = load_convert()


def _comfy_root() -> Path:
    return Path(os.environ.get("COMFYUI_ROOT", "/tmp/comfyui"))


def _ensure_comfyui(root: Path) -> Path:
    if (root / "comfy" / "lora.py").is_file():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", COMFY_REPO, str(root)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        pytest.skip(f"could not clone ComfyUI: {exc}")
    if not (root / "comfy" / "lora.py").is_file():
        pytest.skip(f"ComfyUI checkout at {root} is missing comfy/lora.py")
    return root


@pytest.fixture(scope="module")
def comfy():
    """Import ComfyUI's LoRA stack in CPU mode."""
    root = _ensure_comfyui(_comfy_root())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    try:
        import comfy.lora as lora
        from comfy.ldm.minimax_music.dit import MiniMaxMusic3DiT
        from comfy.text_encoders.minimax_music import MiniMaxMusic3TEModel
    except Exception as exc:
        pytest.skip(f"could not import ComfyUI LoRA / MiniMax modules: {exc}")
    return types.SimpleNamespace(
        lora=lora,
        MiniMaxMusic3DiT=MiniMaxMusic3DiT,
        MiniMaxMusic3TEModel=MiniMaxMusic3TEModel,
    )


def _dit_host(comfy):
    """BaseModel-shaped wrapper: state_dict keys start with diffusion_model."""
    host = torch.nn.Module()
    host.diffusion_model = comfy.MiniMaxMusic3DiT(
        dtype=torch.float32, device="meta", operations=torch.nn)
    host.model_config = types.SimpleNamespace(unet_config={})
    return host


def _clip_host(comfy, merged_qkv=False):
    return comfy.MiniMaxMusic3TEModel(
        device="meta",
        dtype=torch.float32,
        model_options={"custom_operations": torch.nn},
        projection_config={"merged_qkv": merged_qkv},
    )


def _lora_stems(tensors):
    stems = set()
    for key in tensors:
        if key.endswith(".lora_A.weight"):
            stems.add(key[: -len(".lora_A.weight")])
        elif key.endswith(".lora_B.weight"):
            stems.add(key[: -len(".lora_B.weight")])
        elif key.endswith(".alpha"):
            stems.add(key[: -len(".alpha")])
    return stems


def _load_with_unused(comfy, lora_sd, key_map):
    unused = []

    class Handler(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "lora key not loaded:" in msg:
                unused.append(msg.split("lora key not loaded: ", 1)[1])

    handler = Handler()
    log = logging.getLogger()
    log.addHandler(handler)
    try:
        patches = comfy.lora.load_lora(lora_sd, key_map, log_missing=True)
    finally:
        log.removeHandler(handler)
    return patches, unused


def _expected_delta(down, up, alpha):
    rank = int(down.shape[0])
    scale = (float(alpha) / rank) if alpha is not None else 1.0
    return (up.float() * scale) @ down.float()


def _apply_delta_error(comfy, patches, weight_key, down, up, alpha, strength=1.0):
    weight = torch.zeros(up.shape[0], down.shape[1], dtype=torch.float32)
    patched = comfy.lora.calculate_weight(
        [(strength, patches[weight_key], 1.0, None, None)],
        weight.clone(),
        weight_key,
        intermediate_dtype=torch.float32,
    )
    got = patched - weight
    want = _expected_delta(down, up, alpha) * strength
    return bool(not torch.equal(patched, weight)), (got - want).abs().max().item()


def _write_tf_slider(path: Path, layers=2, rank=4, dim=16, alpha=8.0):
    tensors = {}
    for layer in range(layers):
        for tail in ("to_q", "to_k", "to_v", "to_out-0"):
            name = f"lora_unet-transformer_blocks-{layer}-attn-{tail}"
            out_dim = dim
            tensors[f"{name}.lora_down.weight"] = torch.randn(rank, dim)
            tensors[f"{name}.lora_up.weight"] = torch.randn(out_dim, rank)
            tensors[f"{name}.alpha"] = torch.tensor(alpha)
    save_file(tensors, str(path))
    return path


def _write_lm_slider(path: Path, layers=2, rank=4, hidden=32, kv=8, alpha=8.0):
    tensors = {}
    for layer in range(layers):
        for proj, out_dim in (
            ("q_proj", hidden),
            ("k_proj", kv),
            ("v_proj", kv),
            ("o_proj", hidden),
        ):
            name = f"lora_te-model-layers-{layer}-self_attn-{proj}"
            tensors[f"{name}.lora_down.weight"] = torch.randn(rank, hidden)
            tensors[f"{name}.lora_up.weight"] = torch.randn(out_dim, rank)
            tensors[f"{name}.alpha"] = torch.tensor(alpha)
    save_file(tensors, str(path))
    return path


def test_synthetic_music3_dit_binds_and_delta(tmp_path, comfy):
    src = _write_tf_slider(tmp_path / "tf.safetensors")
    tensors, dropped, unmapped = cvt.convert(str(src), "music3", {})
    assert not dropped and not unmapped
    host = _dit_host(comfy)
    key_map = comfy.lora.model_lora_keys_unet(host, {})
    stems = _lora_stems(tensors)
    assert stems <= set(key_map)
    patches, unused = _load_with_unused(comfy, tensors, key_map)
    assert unused == []
    stem = "diffusion_model.diffusion_transformer.transformer.layers.0.self_attn.to_qkv"
    changed, err = _apply_delta_error(
        comfy, patches, f"{stem}.weight",
        tensors[f"{stem}.lora_A.weight"], tensors[f"{stem}.lora_B.weight"],
        float(tensors[f"{stem}.alpha"]),
    )
    assert changed
    assert err < 1e-5


def test_synthetic_music3_lm_binds_and_delta(tmp_path, comfy):
    src = _write_lm_slider(tmp_path / "lm.safetensors")
    tensors, dropped, unmapped = cvt.convert(str(src), "music3_lm", {})
    assert not dropped and not unmapped
    host = _clip_host(comfy, merged_qkv=False)
    key_map = comfy.lora.model_lora_keys_clip(host, {})
    stems = _lora_stems(tensors)
    assert stems <= set(key_map)
    patches, unused = _load_with_unused(comfy, tensors, key_map)
    assert unused == []
    stem = "text_encoders.model.layers.0.self_attn.q_proj"
    changed, err = _apply_delta_error(
        comfy, patches, "model.layers.0.self_attn.q_proj.weight",
        tensors[f"{stem}.lora_A.weight"], tensors[f"{stem}.lora_B.weight"],
        float(tensors[f"{stem}.alpha"]),
    )
    assert changed
    assert err < 1e-5


def test_synthetic_music3_lm_merged_qkv_leaves_qkv_unused(tmp_path, comfy):
    src = _write_lm_slider(tmp_path / "lm.safetensors")
    tensors, dropped, unmapped = cvt.convert(str(src), "music3_lm", {})
    assert not dropped and not unmapped
    host = _clip_host(comfy, merged_qkv=True)
    key_map = comfy.lora.model_lora_keys_clip(host, {})
    patches, unused = _load_with_unused(comfy, tensors, key_map)
    assert unused
    assert all(
        any(part in key for part in (".q_proj.", ".k_proj.", ".v_proj."))
        for key in unused
    )
    assert "model.layers.0.self_attn.o_proj.weight" in patches
    stem = "text_encoders.model.layers.0.self_attn.o_proj"
    changed, err = _apply_delta_error(
        comfy, patches, "model.layers.0.self_attn.o_proj.weight",
        tensors[f"{stem}.lora_A.weight"], tensors[f"{stem}.lora_B.weight"],
        float(tensors[f"{stem}.alpha"]),
    )
    assert changed
    assert err < 1e-5


def _hub_slider(filename: str):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(HUB_REPO, filename)
    except Exception:
        return None


def test_hub_energy_transformer_comfyui_apply(comfy):
    local = _hub_slider(HUB_TF)
    if local is None:
        pytest.skip(f"could not download {HUB_TF}")
    tensors, dropped, unmapped = cvt.convert(local, "music3", cvt.read_config(local))
    assert not dropped and not unmapped
    host = _dit_host(comfy)
    key_map = comfy.lora.model_lora_keys_unet(host, {})
    assert _lora_stems(tensors) <= set(key_map)
    patches, unused = _load_with_unused(comfy, tensors, key_map)
    assert unused == []
    assert len(patches) == 72
    errors = []
    for layer in range(36):
        for kind in ("to_qkv", "to_out"):
            stem = (
                "diffusion_model.diffusion_transformer.transformer."
                f"layers.{layer}.self_attn.{kind}"
            )
            changed, err = _apply_delta_error(
                comfy, patches, f"{stem}.weight",
                tensors[f"{stem}.lora_A.weight"], tensors[f"{stem}.lora_B.weight"],
                float(tensors[f"{stem}.alpha"]),
            )
            errors.append((f"L{layer}.{kind}", changed, err))
    assert all(changed for _name, changed, _err in errors)
    assert max(err for _name, _changed, err in errors) < 1e-5


def test_hub_gender_lm_comfyui_apply(comfy):
    local = _hub_slider(HUB_LM)
    if local is None:
        pytest.skip(f"could not download {HUB_LM}")
    tensors, dropped, unmapped = cvt.convert(local, "music3_lm", cvt.read_config(local))
    assert not dropped and not unmapped
    host = _clip_host(comfy, merged_qkv=False)
    key_map = comfy.lora.model_lora_keys_clip(host, {})
    assert _lora_stems(tensors) <= set(key_map)
    patches, unused = _load_with_unused(comfy, tensors, key_map)
    assert unused == []
    assert len(patches) == 144
    stem = "text_encoders.model.layers.0.self_attn.k_proj"
    changed, err = _apply_delta_error(
        comfy, patches, "model.layers.0.self_attn.k_proj.weight",
        tensors[f"{stem}.lora_A.weight"], tensors[f"{stem}.lora_B.weight"],
        float(tensors[f"{stem}.alpha"]),
    )
    assert changed
    assert err < 1e-5
    # Honest unused-key report on the pruned/merged CLIP tree.
    merged = _clip_host(comfy, merged_qkv=True)
    _patches_m, unused_m = _load_with_unused(
        comfy, tensors, comfy.lora.model_lora_keys_clip(merged, {}))
    assert unused_m
    assert all(
        any(part in key for part in (".q_proj.", ".k_proj.", ".v_proj."))
        for key in unused_m
    )
