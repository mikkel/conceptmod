import torch

from conceptmod.backends.krea_weights import (
    convert_comfy_key,
    dequantize_nvfp4,
    from_blocked,
    looks_turbo,
    pack_nvfp4,
    resolve_local_transformer,
    to_blocked,
)


def test_blocked_roundtrip():
    x = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8)
    y = from_blocked(to_blocked(x), 128, 8)
    assert torch.equal(y, x)


def test_nvfp4_roundtrip_small():
    # 128x32 so scales are (128, 2) and the 128-row swizzle is exact.
    torch.manual_seed(0)
    weight = torch.randn(128, 32)
    block = 16
    grouped = weight.reshape(128, -1, block)
    amax = grouped.abs().amax(dim=-1)
    tensor_scale = weight.abs().amax() / (448.0 * 6.0)
    block_fp8 = (amax / tensor_scale / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    total = tensor_scale * block_fp8.float()
    codes = (grouped / total.unsqueeze(-1)).clamp(-6, 6)
    # snap to the E2M1 grid
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    signed = codes.reshape(128, 32)
    mag = signed.abs()
    idx = (mag.unsqueeze(-1) - lut).abs().argmin(dim=-1)
    snapped = lut[idx] * signed.sign().where(signed != 0, torch.ones_like(signed))
    # encode
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    raw_codes = (snapped.unsqueeze(-1) - e2m1).abs().argmin(dim=-1).to(torch.uint8)
    packed = pack_nvfp4(raw_codes)
    recovered = dequantize_nvfp4(packed, tensor_scale, to_blocked(block_fp8), dtype=torch.float32)
    expected = snapped.reshape(128, -1, block) * (tensor_scale * block_fp8.float()).unsqueeze(-1)
    expected = expected.reshape(128, 32)
    assert recovered.shape == weight.shape
    assert (recovered - expected).abs().max() < 1e-5


def test_convert_comfy_keys():
    assert convert_comfy_key("model.diffusion_model.blocks.3.attn.wq.weight") == (
        "transformer_blocks.3.attn.to_q.weight")
    assert convert_comfy_key("model.diffusion_model.blocks.0.mod.lin") == (
        "transformer_blocks.0.scale_shift_table")
    assert convert_comfy_key("model.diffusion_model.blocks.2.prenorm.scale") == (
        "transformer_blocks.2.norm1.weight")
    assert convert_comfy_key(
        "model.diffusion_model.txtfusion.layerwise_blocks.1.attn.wo.weight"
    ) == "text_fusion.layerwise_blocks.1.attn.to_out.0.weight"
    assert convert_comfy_key("model.diffusion_model.last.modulation.lin") == (
        "final_layer.scale_shift_table")
    assert convert_comfy_key("model.diffusion_model.txtmlp.0.scale") == "txt_in.norm.weight"
    assert convert_comfy_key("model.diffusion_model.blocks.0.attn.wq.weight_scale") is None


def test_resolve_local_and_turbo(tmp_path):
    missing = tmp_path / "nope"
    assert resolve_local_transformer(str(missing)) is None
    f = tmp_path / "kreaturboft_nvfp4.safetensors"
    f.write_bytes(b"x")
    assert resolve_local_transformer(str(f.with_suffix(""))) == f.resolve()
    assert looks_turbo(f)
    assert not looks_turbo(tmp_path / "krea2_raw.safetensors")


def test_local_file_maps_onto_official_keys():
    import json
    from pathlib import Path

    from safetensors import safe_open

    local = Path("models/kreaturboft_nvfp4.safetensors")
    index = Path(
        "/ml2/trained/huggingface/hub/models--krea--Krea-2-Raw/"
        "snapshots/6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603/transformer/"
        "diffusion_pytorch_model.safetensors.index.json"
    )
    if not local.is_file() or not index.is_file():
        import pytest
        pytest.skip("krea weights not on disk")
    official = set(json.loads(index.read_text())["weight_map"])
    with safe_open(str(local), framework="pt") as fh:
        converted = {convert_comfy_key(k) for k in fh.keys()}
    converted.discard(None)
    assert converted == official
