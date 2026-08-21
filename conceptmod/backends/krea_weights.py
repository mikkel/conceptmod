"""Load a ComfyUI-format Krea 2 transformer, including Kitchen NVFP4.

The local NSFW turbo file is a single ``.safetensors`` with ComfyUI keys
(``model.diffusion_model.*``) and NVFP4-packed Linear weights. Training
needs a regular ``nn.Linear``, so we dequantize to bf16 and remap into
the diffusers ``Krea2Transformer2DModel`` layout.

NVFP4 layout matches comfy-kitchen: E2M1 packed uint8, hi-nibble first,
per-tensor ``weight_scale_2`` (fp32) and cuBLAS-swizzled block scales
in ``weight_scale`` (fp8 e4m3, group size 16).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

SKELETON_MODEL = "krea/Krea-2-Raw"
_PREFIX = "model.diffusion_model."
_BLOCK_SIZE = 16

# E2M1 codes 0..15. Sign lives in bit 3.
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
)

_BLOCK_SUFFIX = {
    "attn.wq": "attn.to_q.weight",
    "attn.wk": "attn.to_k.weight",
    "attn.wv": "attn.to_v.weight",
    "attn.gate": "attn.to_gate.weight",
    "attn.wo": "attn.to_out.0.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate": "ff.gate.weight",
    "mlp.up": "ff.up.weight",
    "mlp.down": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
    "mod.lin": "scale_shift_table",
}

_TOP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}


def resolve_local_transformer(model_id: str) -> Path | None:
    """Return a local ``.safetensors`` path, or None for a hub id."""
    raw = Path(model_id)
    candidates = [raw]
    if raw.suffix != ".safetensors":
        candidates.append(Path(f"{model_id}.safetensors"))
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def looks_turbo(path: Path) -> bool:
    return "turbo" in path.name.lower()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(matrix: torch.Tensor, flatten: bool = False) -> torch.Tensor:
    """cuBLAS 128x4 scale swizzle. Inverse of ``from_blocked``."""
    rows, cols = matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows, padded_cols = n_row_blocks * 128, n_col_blocks * 4
    padded = matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = matrix.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols] = matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Undo the cuBLAS tiled scale layout back to ``(num_rows, num_cols)``."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows, padded_cols = n_row_blocks * 128, n_col_blocks * 4
    step1 = blocked.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    unblocked = step4.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


def pack_nvfp4(values: torch.Tensor, hi_first: bool = True) -> torch.Tensor:
    """Pack two E2M1 codes per uint8. Inverse of the unpack in ``dequantize_nvfp4``."""
    flat = values.contiguous().view(-1)
    if hi_first:
        packed = (flat[0::2] << 4) | flat[1::2]
    else:
        packed = (flat[1::2] << 4) | flat[0::2]
    return packed.view(*values.shape[:-1], values.shape[-1] // 2)


def dequantize_nvfp4(
    packed: torch.Tensor,
    tensor_scale: torch.Tensor,
    block_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    hi_first: bool = True,
) -> torch.Tensor:
    """Dequantize a Kitchen / comfy-kitchen NVFP4 Linear weight."""
    lut = _E2M1.to(device=packed.device, dtype=dtype)
    lo = packed & 0x0F
    hi = packed >> 4
    if hi_first:
        codes = torch.stack([hi, lo], dim=-1).reshape(*packed.shape[:-1], -1)
    else:
        codes = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    values = lut[codes.long()]

    rows, cols = values.shape
    n_blocks = cols // _BLOCK_SIZE
    values = values.reshape(rows, n_blocks, _BLOCK_SIZE)
    scales = from_blocked(block_scale, rows, n_blocks).to(dtype=dtype)
    total = tensor_scale.to(device=values.device, dtype=dtype) * scales
    return (values * total.unsqueeze(-1)).reshape(rows, cols)


def convert_comfy_key(comfy_key: str) -> str | None:
    """Map a ComfyUI tensor name onto a diffusers Krea2 name.

    Scale companions (``weight_scale``, ``weight_scale_2``) return None.
    """
    if comfy_key.endswith((".weight_scale", ".weight_scale_2")):
        return None
    key = comfy_key
    if key.startswith(_PREFIX):
        key = key[len(_PREFIX):]
    if key in _TOP:
        return _TOP[key]
    for prefix, dest in (
        ("blocks.", "transformer_blocks."),
        ("txtfusion.layerwise_blocks.", "text_fusion.layerwise_blocks."),
        ("txtfusion.refiner_blocks.", "text_fusion.refiner_blocks."),
    ):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        idx, _, tail = rest.partition(".")
        if not idx.isdigit():
            raise KeyError(f"unmapped krea comfy key: {comfy_key}")
        if tail.endswith(".weight"):
            tail = tail[: -len(".weight")]
        suffix = _BLOCK_SUFFIX.get(tail)
        if suffix is None:
            raise KeyError(f"unmapped krea comfy key: {comfy_key}")
        return f"{dest}{idx}.{suffix}"
    raise KeyError(f"unmapped krea comfy key: {comfy_key}")


def _reshape_table(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if name.endswith("scale_shift_table") and tensor.ndim == 1:
        hidden = tensor.numel() // 6
        return tensor.view(6, hidden)
    return tensor


def load_comfy_krea_state_dict(
    path: str | Path, dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Read a ComfyUI Krea 2 file and return a diffusers state dict."""
    path = Path(path)
    out: dict[str, torch.Tensor] = {}
    n_dequant = 0
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        keyset = set(keys)
        for key in keys:
            if key.endswith((".weight_scale", ".weight_scale_2")):
                continue
            dest = convert_comfy_key(key)
            tensor = f.get_tensor(key)
            scale_key = f"{key}_scale"
            scale2_key = f"{key}_scale_2"
            if scale_key in keyset and scale2_key in keyset:
                tensor = dequantize_nvfp4(
                    tensor,
                    f.get_tensor(scale2_key),
                    f.get_tensor(scale_key),
                    dtype=dtype,
                )
                n_dequant += 1
            else:
                keep_fp32 = dest.endswith((
                    ".norm.weight", ".norm1.weight", ".norm2.weight",
                    ".norm_q.weight", ".norm_k.weight",
                )) or dest.endswith("txt_in.norm.weight")
                tensor = tensor.to(dtype=torch.float32 if keep_fp32 else dtype)
            out[dest] = _reshape_table(dest, tensor)
    print(f"krea weights: {len(out)} tensors from {path.name} ({n_dequant} nvfp4 dequant)")
    return out


def load_comfy_krea_transformer(
    path: str | Path,
    skeleton: str = SKELETON_MODEL,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build a ``Krea2Transformer2DModel`` from a ComfyUI / NVFP4 file."""
    from diffusers import Krea2Transformer2DModel

    config = Krea2Transformer2DModel.load_config(skeleton, subfolder="transformer")
    sd = load_comfy_krea_state_dict(path, dtype=dtype)
    with torch.device("meta"):
        transformer = Krea2Transformer2DModel.from_config(config)
    missing, unexpected = transformer.load_state_dict(sd, assign=True)
    del sd
    if missing:
        raise RuntimeError(f"krea transformer missing {len(missing)} keys, e.g. {missing[:8]}")
    if unexpected:
        raise RuntimeError(
            f"krea transformer unexpected {len(unexpected)} keys, e.g. {unexpected[:8]}")
    return transformer


def quantization_layers(path: str | Path) -> dict:
    """Return the Kitchen ``_quantization_metadata.layers`` map, if present."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    raw = meta.get("_quantization_metadata")
    if not raw:
        return {}
    return json.loads(raw).get("layers", {})
