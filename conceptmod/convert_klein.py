"""Flux.2 Klein PEFT → ComfyUI names (new file; does not rewrite other mappers).

Source of truth: live ComfyUI ``comfy/utils.py`` ``flux_to_diffusers`` and
``comfy/lora.py``'s Flux handler (PR #11981 added the Flux2 lines
``attn.to_qkv_mlp_proj → linear1`` and ``attn.to_out → linear2``).

Double-stream PEFT ``to_q``/``to_k``/``to_v`` fuse into Comfy
``img_attn.qkv``; ``add_*_proj`` fuse into ``txt_attn.qkv``. Official Klein
single-stream already ships fused ``to_qkv_mlp_proj`` → ``linear1``. Separate
Flux.1-style single ``to_q``/``to_k``/``to_v`` also fuse into ``linear1``.

Not verified against a known-good loading LoRA file (same honesty as Music 3).
"""
from __future__ import annotations

import re

UNVERIFIED_WARNING = (
    "Klein names come from ComfyUI flux_to_diffusers / lora.py Flux handler "
    "(PR #11981), not a known-good loading LoRA"
)

# Temporary dests for separate QKV; fuse_qkv rewrites them to the fused Linear.
KLEIN_DOUBLE = {
    "attn.to_q": "img_attn.to_q",
    "attn.to_k": "img_attn.to_k",
    "attn.to_v": "img_attn.to_v",
    "attn.to_out.0": "img_attn.proj",
    "attn.add_q_proj": "txt_attn.to_q",
    "attn.add_k_proj": "txt_attn.to_k",
    "attn.add_v_proj": "txt_attn.to_v",
    "attn.to_add_out": "txt_attn.proj",
}
KLEIN_SINGLE = {
    "attn.to_qkv_mlp_proj": "linear1",
    "attn.to_out": "linear2",
    "attn.to_q": "linear1_to_q",
    "attn.to_k": "linear1_to_k",
    "attn.to_v": "linear1_to_v",
    "proj_mlp": "linear1_proj_mlp",
    "proj_out": "linear2",
}

_DOUBLE_QKV = re.compile(
    r"^diffusion_model\.(double_blocks\.\d+\.(?:img_attn|txt_attn)\.)"
    r"(to_q|to_k|to_v)\.(lora_A|lora_B)\.weight$"
)
_SINGLE_QKV = re.compile(
    r"^diffusion_model\.(single_blocks\.\d+\.)"
    r"linear1_to_(q|k|v)\.(lora_A|lora_B)\.weight$"
)


def map_klein(module: str):
    """Map a PEFT Flux2Transformer2DModel module onto its ComfyUI Flux2 name."""
    if module.startswith("transformer_blocks."):
        idx, _, tail = module[len("transformer_blocks."):].partition(".")
        if idx.isdigit() and tail in KLEIN_DOUBLE:
            return f"double_blocks.{idx}.{KLEIN_DOUBLE[tail]}"
        return None
    if module.startswith("single_transformer_blocks."):
        idx, _, tail = module[len("single_transformer_blocks."):].partition(".")
        if idx.isdigit() and tail in KLEIN_SINGLE:
            return f"single_blocks.{idx}.{KLEIN_SINGLE[tail]}"
        return None
    return None


def detect_from_stems(stems):
    """Return (backend, why) for Klein-only PEFT stems, else None.

    Must run before the generic ``transformer_blocks.N.attn.*`` → krea/qwen
    heuristics: Klein double-stream also uses ``attn.add_q_proj``.
    """
    if any(s.startswith("single_transformer_blocks.") for s in stems):
        return "klein", "key heuristic (single_transformer_blocks.*)"
    if any("to_qkv_mlp_proj" in s for s in stems):
        return "klein", "key heuristic (to_qkv_mlp_proj)"
    return None


def looks_like_comfy(keys) -> bool:
    return any(".img_attn.qkv." in k or ".single_blocks." in k for k in keys)


def _fuse_group(out, dest_alphas, block_diag_up, target_dtype, groups, order,
                dest_of_proj, fused_dest, extra_unmapped):
    for stem, projs in groups.items():
        if set(projs) != set(order):
            extra_unmapped.append(f"incomplete QKV under {stem}")
            continue
        parts = []
        for proj in order:
            dest = dest_of_proj(stem, proj)
            down = out[projs[proj]["lora_A"]].float()
            up = out[projs[proj]["lora_B"]].float()
            parts.append((down, up, dest_alphas.get(dest, float(down.shape[0]))))
        fused_down, fused_up, fused_alpha = block_diag_up(parts)
        for proj in order:
            for side in ("lora_A", "lora_B"):
                del out[projs[proj][side]]
            dest_alphas.pop(dest_of_proj(stem, proj), None)
        dest = fused_dest(stem)
        out[f"diffusion_model.{dest}.lora_A.weight"] = (
            fused_down.to(target_dtype).contiguous())
        out[f"diffusion_model.{dest}.lora_B.weight"] = (
            fused_up.to(target_dtype).contiguous())
        dest_alphas[dest] = fused_alpha


def fuse_qkv(out, dest_alphas, block_diag_up, target_dtype, default_alpha=None):
    """Fuse separate QKV LoRAs onto Comfy ``img_attn.qkv`` / ``linear1``.

    ``default_alpha`` is PEFT ``lora_alpha`` when the file has no per-module
    ``.alpha`` tensors. Missing dest alphas then fall back to rank, which
    under-scales fused QKV when ``lora_alpha != r``.
    """
    if default_alpha is not None:
        for key in list(out):
            if _DOUBLE_QKV.match(key) or _SINGLE_QKV.match(key):
                dest = key[len("diffusion_model."):-len(".lora_A.weight")]
                if dest.endswith(".lora_B.weight"):
                    dest = key[len("diffusion_model."):-len(".lora_B.weight")]
                dest_alphas.setdefault(dest, float(default_alpha))
    extra_unmapped = []
    double_groups = {}
    single_groups = {}
    for key in list(out):
        match = _DOUBLE_QKV.match(key)
        if match:
            stem, proj, side = match.group(1), match.group(2), match.group(3)
            double_groups.setdefault(stem, {}).setdefault(proj, {})[side] = key
            continue
        match = _SINGLE_QKV.match(key)
        if match:
            stem, proj, side = match.group(1), match.group(2), match.group(3)
            single_groups.setdefault(stem, {}).setdefault(proj, {})[side] = key

    _fuse_group(
        out, dest_alphas, block_diag_up, target_dtype, double_groups,
        ("to_q", "to_k", "to_v"),
        lambda stem, proj: f"{stem}{proj}",
        lambda stem: f"{stem}qkv",
        extra_unmapped,
    )

    # Separate single-block q/k/v cannot coexist with an already-fused linear1.
    for stem in list(single_groups):
        if f"diffusion_model.{stem}linear1.lora_A.weight" in out:
            extra_unmapped.append(
                f"single-block {stem} has both fused linear1 and separate QKV")
            del single_groups[stem]
    _fuse_group(
        out, dest_alphas, block_diag_up, target_dtype, single_groups,
        ("q", "k", "v"),
        lambda stem, proj: f"{stem}linear1_to_{proj}",
        lambda stem: f"{stem}linear1",
        extra_unmapped,
    )
    return extra_unmapped


def register(mappers, model_classes, unverified, host, recommended_range, fused_qkv):
    """Hook Klein into the shared convert tables without rewriting them."""
    mappers["klein"] = map_klein
    model_classes["Flux2Transformer2DModel"] = "klein"
    unverified.add("klein")
    host["klein"] = "dit"
    recommended_range["klein"] = [0.8, 1.2]
    fused_qkv.add("klein")
