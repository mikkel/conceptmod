"""Convert PEFT-saved conceptmod LoRAs into ComfyUI-loadable safetensors.

Walks one or more directories, finds ``adapter_model.safetensors`` (or any
safetensors whose keys look like PEFT ``lora_A``/``lora_B``), rewrites the
diffusers module paths onto the native names ComfyUI's model code uses, and
writes ``<name>_comfyui.safetensors`` beside the original. Originals are never
touched. CPU only.

Music 3 concept sliders (LoRANetwork ``lora_down``/``lora_up`` under
``lora_unet-`` / ``lora_te-``) use the same script: detect ``music3`` /
``music3_lm`` and, for the DiT, fuse ``to_q``/``to_k``/``to_v`` into
ComfyUI's ``self_attn.to_qkv``. Anima/Krea mapping is unchanged.

The reference target format is ``anima_masterpiece_example.safetensors``:
896 bf16 tensors, keys ``diffusion_model.blocks.N.<native>.lora_{A,B}.weight``,
no alpha tensors, metadata ``{\"format\": \"pt\"}``.

    python scripts/convert_lora_comfyui.py outputs/
    python scripts/convert_lora_comfyui.py outputs/32_anima --force \
        --check-against anima_masterpiece_example.safetensors
    python scripts/convert_lora_comfyui.py path/to/energy_unit_last.safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PEFT_PREFIX = "base_model.model."
OUT_SUFFIX = "_comfyui.safetensors"
TARGET_DTYPE = torch.bfloat16
METADATA = {"format": "pt"}

# ---------------------------------------------------------------- key mapping

# diffusers CosmosTransformer3DModel -> native Cosmos-Predict2 / Anima names.
# Verified against every one of the 896 keys in the masterpiece example.
ANIMA_BLOCK = {
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "norm1.linear_1": "adaln_modulation_self_attn.1",
    "norm1.linear_2": "adaln_modulation_self_attn.2",
    "norm2.linear_1": "adaln_modulation_cross_attn.1",
    "norm2.linear_2": "adaln_modulation_cross_attn.2",
    "norm3.linear_1": "adaln_modulation_mlp.1",
    "norm3.linear_2": "adaln_modulation_mlp.2",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
}
# Widened target_modules can match these; the example file has no counterpart
# for anything outside blocks.N, so they are dropped rather than guessed at.
ANIMA_DROP = ("patch_embed.", "time_embed.", "norm_out.", "proj_out.")

# Krea2Transformer2DModel -> ComfyUI, the inverse of krea_weights._BLOCK_SUFFIX.
KREA_BLOCK = {
    "attn.to_q": "attn.wq",
    "attn.to_k": "attn.wk",
    "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate",
    "attn.to_out.0": "attn.wo",
    "ff.gate": "mlp.gate",
    "ff.up": "mlp.up",
    "ff.down": "mlp.down",
}
KREA_STEMS = (
    ("transformer_blocks.", "blocks."),
    ("text_fusion.layerwise_blocks.", "txtfusion.layerwise_blocks."),
    ("text_fusion.refiner_blocks.", "txtfusion.refiner_blocks."),
)

# base_model_class in adapter_config.json -> our backend name.
MODEL_CLASSES = {
    "CosmosTransformer3DModel": "anima",
    "Krea2Transformer2DModel": "krea",
    "ZImageTransformer2DModel": "zimage",
    "SanaTransformer2DModel": "sana",
    "MiniMaxMusic3Transformer1DModel": "music3",
}
# Backends whose ComfyUI key naming we could not verify against a known-good
# file. anima is verified; krea is derived from this repo's own loader.
# music3 is derived from ComfyUI MiniMax Music 3 module names, not a loading file.
UNVERIFIED = {"krea", "music3", "music3_lm"}
UNSUPPORTED = {
    "zimage": "ComfyUI key naming for Z-Image is unknown; refusing to guess",
    "sana": "ComfyUI key naming for Sana is unverified; refusing to guess",
}

DROP = object()


def map_anima(module: str):
    """Map a diffusers Cosmos module path onto its native name."""
    if module.startswith("transformer_blocks."):
        idx, _, tail = module[len("transformer_blocks."):].partition(".")
        if idx.isdigit() and tail in ANIMA_BLOCK:
            return f"blocks.{idx}.{ANIMA_BLOCK[tail]}"
        return None
    if module.startswith(ANIMA_DROP):
        return DROP
    return None


def map_krea(module: str):
    """Map a diffusers Krea2 module path onto its ComfyUI name."""
    for stem, dest in KREA_STEMS:
        if not module.startswith(stem):
            continue
        idx, _, tail = module[len(stem):].partition(".")
        if idx.isdigit() and tail in KREA_BLOCK:
            return f"{dest}{idx}.{KREA_BLOCK[tail]}"
        return None
    return None


# sliders-conceptmod LoRANetwork (delimiter="-") → ComfyUI MiniMax Music 3 DiT.
# Native names: comfy/ldm/minimax_music/dit.py (fused self_attn.to_qkv).
MUSIC3_TF_PREFIX = "lora_unet-"
MUSIC3_LM_PREFIX = "lora_te-"
MUSIC3_QKV = {
    "attn-to_q": "self_attn.to_q",
    "attn-to_k": "self_attn.to_k",
    "attn-to_v": "self_attn.to_v",
}
MUSIC3_BLOCK = {
    "attn-to_out-0": "self_attn.to_out",
    "ff_in": "ff.ff.0.proj",
    "ff_out": "ff.ff.2",
}
MUSIC3_ROOT = {
    "proj_in": "diffusion_transformer.transformer.project_in",
    "proj_out": "diffusion_transformer.transformer.project_out",
    "preprocess_conv": "diffusion_transformer.preprocess_conv",
    "postprocess_conv": "diffusion_transformer.postprocess_conv",
    "time_embed-linear_1": "diffusion_transformer.to_timestep_embed.0",
    "time_embed-linear_2": "diffusion_transformer.to_timestep_embed.2",
}
MUSIC3_BLOCK_RE = re.compile(r"^transformer_blocks-(\d+)-(.+)$")
MUSIC3_LM_RE = re.compile(
    r"^model-layers-(\d+)-self_attn-(q_proj|k_proj|v_proj|o_proj)$")
_MUSIC3_QKV_KEY = re.compile(
    r"^(diffusion_model\.diffusion_transformer\.transformer\.layers\.\d+"
    r"\.self_attn\.)(to_q|to_k|to_v)\.(lora_A|lora_B)\.weight$"
)


def map_music3(module: str):
    """Map a LoRANetwork Music 3 transformer module onto its ComfyUI DiT name."""
    if module.startswith(MUSIC3_TF_PREFIX):
        module = module[len(MUSIC3_TF_PREFIX):]
    if module in MUSIC3_ROOT:
        return MUSIC3_ROOT[module]
    match = MUSIC3_BLOCK_RE.match(module)
    if not match:
        return None
    idx, tail = match.group(1), match.group(2)
    if tail in MUSIC3_QKV:
        return f"diffusion_transformer.transformer.layers.{idx}.{MUSIC3_QKV[tail]}"
    if tail in MUSIC3_BLOCK:
        return f"diffusion_transformer.transformer.layers.{idx}.{MUSIC3_BLOCK[tail]}"
    return None


def map_music3_lm(module: str):
    """Map a LoRANetwork Music 3 LM module onto the ComfyUI CLIP stem."""
    if module.startswith(MUSIC3_LM_PREFIX):
        module = module[len(MUSIC3_LM_PREFIX):]
    match = MUSIC3_LM_RE.match(module)
    if not match:
        return None
    return f"model.layers.{match.group(1)}.self_attn.{match.group(2)}"


MAPPERS = {
    "anima": map_anima,
    "krea": map_krea,
    "music3": map_music3,
    "music3_lm": map_music3_lm,
}

# LoRANetwork stores down/up; PEFT stores A/B. Normalize to lora_A / lora_B.
_LORA_SIDES = (
    (".lora_A.weight", "lora_A"),
    (".lora_B.weight", "lora_B"),
    (".lora_down.weight", "lora_A"),
    (".lora_up.weight", "lora_B"),
    (".lora.down.weight", "lora_A"),
    (".lora.up.weight", "lora_B"),
)


def split_lora_key(key: str):
    """Return (module_path, 'lora_A'|'lora_B') or None if not a LoRA weight."""
    if key.startswith(PEFT_PREFIX):
        key = key[len(PEFT_PREFIX):]
    for tail, side in _LORA_SIDES:
        if key.endswith(tail):
            return key[: -len(tail)], side
    return None


# ------------------------------------------------------------------ discovery


def classify(path: str):
    """Return ('lora', keys) for a PEFT LoRA, or ('<reason>', None) to skip."""
    try:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception as exc:                    # not a safetensors file at all
        return f"unreadable ({exc})", None
    if not keys:
        return "empty file", None
    if any(k.startswith(("diffusion_model.", "text_encoders.")) for k in keys):
        return "already ComfyUI format", None
    if not any(split_lora_key(k) for k in keys):
        return "no lora_A/lora_B keys (full checkpoint?)", None
    return "lora", keys


def find_loras(roots):
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(os.path.abspath(root))
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                if not name.endswith(".safetensors"):
                    continue
                if name.endswith(OUT_SUFFIX):
                    continue
                found.append(os.path.join(dirpath, name))
    return sorted(dict.fromkeys(found))


def read_config(path: str):
    """Load the adapter_config.json sitting next to a LoRA, if any."""
    cfg_path = os.path.join(os.path.dirname(path), "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path) as fh:
        return json.load(fh)


def detect_backend(cfg, keys):
    """Name the backend from adapter_config.json, falling back to key shape."""
    cls = (cfg.get("auto_mapping") or {}).get("base_model_class")
    if cls in MODEL_CLASSES:
        return MODEL_CLASSES[cls], f"adapter_config base_model_class={cls}"
    stems = {split_lora_key(k)[0] for k in keys if split_lora_key(k)}
    if any(s.startswith(MUSIC3_TF_PREFIX) or s.startswith("transformer_blocks-")
           for s in stems):
        return "music3", "key heuristic (LoRANetwork music3 transformer)"
    if any(s.startswith(MUSIC3_LM_PREFIX) or "-self_attn-q_proj" in s
           for s in stems):
        return "music3_lm", "key heuristic (LoRANetwork music3 LM)"
    if cfg.get("kind") == "language_model":
        return "music3_lm", "sidecar kind=language_model"
    if cfg.get("kind") == "transformer":
        return "music3", "sidecar kind=transformer"
    if any(s.startswith("text_fusion.") for s in stems):
        return "krea", "key heuristic (text_fusion.*)"
    if any(re.match(r"transformer_blocks\.\d+\.attn\.", s) for s in stems):
        return "krea", "key heuristic (transformer_blocks.N.attn.*)"
    if any(s.startswith(("noise_refiner.", "context_refiner.")) for s in stems):
        return "zimage", "key heuristic (noise_refiner/context_refiner)"
    if any(re.match(r"layers\.\d+\.attention\.", s) for s in stems):
        return "zimage", "key heuristic (layers.N.attention.*)"
    # anima and sana share transformer_blocks.N.attn1/attn2 -- ambiguous.
    return None, "ambiguous; pass --backend"


# ----------------------------------------------------------------- conversion


def _block_diag_up(parts):
    """Fuse independent LoRAs that share an input dim onto one Linear."""
    downs = [down for down, _up, _alpha in parts]
    ups = [up for _down, up, _alpha in parts]
    ranks = [int(down.shape[0]) for down in downs]
    out_sizes = [int(up.shape[0]) for up in ups]
    total_rank = sum(ranks)
    scales = [float(alpha) / max(rank, 1)
              for (_d, _u, alpha), rank in zip(parts, ranks)]
    shared = all(abs(scale - scales[0]) <= 1e-6 for scale in scales)
    fused_down = torch.cat(downs, dim=0)
    fused_up = ups[0].new_zeros((sum(out_sizes), total_rank))
    rank_offset = out_offset = 0
    for up, rank, out_dim, scale in zip(ups, ranks, out_sizes, scales):
        fused_up[out_offset:out_offset + out_dim,
                 rank_offset:rank_offset + rank] = up if shared else up * scale
        rank_offset += rank
        out_offset += out_dim
    fused_alpha = (scales[0] * total_rank) if shared else float(total_rank)
    return fused_down, fused_up, fused_alpha


def _fuse_music3_qkv(out, dest_alphas):
    """Replace separate to_q/to_k/to_v with ComfyUI's fused to_qkv."""
    groups = {}
    for key in list(out):
        match = _MUSIC3_QKV_KEY.match(key)
        if not match:
            continue
        stem, proj, side = match.group(1), match.group(2), match.group(3)
        groups.setdefault(stem, {}).setdefault(proj, {})[side] = key
    extra_unmapped = []
    for stem, projs in groups.items():
        if set(projs) != {"to_q", "to_k", "to_v"}:
            extra_unmapped.append(f"incomplete QKV under {stem}")
            continue
        parts = []
        for proj in ("to_q", "to_k", "to_v"):
            dest = f"{stem}{proj}"[len("diffusion_model."):]
            down = out[projs[proj]["lora_A"]].float()
            up = out[projs[proj]["lora_B"]].float()
            parts.append((down, up, dest_alphas.get(dest, float(down.shape[0]))))
        fused_down, fused_up, fused_alpha = _block_diag_up(parts)
        for proj in ("to_q", "to_k", "to_v"):
            for side in ("lora_A", "lora_B"):
                del out[projs[proj][side]]
            dest_alphas.pop(f"{stem}{proj}"[len("diffusion_model."):], None)
        dest = f"{stem}to_qkv"[len("diffusion_model."):]
        out[f"diffusion_model.{dest}.lora_A.weight"] = (
            fused_down.to(TARGET_DTYPE).contiguous())
        out[f"diffusion_model.{dest}.lora_B.weight"] = (
            fused_up.to(TARGET_DTYPE).contiguous())
        dest_alphas[dest] = fused_alpha
    return extra_unmapped


def convert(path: str, backend: str, cfg: dict):
    """Return (tensors, dropped, unmapped) for one PEFT LoRA file."""
    mapper = MAPPERS[backend]
    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    emit_alpha = rank is not None and alpha is not None and alpha != rank
    key_prefix = "text_encoders" if backend == "music3_lm" else "diffusion_model"

    out, dropped, unmapped, alpha_paths = {}, [], [], set()
    file_alphas = {}
    dest_from_module = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".alpha"):
                module = key[: -len(".alpha")]
                file_alphas[module] = float(
                    f.get_tensor(key).detach().float().cpu().reshape(-1)[0])
                continue
            parts = split_lora_key(key)
            if parts is None:
                unmapped.append(key)
                continue
            module, side = parts
            dest = mapper(module)
            if dest is DROP:
                dropped.append(key)
                continue
            if dest is None:
                unmapped.append(key)
                continue
            name = f"{key_prefix}.{dest}.{side}.weight"
            out[name] = f.get_tensor(key).to(TARGET_DTYPE).contiguous().cpu()
            alpha_paths.add(dest)
            dest_from_module[module] = dest

    dest_alphas = {}
    for module, dest in dest_from_module.items():
        if module in file_alphas:
            dest_alphas[dest] = file_alphas[module]
    if backend == "music3":
        unmapped.extend(_fuse_music3_qkv(out, dest_alphas))

    # The example file carries no alpha tensors: a missing alpha means scale
    # 1.0, which is exactly alpha/r when lora_alpha == r (every backend here).
    # Music 3 sliders carry per-module alpha (unit-normalized files bake scale
    # into it), so those are always written.
    if dest_alphas:
        for dest, value in dest_alphas.items():
            out[f"{key_prefix}.{dest}.alpha"] = torch.tensor(
                float(value), dtype=torch.float32)
    elif emit_alpha:
        for dest in sorted(alpha_paths):
            out[f"{key_prefix}.{dest}.alpha"] = torch.tensor(
                float(alpha), dtype=torch.float32)
    return out, dropped, unmapped


def pattern(key: str) -> str:
    """Collapse the block index so keys from different blocks compare equal.

    Anchored on ``blocks.`` so that genuine module suffixes -- the .1/.2 of
    ``adaln_modulation_*`` -- survive and are still compared.
    """
    key = re.sub(r"(blocks\.)\d+\.", r"\1N.", key)
    return re.sub(r"(layers\.)\d+\.", r"\1N.", key)


def reference_backend(example: str):
    """Name the backend a reference ComfyUI LoRA belongs to, by its key style."""
    with safe_open(example, framework="pt") as f:
        keys = list(f.keys())
    if any(".self_attn.to_qkv." in k for k in keys):
        return "music3"
    if any(k.startswith("text_encoders.model.layers.") for k in keys):
        return "music3_lm"
    if any(".self_attn.q_proj." in k or ".cross_attn.q_proj." in k for k in keys):
        return "anima"
    if any(".attn.wq." in k for k in keys):
        return "krea"
    return None


def check_against(tensors, example: str, backend: str):
    """Assert every converted key pattern exists in the reference file.

    Alpha keys are exempt: the reference has alpha == r so it carries none,
    while we emit them only in the alpha != r case.
    """
    ref = reference_backend(example)
    if ref != backend:
        print(f"    check skipped: reference is {ref}, this file is {backend}")
        return True
    with safe_open(example, framework="pt") as f:
        known = {pattern(k) for k in f.keys()}
    mine = {pattern(k) for k in tensors if not k.endswith(".alpha")}
    bad, ok = sorted(mine - known), sorted(mine & known)
    print(f"    check: {len(ok)} of {len(ok) + len(bad)} key patterns present "
          f"in {os.path.basename(example)}")
    for p in bad:
        print(f"      NOT IN EXAMPLE: {p}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", default=["."],
                    help="directories to walk (default: .)")
    ap.add_argument("--backend", choices=sorted(MAPPERS) + sorted(UNSUPPORTED),
                    help="override backend detection")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing _comfyui output")
    ap.add_argument("--check-against", metavar="FILE",
                    help="verify key patterns against a reference ComfyUI LoRA")
    ap.add_argument("--skip-unsupported", action="store_true",
                    help="warn instead of failing on unsupported backends")
    args = ap.parse_args()
    roots = args.roots or ["."]

    candidates = find_loras(roots)
    print(f"scanning {len(roots)} root(s): {len(candidates)} safetensors found")
    failures, written = [], 0

    for path in candidates:
        kind, keys = classify(path)
        if keys is None:
            print(f"skip {path}\n    {kind}")
            continue
        out_path = path[: -len(".safetensors")] + OUT_SUFFIX
        print(f"convert {path}")
        if os.path.exists(out_path) and not args.force:
            print(f"    exists, skipping (use --force): {out_path}")
            continue

        cfg = read_config(path)
        backend = args.backend
        why = "--backend"
        if backend is None:
            backend, why = detect_backend(cfg, keys)
        if backend is None:
            print(f"    ERROR: backend undetermined -- {why}")
            failures.append(path)
            continue
        print(f"    backend={backend} ({why}) r={cfg.get('r')} "
              f"alpha={cfg.get('lora_alpha')}")
        if backend in UNSUPPORTED:
            msg = UNSUPPORTED[backend]
            print(f"    {'WARN' if args.skip_unsupported else 'ERROR'}: {msg}")
            if not args.skip_unsupported:
                failures.append(path)
            continue
        if backend in UNVERIFIED:
            if backend in ("music3", "music3_lm"):
                print("    WARNING: Music 3 names come from ComfyUI "
                      "minimax_music/dit.py and model_lora_keys_clip, "
                      "not a known-good loading LoRA")
            else:
                print("    WARNING: key naming for this backend is derived from "
                      "this repo's loader, not verified against a loading file")

        tensors, dropped, unmapped = convert(path, backend, cfg)
        print(f"    {len(tensors)} keys converted from {len(keys)} source keys")
        if dropped:
            print(f"    {len(dropped)} dropped (no ComfyUI counterpart): "
                  f"{', '.join(sorted(dropped)[:3])}"
                  f"{' ...' if len(dropped) > 3 else ''}")
        if unmapped:
            print(f"    ERROR: {len(unmapped)} unmappable keys:")
            for key in sorted(unmapped)[:10]:
                print(f"      {key}")
            if len(unmapped) > 10:
                print(f"      ... and {len(unmapped) - 10} more")
            failures.append(path)
            continue
        if not tensors:
            print("    ERROR: nothing to write")
            failures.append(path)
            continue
        if args.check_against and not check_against(
                tensors, args.check_against, backend):
            failures.append(path)
            continue

        save_file(tensors, out_path, metadata=METADATA)
        size = os.path.getsize(out_path)
        print(f"    wrote {out_path} ({size / 2**20:.1f} MiB, bf16)")
        written += 1

    print(f"\n{written} written, {len(failures)} failed")
    if failures:
        for path in failures:
            print(f"  FAILED {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
