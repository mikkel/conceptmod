"""Convert PEFT-saved conceptmod LoRAs into ComfyUI-loadable safetensors.

Walks one or more directories, finds ``adapter_model.safetensors`` (or any
safetensors whose keys look like PEFT ``lora_A``/``lora_B``), rewrites the
diffusers module paths onto the native names ComfyUI's model code uses, and
writes ``<name>_comfyui.safetensors`` beside the original. Originals are never
touched. CPU only.

The reference target format is ``anima_masterpiece_example.safetensors``:
896 bf16 tensors, keys ``diffusion_model.blocks.N.<native>.lora_{A,B}.weight``,
no alpha tensors, metadata ``{"format": "pt"}``.

    python scripts/convert_lora_comfyui.py outputs/
    python scripts/convert_lora_comfyui.py outputs/32_anima --force \
        --check-against anima_masterpiece_example.safetensors
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
}
# Backends whose ComfyUI key naming we could not verify against a known-good
# file. anima is verified; krea is derived from this repo's own loader.
UNVERIFIED = {"krea"}
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


MAPPERS = {"anima": map_anima, "krea": map_krea}


def split_lora_key(key: str):
    """Return (module_path, 'lora_A'|'lora_B') or None if not a LoRA weight."""
    if key.startswith(PEFT_PREFIX):
        key = key[len(PEFT_PREFIX):]
    for side in ("lora_A", "lora_B"):
        tail = f".{side}.weight"
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
    if any(k.startswith("diffusion_model.") for k in keys):
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


def convert(path: str, backend: str, cfg: dict):
    """Return (tensors, dropped, unmapped) for one PEFT LoRA file."""
    mapper = MAPPERS[backend]
    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    emit_alpha = rank is not None and alpha is not None and alpha != rank

    out, dropped, unmapped, alpha_paths = {}, [], [], set()
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
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
            name = f"diffusion_model.{dest}.{side}.weight"
            out[name] = f.get_tensor(key).to(TARGET_DTYPE).contiguous().cpu()
            alpha_paths.add(dest)

    # The example file carries no alpha tensors: a missing alpha means scale
    # 1.0, which is exactly alpha/r when lora_alpha == r (every backend here).
    if emit_alpha:
        for dest in sorted(alpha_paths):
            out[f"diffusion_model.{dest}.alpha"] = torch.tensor(
                float(alpha), dtype=torch.float32)
    return out, dropped, unmapped


def pattern(key: str) -> str:
    """Collapse the block index so keys from different blocks compare equal.

    Anchored on ``blocks.`` so that genuine module suffixes -- the .1/.2 of
    ``adaln_modulation_*`` -- survive and are still compared.
    """
    return re.sub(r"(blocks\.)\d+\.", r"\1N.", key)


def reference_backend(example: str):
    """Name the backend a reference ComfyUI LoRA belongs to, by its key style."""
    with safe_open(example, framework="pt") as f:
        keys = list(f.keys())
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
