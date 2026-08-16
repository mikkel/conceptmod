"""conceptmod 2.0 CLI.

Examples:
    python train.py --phrase "#:0.4|human=robot:0.8|robot%human:-0.1" \
        --verify-prompt "a human walking in a city" \
        --verify-prompt "a portrait of a human" \
        --verify-prompt "a bowl of fruit on a table" \
        --out outputs/composite

    python train.py --phrase "cat~dog" --backend zimage --lora 8 ...
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from conceptmod import ops
from conceptmod.backends import load_backend
from conceptmod.model_train import train_model
from conceptmod.verify import before_after_grid


def main():
    p = argparse.ArgumentParser(description="finetuning with words")
    p.add_argument("--phrase", required=True, help="conceptmod DSL phrase")
    p.add_argument("--backend", default="sana", choices=["sana", "zimage"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--stage", default="both", choices=["encoder", "model", "both"],
                   help="both = encoder first, verify, then model (default); "
                        "encoder = text-encoder LoRA only; "
                        "model = DiT finetune only")
    p.add_argument("--encoder-iterations", type=int, default=400)
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--encoder-rank", type=int, default=8)
    p.add_argument("--encoder-strength", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--train-method", default="xattn",
                   choices=["xattn", "selfattn", "attn", "full", "noxattn"])
    p.add_argument("--lora", type=int, default=None, metavar="RANK",
                   help="train a LoRA of this rank instead of direct params")
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-prompt", default=None,
                   help="save a progress image of this prompt periodically")
    p.add_argument("--verify-prompt", action="append", default=[],
                   help="prompt for the before/after grid (repeatable)")
    p.add_argument("--verify-seeds", type=int, nargs="+", default=[42, 1234])
    p.add_argument("--exaggerate-guidance", type=float, default=None)
    p.add_argument("--erase-guidance", type=float, default=None)
    p.add_argument("--write-guidance", type=float, default=None)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--sample-guidance", type=float, default=None)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    cfg = ops.OpDefaults()
    for name in ("exaggerate_guidance", "erase_guidance", "write_guidance",
                 "sample_steps", "sample_guidance"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    backend = load_backend(args.backend, device=args.device, lora_rank=args.lora)
    print(f"backend loaded in {time.time() - t0:.0f}s")

    def verify(name):
        if args.verify_prompt:
            path = before_after_grid(
                backend, args.verify_prompt,
                os.path.join(args.out, name),
                seeds=tuple(args.verify_seeds),
                phrase=args.phrase,
                stage=args.stage,
            )
            print("verification grid:", path)

    if args.stage in ("encoder", "both"):
        from conceptmod.encoder_train import train_encoder

        train_encoder(backend, args.phrase,
                      iterations=args.encoder_iterations,
                      lr=args.encoder_lr, rank=args.encoder_rank,
                      strength=args.encoder_strength, seed=args.seed)
        verify("grid_encoder.png")

    if args.stage in ("model", "both"):
        train_model(
            backend,
            args.phrase,
            iterations=args.iterations,
            lr=args.lr,
            train_method=args.train_method,
            accumulation_steps=args.accumulation_steps,
            seed=args.seed,
            op_defaults=cfg,
            sample_prompt=args.sample_prompt,
            sample_dir=os.path.join(args.out, "progress") if args.sample_prompt else None,
        )
        verify("grid.png")

    if args.save_model:
        ext = "" if args.lora else "transformer.safetensors"
        backend.save_trained(os.path.join(args.out, ext or "lora"))
        print("saved trained weights")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
