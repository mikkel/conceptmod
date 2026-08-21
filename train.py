"""conceptmod 2.0 CLI.

Examples:
    python train.py --phrase "#:0.4|human=robot:0.8|robot%human:-0.1" \
        --verify-prompt "a human walking in a city" \
        --verify-prompt "a portrait of a human" \
        --verify-prompt "a bowl of fruit on a table" \
        --out outputs/composite

    python train.py --phrase "cat~dog" --backend zimage --lora 8 ...
    python train.py --phrase "#:0.4|human=robot:0.8|robot%human:-0.1" \
        --backend anima --out outputs/anima_composite
    python train.py --phrase "#:0.4|human=robot:0.8|robot%human:-0.1" \
        --backend krea --out outputs/krea_composite
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import torch

from conceptmod import ops
from conceptmod.backends import BACKENDS, load_backend
from conceptmod.model_train import train_model
from conceptmod.verify import before_after_grid


def main():
    p = argparse.ArgumentParser(description="finetuning with words")
    p.add_argument("--phrase", required=True, help="conceptmod DSL phrase")
    p.add_argument("--backend", default="sana", choices=list(BACKENDS))
    p.add_argument("--model-id", default=None,
                   help="override the backend's default hub id")
    p.add_argument("--resolution", type=int, default=None,
                   help="square training / verify resolution")
    p.add_argument("--device", default="cuda:0")
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
    p.add_argument("--accumulation-steps", type=int, default=None,
                   help="micro-steps averaged per optimizer step. Not a speed "
                        "knob: one draw per update lets AdamW's scale "
                        "invariance turn the quiet end of the schedule into a "
                        "full-size random walk. Defaults to the backend's "
                        "training_defaults (1 for the diffusers backends, "
                        "8 for SenseNova).")
    p.add_argument("--train-mlp", action="store_true",
                   help="SenseNova only: also adapt the generation branch's "
                        "MLP (gate/up/down), not just q/k/v/o.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-prompt", default=None,
                   help="save a progress image of this prompt periodically")
    p.add_argument("--verify-prompt", action="append", default=[],
                   help="prompt for the before/after grid (repeatable)")
    p.add_argument("--verify-seeds", type=int, nargs="+", default=[42, 1234])
    p.add_argument("--exaggerate-guidance", type=float, default=None)
    p.add_argument("--erase-guidance", type=float, default=None)
    p.add_argument("--write-guidance", type=float, default=None)
    p.add_argument("--write-context", default=None, choices=["target", "source"],
                   help="trajectory the '=' op draws z_ctx from: 'source' = the "
                        "prompt being trained (correct for pixel-space "
                        "backends), 'target' = the concept being written. "
                        "Defaults to the backend's training_defaults.")
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--sample-guidance", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=0,
                   help="optional write/erase probe eval period (0 = off; not a universal stop)")
    p.add_argument("--swap-stop", type=float, default=0.35)
    p.add_argument("--hold-max", type=float, default=0.25)
    p.add_argument("--min-steps", type=int, default=75)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    backend_kwargs = {}
    if args.train_mlp:
        # only the backends that understand it should ever see it
        backend_kwargs["train_mlp"] = True
    backend = load_backend(
        args.backend, device=args.device, lora_rank=args.lora,
        model_id=args.model_id, resolution=args.resolution, **backend_kwargs,
    )
    print(f"backend loaded in {time.time() - t0:.0f}s")

    cfg = ops.OpDefaults()
    for name, value in getattr(backend, "training_defaults", lambda: {})().items():
        setattr(cfg, name, value)
    for name in ("exaggerate_guidance", "erase_guidance", "write_guidance",
                 "write_context", "sample_steps", "sample_guidance",
                 "accumulation_steps"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    # Re-record the run with the *resolved* defaults. Most of what decides
    # whether a round works (sample_steps, write_context, accumulation_steps)
    # comes from the backend and is None on the command line, so a run.json of
    # raw args reads as "everything default" and forces the next reader to
    # guess what actually ran.
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({**vars(args), "resolved": dataclasses.asdict(cfg)},
                  f, indent=2)
    print("op defaults:", dataclasses.asdict(cfg))

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
        from conceptmod.verify import is_control_prompt

        sample_prompt = args.sample_prompt
        if sample_prompt is None:
            sample_prompt = next(
                (p for p in args.verify_prompt if not is_control_prompt(p)), None)
        if sample_prompt == "":
            sample_prompt = None
        if sample_prompt:
            print(f"progress shots: {sample_prompt!r} -> {args.out}/progress")
        train_model(
            backend,
            args.phrase,
            iterations=args.iterations,
            lr=args.lr,
            train_method=args.train_method,
            accumulation_steps=cfg.accumulation_steps,
            seed=args.seed,
            op_defaults=cfg,
            sample_prompt=sample_prompt,
            sample_dir=os.path.join(args.out, "progress") if sample_prompt else None,
            eval_every=args.eval_every,
            swap_stop=args.swap_stop,
            hold_max=args.hold_max,
            min_steps=args.min_steps,
            metrics_path=os.path.join(args.out, "metrics.jsonl"),
        )
        verify("grid.png")

    if args.save_model:
        ext = "" if args.lora else "transformer.safetensors"
        backend.save_trained(os.path.join(args.out, ext or "lora"))
        print("saved trained weights")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
