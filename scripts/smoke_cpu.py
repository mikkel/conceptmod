"""CPU sample problem from the CLI (no GPU, no Hub weights).

Runs the same ``red=blue`` write as ``tests/test_cpu_sample.py`` through
``train_model`` → dsl/ops, then prints the alignment cosine and wall time.

    python scripts/smoke_cpu.py

Equivalent train.py invocation:

    python train.py --backend cpu --device cpu --stage model --phrase red=blue \
        --lora 4 --iterations 80 --lr 1e-2 --out outputs/cpu_sample
"""
from __future__ import annotations

import argparse

from conceptmod.backends.cpu import (
    COSINE_THRESHOLD,
    SAMPLE_LR,
    SAMPLE_PHRASE,
    SAMPLE_STEPS,
    run_sample_problem,
)


def main():
    p = argparse.ArgumentParser(description="CPU sample problem (red=blue)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--phrase", default=SAMPLE_PHRASE)
    p.add_argument("--iterations", type=int, default=SAMPLE_STEPS)
    p.add_argument("--lr", type=float, default=SAMPLE_LR)
    p.add_argument("--lora", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    result = run_sample_problem(
        device=args.device,
        phrase=args.phrase,
        iterations=args.iterations,
        lr=args.lr,
        lora_rank=args.lora,
        seed=args.seed,
    )
    print(f"phrase: {result.phrase}")
    print(f"cosine before: {result.cosine_before:.4f}")
    print(f"cosine after:  {result.cosine_after:.4f}  (threshold {COSINE_THRESHOLD})")
    print(f"lora B norm:   {result.lora_delta_norm:.4f}")
    print(f"wall time:     {result.elapsed_s:.2f}s")
    if result.cosine_after <= COSINE_THRESHOLD:
        raise SystemExit(
            f"sample failed: cosine {result.cosine_after:.4f} "
            f"did not pass {COSINE_THRESHOLD}"
        )
    print("ok")


if __name__ == "__main__":
    main()
