"""CPU 2-D analysis suite: pictures + numbers for DSL / erase geometry.

    python scripts/analyze_2d.py
    python scripts/analyze_2d.py --out outputs/2d_analysis --steps 40

No GPU, no Hub. Writes quiver / trajectory / table PNGs and metrics.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptmod.analysis_2d import (
    DEFAULT_LR,
    DEFAULT_OUT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    METHOD_TITLES,
    run_suite,
    write_artifacts,
)
from conceptmod.analysis_dsl import JOB_TITLES, run_jobs, write_job_artifacts


def main():
    p = argparse.ArgumentParser(description="2-D CPU geometry suite")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--jobs", action="store_true",
        help="score phrase-DSL jobs (recipes vs missing ops) instead of the erase/GEM suite",
    )
    args = p.parse_args()

    if args.jobs:
        titles = JOB_TITLES
        results = run_jobs(steps=args.steps, lr=args.lr, seed=args.seed)
        paths = write_job_artifacts(results, out=args.out)
        label = "DSL jobs"
    else:
        titles = METHOD_TITLES
        results = run_suite(steps=args.steps, lr=args.lr, seed=args.seed)
        paths = write_artifacts(results, out=args.out)
        label = "erase / GEM"
    total = sum(r.elapsed_s for r in results)
    print(f"fixture: red/blue (color) vs stripe/dot (pattern), shared LoRA")
    print(f"suite: {label}  wall time: {total:.2f}s  steps={args.steps}")
    print()
    for r in results:
        print(
            f"  {r.verdict:11s}  {titles.get(r.name, r.name):42s}  "
            f"red={r.after.color_on_red:+.3f}  "
            f"stripe_hold={r.after.stripe_hold:+.3f}  "
            f"leak={r.after.pattern_on_red:+.3f}  "
            f"write_cos={r.after.write_cosine:+.3f}"
        )
        print(f"             {r.note}")
    print()
    for key, path in paths.items():
        print(f"wrote {key}: {path}")


if __name__ == "__main__":
    main()
