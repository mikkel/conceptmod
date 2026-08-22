"""SMOKE (not CI): Flux.2 Klein GPU path — baseline generate + one op step.

Requires CUDA and a download of
``black-forest-labs/FLUX.2-klein-base-4B`` (default) or
``KLEIN_MODEL=9b-base`` for the gated 9B Base. This is not a substitute
for a full train — it only checks that the backend loads, a LoRA
attaches, and one DSL step produces a gradient.

    CONCEPTMOD_DEVICE=cuda:0 python scripts/smoke_klein.py
    KLEIN_MODEL=4b CONCEPTMOD_DEVICE=cuda:0 python scripts/smoke_klein.py

Skip this on CPU / small VMs. Mapper + convert coverage lives in
``tests/test_convert_klein.py`` (CPU, no weights).
"""
import os
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend

DEVICE = os.environ.get("CONCEPTMOD_DEVICE", "cuda:0")
MODEL = os.environ.get("KLEIN_MODEL", "black-forest-labs/FLUX.2-klein-base-4B")

if not torch.cuda.is_available():
    raise SystemExit(
        "smoke_klein.py is a labeled GPU smoke path; CUDA is required. "
        "Run the convert/mapper unit tests instead."
    )

t0 = time.time()
backend = load_backend("klein", device=DEVICE, lora_rank=16, model_id=MODEL,
                       resolution=512)
print(f"loaded in {time.time() - t0:.0f}s; latent shape {backend.latent_shape}")
print("scheduler:", type(backend.pipe.scheduler).__name__,
      "distilled:", backend.is_distilled,
      "steps", backend.generate_steps, "cfg", backend.generate_guidance)

os.makedirs("outputs/60_klein_baseline", exist_ok=True)
prompts = [
    "a cat sitting on a windowsill",
    "a bowl of fruit on a table",
]
for i, p in enumerate(prompts):
    t1 = time.time()
    img = backend.generate(p, seed=42 + i)
    slug = p.replace(" ", "_")[:36]
    img.save(f"outputs/60_klein_baseline/{i}_{slug}.png")
    print("saved", p, f"{time.time() - t1:.1f}s")

params = backend.trainable_parameters("lora")
print("lora params:", sum(p.numel() for p in params) / 1e6, "M")

cfg = ops.OpDefaults(**backend.training_defaults())
rules = dsl.parse_phrase("vibrant colors++")
ctx = ops.StepContext(backend, stop_index=4, seed=7, cfg=cfg)
t1 = time.time()
loss = sum(r.alpha * ops.rule_loss(r, ctx) for r in rules)
loss.backward()
g = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
print(f"op step: loss={loss.item():.5f} gradnorm={g:.2e} t={time.time()-t1:.1f}s")
print("max vram GiB:", torch.cuda.max_memory_allocated(DEVICE) / 2**30)
