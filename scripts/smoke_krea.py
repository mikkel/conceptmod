"""Smoke test Krea 2: baseline generation + one op step.

Default checkpoint is the local Kitchen NVFP4 turbo file when present,
otherwise the hub Raw model. SFW prompts only.
"""
import os
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend
from conceptmod.backends.krea_weights import resolve_local_transformer

DEVICE = os.environ.get("CONCEPTMOD_DEVICE", "cuda:0")
LOCAL = os.environ.get("KREA_MODEL", "models/kreaturboft_nvfp4.safetensors")
MODEL = LOCAL if resolve_local_transformer(LOCAL) else "krea/Krea-2-Raw"

t0 = time.time()
backend = load_backend("krea", device=DEVICE, lora_rank=16, model_id=MODEL,
                       resolution=512)
print(f"loaded in {time.time() - t0:.0f}s; latent shape {backend.latent_shape}")
print("scheduler:", type(backend.pipe.scheduler).__name__,
      "distilled:", backend.is_distilled,
      "steps", backend.generate_steps, "cfg", backend.generate_guidance)

os.makedirs("outputs/42_krea_turbo_baseline", exist_ok=True)
prompts = [
    "a cat sitting on a windowsill",
    "a bowl of fruit on a table",
]
for i, p in enumerate(prompts):
    t1 = time.time()
    img = backend.generate(p, seed=42 + i)
    slug = p.replace(" ", "_")[:36]
    img.save(f"outputs/42_krea_turbo_baseline/{i}_{slug}.png")
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
