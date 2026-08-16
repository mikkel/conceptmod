"""Smoke test Z-Image Turbo backend: baseline generation + one op step."""
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend

t0 = time.time()
backend = load_backend("zimage", device="cuda:1", lora_rank=16)
print(f"loaded in {time.time() - t0:.0f}s; latent shape {backend.latent_shape}")
print("scheduler:", type(backend.pipe.scheduler).__name__)

import os
os.makedirs("outputs/20_zimage_baseline", exist_ok=True)
for i, p in enumerate(["a cat sitting on a windowsill",
                       "a portrait of a woman, photo"]):
    t1 = time.time()
    img = backend.generate(p, seed=42 + i)
    img.save(f"outputs/20_zimage_baseline/{i}_{p.replace(' ', '_')[:36]}.png")
    print("saved", p, f"{time.time() - t1:.1f}s")

params = backend.trainable_parameters("lora")
print("lora params:", sum(p.numel() for p in params) / 1e6, "M")

cfg = ops.OpDefaults(sample_steps=8, sample_guidance=0.0)
rules = dsl.parse_phrase("vibrant colors++")
ctx = ops.StepContext(backend, stop_index=4, seed=7, cfg=cfg)
t1 = time.time()
loss = sum(r.alpha * ops.rule_loss(r, ctx) for r in rules)
loss.backward()
g = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
print(f"op step: loss={loss.item():.5f} gradnorm={g:.2e} t={time.time()-t1:.1f}s")
print("max vram GiB:", torch.cuda.max_memory_allocated("cuda:1") / 2**30)
