"""One-step sanity check of every DSL op: loss computes, grads flow, no NaN."""
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend

t0 = time.time()
backend = load_backend("sana", device="cuda:0")
print(f"loaded in {time.time() - t0:.0f}s")
params = backend.trainable_parameters("xattn")
print("trainable:", sum(p.numel() for p in params) / 1e6, "M")

cfg = ops.OpDefaults()
phrases = {
    "exaggerate": "vibrant colors++",
    "erase": "monochrome--",
    "write_uncond": "=snow",
    "write": "human=robot",
    "freeze": "#",
    "orthogonal": "cat%dog",
    "blend": "anime%hyperrealistic:-1.0",
    "pixel": "photo^painting",
}

for name, phrase in phrases.items():
    torch.manual_seed(0)
    rules = dsl.parse_phrase(phrase)
    ctx = ops.StepContext(backend, stop_index=5, seed=123, cfg=cfg)
    t0 = time.time()
    loss = sum(r.alpha * ops.rule_loss(r, ctx) for r in rules)
    loss.backward()
    gnorm = sum((p.grad ** 2).sum().item() for p in params if p.grad is not None) ** 0.5
    print(f"{name:14s} loss={loss.item():+.5f} gradnorm={gnorm:.2e} "
          f"nan={torch.isnan(loss).item()} t={time.time() - t0:.1f}s")
    for p in params:
        p.grad = None

print("max vram GiB:", torch.cuda.max_memory_allocated("cuda:0") / 2**30)
