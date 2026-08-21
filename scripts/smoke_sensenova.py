"""Smoke test the SenseNova-U1.5-8B-MoT backend: baseline images + one op step.

Modest footprint: defaults to 512px (256 image tokens) so the peak sits near
the ~33 GiB weight floor. Bump --resolution for a quality check.

    CONCEPTMOD_DEVICE=cuda:1 .venv/bin/python scripts/smoke_sensenova.py

After a training run, point it at the adapter to check the edit actually held
the control prompt instead of globally relighting the model -- and at the
*training* schedule, where a schedule-local bias is visible:

    CONCEPTMOD_DEVICE=cuda:1 .venv/bin/python scripts/smoke_sensenova.py \\
        --adapter outputs/50_sensenova_composite/lora --skip-train
"""
import argparse
import os
import time

import torch

from conceptmod import dsl, ops
from conceptmod.backends import load_backend

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=512)
ap.add_argument("--steps", type=int, default=None, help="override generate steps")
ap.add_argument("--out", default="outputs/50_sensenova_smoke")
ap.add_argument("--lora", type=int, default=16)
ap.add_argument("--skip-train", action="store_true")
ap.add_argument("--adapter", default=None, metavar="DIR",
                help="load a trained adapter and run the control-preservation "
                     "check (e.g. outputs/50_sensenova_composite/lora)")
ap.add_argument("--control-prompt", default="a bowl of fruit on a table",
                help="prompt the edit must leave alone")
ap.add_argument("--control-tol", type=float, default=0.12,
                help="max relative mean-luminance shift on the control")
args = ap.parse_args()

DEVICE = os.environ.get("CONCEPTMOD_DEVICE", "cuda:0")

t0 = time.time()
backend = load_backend("sensenova", device=DEVICE, lora_rank=args.lora,
                       resolution=args.resolution)
if args.steps:
    backend.generate_steps = args.steps
print(f"loaded in {time.time() - t0:.0f}s; latent shape {backend.latent_shape}")
print(f"tokens {backend.image_seq_len} noise_scale {backend.noise_scale:.3f} "
      f"shift {backend.timestep_shift} steps {backend.generate_steps} "
      f"cfg {backend.generate_guidance}")
print("load vram GiB:", torch.cuda.max_memory_allocated(DEVICE) / 2**30)

os.makedirs(args.out, exist_ok=True)
prompts = [
    "a human walking in a city",
    "a cat sitting on a windowsill",
    "a bowl of fruit on a table",
]
for i, p in enumerate(prompts):
    t1 = time.time()
    img = backend.generate(p, seed=42 + i)
    slug = p.replace(" ", "_")[:36]
    path = f"{args.out}/{i}_{slug}.png"
    img.save(path)
    ext = img.convert("L").getextrema()
    print(f"saved {path} {img.size} range={ext} {time.time() - t1:.1f}s")
    assert ext[1] > ext[0], f"{path} is a flat image"

cfg = ops.OpDefaults(**backend.training_defaults())

# --- the loss must be commensurate across the training schedule.
# _t2i_predict_v returns (x_pred - z) / (1 - t), so raw |v| grows several-fold
# toward t=1 and a uniform-in-t MSE would weigh those few samples ~(1-t)^-2
# heavier than the rest -- which is how the first composite run learned a
# global relight (control luminance halved) instead of human->robot.
ts = backend._timesteps(cfg.sample_steps)
gen = torch.Generator(device=DEVICE).manual_seed(0)
z = backend._noise(gen)
raw, scaled = [], []
for t in (ts[1], ts[cfg.sample_steps - 1]):
    v = backend.predict_v(prompts[0], z, t, frozen=True)
    w = backend.velocity_loss_scale(t)
    raw.append(float(v.std()))
    scaled.append(float((v * w).std()))
print(f"velocity std across the schedule: raw {raw[0]:.3f} -> {raw[1]:.3f} "
      f"(x{raw[1]/raw[0]:.2f}), scaled {scaled[0]:.3f} -> {scaled[1]:.3f} "
      f"(x{scaled[1]/scaled[0]:.2f})")
assert 0.6 < scaled[1] / scaled[0] < 1.6, (
    "velocity_loss_scale does not flatten the schedule; an unweighted op MSE "
    "will be dominated by the high-t samples", scaled)

# --- ...but the *loss* is still not flat, and that is not a bug to weight
# away. velocity_loss_scale only removes the 1/(1-t) in the parameterisation
# of v; what is left is how much a prompt can still change the prediction at
# that t, and near the image end it can barely change it at all. Measured
# here so the next reader sees the real profile rather than "flat 1e-4 noise":
# the '=' loss ran 0.196 / 0.0124 / 0.0026 at indices 0/1/2 of 16 and 5.3e-5
# at index 15, so 96% of it sits in the first three points. One uniform draw
# per optimizer step therefore spends ~13 of every 16 updates on nothing --
# and AdamW, being scale invariant, still walks a full lr along that nothing.
_write = dsl.parse_phrase("human=robot")[0]
profile = []
for stop in (0, 1, cfg.sample_steps // 2, cfg.sample_steps - 1):
    c = ops.StepContext(backend, stop_index=stop, seed=7, cfg=cfg)
    with torch.no_grad():
        loss = float(ops.rule_loss(_write, c))
        # whatever templated context the WRITE branch sampled, not a second one
        profile.append((stop, float(next(iter(c._z.values()))[1]), loss))
    del c
print("'=' loss across the training schedule:")
for stop, t, loss in profile:
    print(f"  idx {stop:2d}  t={t:.4f}  loss={loss:.6f}")
span = profile[0][2] / max(profile[-1][2], 1e-12)
print(f"  noise-end / image-end ratio: {span:.0f}x  "
      f"(accumulation_steps={cfg.accumulation_steps})")
assert span > 20, (
    "expected the write signal to be concentrated at the noise end of the "
    "schedule on this backend; if it is not, re-derive accumulation_steps",
    profile)
assert cfg.sample_steps // cfg.accumulation_steps <= 2, (
    "accumulation_steps is too small for this schedule: model_train strata "
    "are sample_steps/accumulation_steps wide, and the first stratum must "
    "land inside the two or three indices that carry the write signal, or "
    "an optimizer step can average nothing but noise",
    cfg.sample_steps, cfg.accumulation_steps)

if args.adapter:
    # Control preservation, checked at the schedule the *trainer* used.
    # before_after_grid renders at generate_steps (50), where a schedule-local
    # bias shows up as merely dim; at sample_steps the same weights rendered
    # near-black and the failure was obvious.
    import numpy as np
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    res = set_peft_model_state_dict(backend.transformer, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    assert not unexpected, f"adapter did not load cleanly: {unexpected[:5]}"
    print(f"loaded {len(sd)} adapter tensors from {args.adapter}")

    def _mean_l(prompt, steps, frozen):
        img = backend.generate(prompt, seed=42, num_steps=steps, frozen=frozen)
        return float(np.asarray(img.convert("L"), dtype=np.float32).mean())

    failed = []
    for steps in (cfg.sample_steps, backend.generate_steps):
        before = _mean_l(args.control_prompt, steps, True)
        after = _mean_l(args.control_prompt, steps, False)
        shift = abs(after - before) / max(before, 1e-6)
        print(f"control @{steps} steps: mean L {before:.1f} -> {after:.1f} "
              f"({shift:+.1%})")
        if shift > args.control_tol:
            failed.append((steps, shift))
    assert not failed, (
        f"{args.control_prompt!r} was not held: {failed} exceeds "
        f"{args.control_tol:.0%}. A global luminance shift on a control "
        f"prompt is a DC drift, not a concept edit.")

if args.skip_train:
    raise SystemExit(0)

params = backend.trainable_parameters("lora")
print("lora params:", sum(p.numel() for p in params) / 1e6, "M")

rules = dsl.parse_phrase("vibrant colors++")
ctx = ops.StepContext(backend, stop_index=4, seed=7, cfg=cfg)
t1 = time.time()
loss = sum(r.alpha * ops.rule_loss(r, ctx) for r in rules)
loss.backward()
grads = [p for p in params if p.grad is not None]
g = sum((p.grad ** 2).sum().item() for p in grads) ** 0.5
print(f"op step: loss={loss.item():.5f} gradnorm={g:.3e} "
      f"params_with_grad={len(grads)}/{len(params)} t={time.time()-t1:.1f}s")
assert g > 0, "no gradient reached the LoRA params"
# The flash-prealloc KV path would silently drop grads to k_proj_mot_gen /
# v_proj_mot_gen; make sure the torch.cat fallback kept the graph. Check
# lora_B, not lora_A: lora_B is zero-initialised, so dL/dA is exactly zero on
# the very first step for every module and would prove nothing.
by_kind = {}
for n, p in backend.transformer.named_parameters():
    if p.requires_grad and "lora_B" in n:
        kind = n.split(".lora_B")[0].split(".")[-1]
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += 1
        by_kind[kind][1] += int(p.grad is not None and p.grad.abs().sum() > 0)
for kind, (tot, live) in sorted(by_kind.items()):
    print(f"  {kind}: {live}/{tot} with nonzero grad")
    assert live == tot, f"{kind} lost its gradient path"
print("max vram GiB:", torch.cuda.max_memory_allocated(DEVICE) / 2**30)
