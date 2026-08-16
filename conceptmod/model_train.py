"""Stage 2: finetune the diffusion transformer with DSL losses."""

from __future__ import annotations

import random

import torch
from tqdm import tqdm

from conceptmod import dsl, ops


def load_random_prompts():
    from datasets import load_dataset

    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    return ds["Prompt"]


def train_model(
    backend,
    phrase: str,
    iterations: int = 500,
    lr: float = 1e-5,
    train_method: str = "xattn",
    accumulation_steps: int = 1,
    seed: int = 0,
    op_defaults: ops.OpDefaults | None = None,
    sample_prompt: str | None = None,
    sample_every: int = 100,
    sample_dir: str | None = None,
    log_every: int = 25,
):
    cfg = op_defaults or ops.OpDefaults()
    rules = dsl.parse_phrase(phrase)
    for r in rules:
        if r.op == dsl.REWARD:
            raise NotImplementedError(
                "the ';' (ImageReward) op is not implemented in conceptmod 2.x")

    random_prompts = None
    if any(r.needs_random_prompt or r.op == dsl.EXAGGERATE for r in rules):
        random_prompts = load_random_prompts()
    has_exaggerate = any(r.op == dsl.EXAGGERATE for r in rules)

    params = backend.trainable_parameters(train_method)
    n_params = sum(p.numel() for p in params)
    print(f"training {len(params)} tensors / {n_params/1e6:.1f}M params "
          f"({train_method}), phrase: {phrase!r}")
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)

    history = []
    pbar = tqdm(range(iterations))
    for i in pbar:
        step_rules = dsl.materialize(
            rules, rng.choice(random_prompts) if random_prompts else None)
        stop_index = rng.randrange(0, cfg.sample_steps)
        step_seed = rng.randrange(2**31)
        probe = None
        if has_exaggerate and rng.random() < cfg.probe_p:
            probe = dsl.sanitize_prompt(rng.choice(random_prompts))[:200]
        ctx = ops.StepContext(backend, stop_index, step_seed, cfg, probe=probe)

        losses = {}
        total = None
        for rule in step_rules:
            loss = rule.alpha * ops.rule_loss(rule, ctx)
            losses[rule.raw] = loss.item()
            total = loss if total is None else total + loss

        (total / accumulation_steps).backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        history.append(sum(losses.values()))
        pbar.set_postfix({"loss": f"{history[-1]:.4f}"})
        if log_every and i % log_every == 0:
            parts = " ".join(f"[{k}]={v:.4f}" for k, v in losses.items())
            tqdm.write(f"step {i}: {parts}")

        if sample_prompt and sample_dir and (i + 1) % sample_every == 0:
            import os

            os.makedirs(sample_dir, exist_ok=True)
            img = backend.generate(sample_prompt, seed=42)
            img.save(f"{sample_dir}/{i + 1:05d}.png")

    backend.transformer.eval()
    return history
