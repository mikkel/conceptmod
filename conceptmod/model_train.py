"""Stage 2: finetune the diffusion transformer with DSL losses."""

from __future__ import annotations

import random

import torch
from tqdm import tqdm

from conceptmod import dsl, metrics, ops


def load_random_prompts():
    from datasets import load_dataset

    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
    return ds["Prompt"]


def stop_index_for(rng, micro: int, accumulation_steps: int,
                   sample_steps: int) -> int:
    """Which schedule point this micro-step trains on.

    A single optimizer step must see the whole schedule, not one uniform draw
    from it, because the DSL losses are wildly unequal across ``t``: on
    SenseNova-U1.5 the ``=`` loss is 0.196 at the noise end of the 16-point
    training schedule and 5e-5 at the image end (3700x), with 96% of the total
    in the first three indices. An i.i.d. draw per *optimizer* step therefore
    spends most updates on near-zero signal -- and AdamW, being scale
    invariant, still moves the parameters a full ``lr`` in whatever direction
    that noise points. Averaging the gradient over an accumulation window
    first restores the loss magnitude as the importance weight it is.

    i.i.d. sampling would not do that reliably: an 8-draw window has a
    ``(13/16)**8`` = 17% chance of containing none of the three indices that
    carry the signal. So each micro-step of a window takes its own stratum of
    the schedule, which makes every optimizer step see both ends of it. With
    ``accumulation_steps == 1`` there is nothing to stratify and the draw stays
    uniform, so the backends whose proofs already landed are unaffected.
    """
    if accumulation_steps <= 1:
        return rng.randrange(0, sample_steps)
    j = micro % accumulation_steps
    if accumulation_steps >= sample_steps:
        # more micro-steps than schedule points: walk them in order so every
        # point is visited at least once per window.
        return j % sample_steps
    lo = (j * sample_steps) // accumulation_steps
    hi = ((j + 1) * sample_steps) // accumulation_steps
    return rng.randrange(lo, max(hi, lo + 1))


def train_model(
    backend,
    phrase: str,
    iterations: int = 500,
    lr: float = 1e-5,
    train_method: str = "xattn",
    accumulation_steps: int | None = None,
    seed: int = 0,
    op_defaults: ops.OpDefaults | None = None,
    sample_prompt: str | None = None,
    sample_every: int = 100,
    sample_dir: str | None = None,
    log_every: int = 25,
    eval_every: int = 0,
    swap_stop: float = 0.35,
    hold_max: float = 0.25,
    min_steps: int = 75,
    patience: int = 2,
    metrics_path: str | None = None,
):
    cfg = op_defaults or ops.OpDefaults()
    # an explicit argument (the CLI flag) wins over the backend's declared
    # default; see OpDefaults.accumulation_steps for why this is not a speed
    # knob.
    if accumulation_steps is None:
        accumulation_steps = cfg.accumulation_steps
    accumulation_steps = max(1, int(accumulation_steps))
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
    print(f"{iterations} micro-steps, {accumulation_steps} per optimizer step "
          f"(~{max(1, iterations // accumulation_steps)} updates), lr {lr}")

    history = []
    streak = 0
    pending = 0            # micro-steps whose gradient is not yet applied
    window: list[dict] = []  # their per-rule losses, for an honest log line
    last_window: dict = {}
    pbar = tqdm(range(iterations))
    for i in pbar:
        step_rules = dsl.materialize(
            rules, rng.choice(random_prompts) if random_prompts else None)
        stop_index = stop_index_for(rng, i, accumulation_steps, cfg.sample_steps)
        step_seed = rng.randrange(2**31)
        probe = None
        if has_exaggerate and rng.random() < cfg.probe_p:
            probe = dsl.sanitize_prompt(rng.choice(random_prompts))[:200]
        ctx = ops.StepContext(backend, stop_index, step_seed, cfg, probe=probe)

        losses = {}
        for rule in step_rules:
            loss = rule.alpha * ops.rule_loss(rule, ctx)
            losses[rule.raw] = loss.item()
            # One rule at a time so a 12B composite does not keep four
            # graphs alive. Sum of backwards == backward of the sum.
            (loss / accumulation_steps).backward()
            ctx._v = {k: t for k, t in ctx._v.items() if not k[3]}
        pending += 1
        window.append(losses)
        if pending >= accumulation_steps:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending = 0
            last_window = {k: sum(w.get(k, 0.0) for w in window) / len(window)
                           for k in window[-1]}
            window = []

        history.append(sum(losses.values()))
        postfix = {"loss": f"{history[-1]:.4f}"}
        if eval_every and i % eval_every == 0:
            report = metrics.eval_phrase(
                backend, phrase, cfg,
                swap_stop=swap_stop, hold_max=hold_max)
            postfix["swap"] = (
                f"{report.swap_ratio:.3f}" if report.swap_ratio is not None else "—")
            tqdm.write(f"step {i}: {report.summary()}")
            if metrics_path:
                import json
                import os

                os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
                with open(metrics_path, "a") as f:
                    rec = {"step": i, "swap_ratio": report.swap_ratio,
                           "hold": report.hold, "remain": report.remain,
                           "done": report.done,
                           "scores": [s.__dict__ for s in report.scores]}
                    f.write(json.dumps(rec) + "\n")
            if report.done and (i + 1) >= min_steps:
                streak += 1
                if streak >= patience:
                    tqdm.write(
                        f"early stop at step {i}: {report.summary()} "
                        f"(swap<{swap_stop}, hold<{hold_max}, patience={patience})")
                    pbar.set_postfix(postfix)
                    break
            else:
                streak = 0
        pbar.set_postfix(postfix)
        if log_every and i % log_every == 0:
            parts = " ".join(f"[{k}]={v:.4f}" for k, v in losses.items())
            # Always say which schedule point the number came from: a loss
            # that spans three orders of magnitude across t reads as "flat
            # 1e-4 noise" without it, which is how two rounds of this
            # backend's training looked healthy while learning nothing.
            head = f"step {i}: t[{stop_index}/{cfg.sample_steps}]"
            if accumulation_steps > 1 and last_window:
                mean = " ".join(f"[{k}]={v:.4f}" for k, v in last_window.items())
                tqdm.write(f"{head} {parts}  | last update mean: {mean}")
            else:
                tqdm.write(f"{head} {parts}")

        if sample_prompt and sample_dir and (i + 1) % sample_every == 0:
            import os

            os.makedirs(sample_dir, exist_ok=True)
            img = backend.generate(sample_prompt, seed=42)
            img.save(f"{sample_dir}/{i + 1:05d}.png")

    if pending:
        # a trailing partial window (early stop, or iterations not a multiple
        # of accumulation_steps) still carries real gradient -- apply it
        # rather than dropping it on the floor.
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    backend.transformer.eval()
    return history
