"""Stage 1: train a LoRA on the *text encoder only*, in embedding space.

This is the modernized "notrigger" method from ntc-ai/sliders-conceptmod:
no diffusion sampling in the loop, so it is orders of magnitude faster than
model training and can be verified with images before stage 2 ever runs.

The mechanism differs from the CLIP-era original because Sana/Z-Image use
LLM encoders with variable-length valid tokens: instead of matching full
fixed-length CLIP sequences, we steer with *pooled concept directions*.
For a concept c, its direction is

    d_c = pool(E_f(c)) - pool(E_f(''))

(pool = attention-masked mean over valid token positions of the frozen
encoder). Ops then move every valid token embedding of random *probe
prompts*:

    c++   E_t(p) -> E_f(p) + s * d_c          (concept added everywhere)
    c--   E_t(p) -> E_f(p) - s * d_c          (concept pushed out everywhere)
    a=b   E_t(p_a) -> E_f(p_a) + (pool(E_f(p_b)) - pool(E_f(p_a)))
          where p_a / p_b are the same probe template filled with a / b
          ('=b' with empty a behaves like b++)
    a#b   E_t(template(a)) pinned to E_f(template(b)); bare '#' pins the
          neutral probe templates themselves (drift anchor)
    a%b   decorrelate pooled trained-b direction from frozen-a direction
          (negative alpha aligns instead - blend)

Kept from notrigger: the fixed-distance curriculum (move a fixed fraction
of the initial distance per step instead of jumping to the target, which
kept training from collapsing) and gradient clipping.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from conceptmod import dsl

PROBE_TEMPLATES = [
    "a photo of {}",
    "a painting of {}",
    "{}",
    "a detailed picture of {}, high quality",
    "{} in a landscape",
    "a portrait of {}",
]

NEUTRAL_PROBES = [
    "a photo of a person",
    "a city street",
    "a landscape with trees",
    "an animal",
    "a room interior",
    "a portrait of a woman",
    "a bowl of fruit on a table",
]


def _token_features(embeds: torch.Tensor) -> torch.Tensor:
    """(B, L, D) or (B, L, layers, D) -> (B, L, D'). Krea stacks layer taps."""
    e = embeds.float()
    if e.ndim == 4:
        return e.flatten(-2)
    return e


def pool(text_embeds) -> torch.Tensor:
    """Attention-masked mean over valid token positions -> (B, D).
    Handles both padded (tensor + mask) and unpadded (list of tensors)
    embedding formats."""
    e, m = text_embeds.embeds, text_embeds.mask
    if isinstance(e, list):
        return torch.stack([_token_features(t.unsqueeze(0)).squeeze(0).mean(dim=0)
                            for t in e])
    e = _token_features(e)
    if m is None:
        return e.mean(dim=1)
    m = m.to(e.dtype).unsqueeze(-1)
    return (e * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def as_padded(text_embeds) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize either embedding format to (B, L, D) + boolean mask."""
    e, m = text_embeds.embeds, text_embeds.mask
    if isinstance(e, list):
        assert len(e) == 1, "encoder stage expects batch size 1"
        t = _token_features(e[0].unsqueeze(0))
        return t, torch.ones(t.shape[:2], device=t.device, dtype=torch.bool)
    e = _token_features(e)
    if m is None:
        m = torch.ones(e.shape[:2], device=e.device, dtype=torch.bool)
    return e, m


def masked_mse(a, b, mask) -> torch.Tensor:
    m = mask.to(a.dtype).unsqueeze(-1)
    return (((a.float() - b.float()) ** 2) * m).sum() / (m.sum() * a.shape[-1])


def fixed_distance_target(current, target, step_dist):
    """Notrigger's curriculum: a target at most ``step_dist`` away from
    ``current`` along the direction to ``target``."""
    diff = target - current
    dist = diff.norm(dim=-1, keepdim=True)
    direction = diff / (dist + 1e-8)
    move = torch.minimum(dist, torch.full_like(dist, float(step_dist)))
    return current + direction * move


def train_encoder(
    backend,
    phrase: str,
    iterations: int = 400,
    lr: float = 1e-4,
    rank: int = 8,
    seed: int = 0,
    strength: float = 1.0,
    curriculum_split: int = 20,
    log_every: int = 50,
):
    rules = dsl.parse_phrase(phrase)
    usable, skipped = [], []
    for r in rules:
        if r.needs_random_prompt:
            skipped.append(r)  # random-prompt rules are a model-stage feature
        elif r.op in (dsl.EXAGGERATE, dsl.ERASE, dsl.WRITE, dsl.FREEZE,
                      dsl.ORTHOGONAL):
            usable.append(r)
        else:
            skipped.append(r)
    for r in skipped:
        print(f"encoder stage: skipping rule {r.raw!r} (op {r.op} is model-stage only)")
    if not usable:
        print("encoder stage: nothing to train")
        return []

    params = backend.attach_encoder_lora(rank)
    n = sum(p.numel() for p in params)
    print(f"encoder LoRA rank {rank}: {n/1e6:.1f}M params, phrase {phrase!r}")
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)

    # ---- precompute frozen embeddings and pooled concept directions ----
    with torch.no_grad():
        pool_uncond = pool(backend.encode_text("", frozen=True))
        directions = {}
        for r in usable:
            for c in {r.a, r.b} - {""}:
                if c not in directions:
                    directions[c] = pool(backend.encode_text(c, frozen=True)) - pool_uncond

    # per-rule curriculum step size: |direction| / split
    step_dist = {}
    for r in usable:
        c = r.b or r.a
        if c in directions:
            step_dist[r.raw] = directions[c].norm().item() / curriculum_split

    history = []
    pbar = tqdm(range(iterations))
    for i in pbar:
        losses = {}
        total = None
        for rule in usable:
            template = rng.choice(PROBE_TEMPLATES)
            neutral = rng.choice(NEUTRAL_PROBES)

            if rule.op in (dsl.EXAGGERATE, dsl.ERASE) or (
                    rule.op == dsl.WRITE and rule.a == ""):
                concept = rule.b if rule.op == dsl.WRITE else rule.a
                sign = -1.0 if rule.op == dsl.ERASE else 1.0
                probe = neutral
                cur, mask = as_padded(backend.encode_text_grad(probe))
                fro, _ = as_padded(backend.encode_text(probe, frozen=True))
                target = fro.float() + sign * strength * directions[concept].unsqueeze(1)
                capped = fixed_distance_target(cur.float().detach(), target,
                                               step_dist[rule.raw])
                loss = masked_mse(cur, capped, mask)

            elif rule.op == dsl.WRITE:
                p_a = template.format(rule.a)
                p_b = template.format(rule.b)
                cur, mask = as_padded(backend.encode_text_grad(p_a))
                frozen_a = backend.encode_text(p_a, frozen=True)
                fro, _ = as_padded(frozen_a)
                d = (pool(backend.encode_text(p_b, frozen=True))
                     - pool(frozen_a)).unsqueeze(1)
                target = fro.float() + strength * d
                capped = fixed_distance_target(cur.float().detach(), target,
                                               step_dist[rule.raw])
                loss = masked_mse(cur, capped, mask)

            elif rule.op == dsl.FREEZE:
                p_a = template.format(rule.a) if rule.a else neutral
                p_b = template.format(rule.b) if rule.b else neutral
                cur, mask = as_padded(backend.encode_text_grad(p_a))
                fro, _ = as_padded(backend.encode_text(p_b, frozen=True))
                L = min(cur.shape[1], fro.shape[1])  # unpadded formats may differ
                loss = masked_mse(cur[:, :L], fro[:, :L], mask[:, :L])

            elif rule.op == dsl.ORTHOGONAL:
                d_a = directions[rule.a]
                d_b = (pool(backend.encode_text_grad(rule.b)) - pool_uncond)
                cos = F.cosine_similarity(d_a, d_b, dim=-1, eps=1e-6)
                loss = cos.abs().mean()

            losses[rule.raw] = rule.alpha * loss
            total = losses[rule.raw] if total is None else total + losses[rule.raw]

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        history.append(total.item())
        pbar.set_postfix({"loss": f"{history[-1]:.5f}"})
        if log_every and i % log_every == 0:
            parts = " ".join(f"[{k}]={v.item():.5f}" for k, v in losses.items())
            tqdm.write(f"step {i}: {parts}")

    backend._text_cache.clear()
    return history
