"""Fast plumbing check for the SenseNova backend: real code, random 2-layer model.

Loading the real 50GB checkpoint costs ~18 minutes, which makes iterating on
the backend contract painful. This builds a randomly-initialised NEOChatModel
with 2 decoder layers instead and runs the *actual* ``SenseNovaBackend``
through every method the trainer calls, in well under a minute.

It proves shapes, devices, dtypes, caching, the Euler loop and the gradient
path. It proves nothing about image quality -- the weights are noise. Use
``smoke_sensenova.py`` for that.

    CONCEPTMOD_DEVICE=cuda:1 .venv/bin/python scripts/smoke_sensenova_tiny.py
"""
import os
import tempfile

import torch
from transformers import AutoConfig, AutoModel

DEVICE = os.environ.get("CONCEPTMOD_DEVICE", "cuda:0")
RES = 128  # 4x4 = 16 image tokens


def _tiny_from_pretrained(model_id, config=None, dtype=None, device_map=None, **kw):
    from sensenova_u1 import NEOChatModel

    c = AutoConfig.from_pretrained(model_id)
    # hidden_size must stay 4096: fm_modules.vision_model_mot_gen's
    # dense_embedding output width is fixed by vision_config and its output is
    # added straight into the llm stream. Only depth and MLP width shrink.
    c.llm_config.num_hidden_layers = 2
    c.llm_config.intermediate_size = 1024
    torch.manual_seed(0)
    return NEOChatModel(c).to(dtype=dtype or torch.bfloat16).to(DEVICE)


AutoModel.from_pretrained = staticmethod(_tiny_from_pretrained)

from conceptmod import dsl, ops                              # noqa: E402
from conceptmod.backends.sensenova import SenseNovaBackend   # noqa: E402

b = SenseNovaBackend(device=DEVICE, resolution=RES, lora_rank=8)
print("latent_shape", b.latent_shape, "tokens", b.image_seq_len,
      "noise_scale", b.noise_scale)
assert b.latent_shape == (16, 3072), b.latent_shape

for attr in ("device", "latent_shape", "transformer", "_text_cache",
             "encoder_lora", "compute_dtype", "lora_rank", "resolution"):
    assert hasattr(b, attr), attr

# --- text conditioning is a KV prefix, cached per (prompt, frozen)
te = b.encode_text("a cat")
assert b.encode_text("a cat") is te, "prefix cache miss"
assert b.encode_text("") is not te, "'' must be its own prefix"
print("prefix", tuple(te.embeds.shape), "text_len", te.text_len)

# --- '' is the empty *prompt*, not the CFG negative. t2i_generate templates
# every user prompt ('' included) with SYSTEM_MESSAGE_FOR_GEN and the think
# block, and keeps a separate bare query for the unconditional. Routing ''
# to the bare query shrinks its prefix by ~240 tokens and injects that
# template difference into every v(p) - v('') the DSL forms.
from conceptmod.backends.sensenova import CFG_UNCOND                # noqa: E402

_empty_len = b.encode_text("").text_len
_neg_len = b.encode_text(CFG_UNCOND).text_len
print(f"prefix len: '' {_empty_len}  CFG_UNCOND {_neg_len}  "
      f"'a cat' {te.text_len}")
assert _neg_len < 16, ("the CFG negative must stay the official bare query",
                       _neg_len)
assert _empty_len > _neg_len + 100, (
    "'' must take the same generation template as any other prompt", _empty_len)
assert abs(_empty_len - te.text_len) < 32, (
    "'' and a real prompt must differ only by their text", _empty_len,
    te.text_len)

# --- noise carries the model's resolution-aware scale, not unit variance
g = torch.Generator(device=DEVICE).manual_seed(0)
z = b._noise(g)
assert abs(float(z.std()) - b.noise_scale) < 0.05 * b.noise_scale, float(z.std())

# --- timesteps run 0 (noise) -> 1 (image)
ts = b._timesteps(6)
assert float(ts[0]) == 0.0 and abs(float(ts[-1]) - 1.0) < 1e-6
assert bool((ts[1:] > ts[:-1]).all()), "timesteps must increase"
print("timesteps", [round(float(x), 4) for x in ts])

# --- the velocity head divides by (1 - t), so |v| is NOT flat across the
# schedule; velocity_loss_scale must report that divisor or a uniform-in-t
# MSE silently weighs the top of the schedule ~(1-t)^-2 heavier and the LoRA
# learns a global DC offset instead of a concept edit.
_raw, _scaled = [], []
for _t in (ts[1], ts[-2]):
    _v = b.predict_v("a cat", z, _t, frozen=True)
    _w = b.velocity_loss_scale(_t)
    assert abs(_w - (1.0 - float(_t))) < 1e-6, (_w, float(_t))
    _raw.append(float(_v.std()))
    _scaled.append(float((_v * _w).std()))
print(f"velocity std raw {_raw[0]:.3f} -> {_raw[1]:.3f} "
      f"(x{_raw[1]/_raw[0]:.2f}); scaled {_scaled[0]:.3f} -> {_scaled[1]:.3f} "
      f"(x{_scaled[1]/_scaled[0]:.2f})")
assert _raw[1] / _raw[0] > 2.0, "expected raw |v| to grow toward t=1"
assert 0.5 < _scaled[1] / _scaled[0] < 2.0, (
    "velocity_loss_scale did not flatten the schedule", _scaled)

# --- velocity: same shape as z, fp32 out, graph alive when not frozen
v = b.predict_v("a cat", z, ts[2], frozen=False)
assert v.shape == z.shape and v.dtype == torch.float32, (v.shape, v.dtype)
assert v.requires_grad
assert not b.predict_v("a cat", z, ts[2], frozen=True).requires_grad
print("predict_v ok", tuple(v.shape))

# --- partial_denoise must handle stop_index == 0 (pure noise)
g = torch.Generator(device=DEVICE).manual_seed(1)
z0, t0 = b.partial_denoise("a cat", 0, 6, 4.0, g)
assert z0.shape == (1, *b.latent_shape) and float(t0) == float(ts[0])
g = torch.Generator(device=DEVICE).manual_seed(1)
z3, _ = b.partial_denoise("a cat", 3, 6, 4.0, g)
assert not torch.equal(z0, z3), "stop_index had no effect"

img = b.render("a cat", torch.Generator(device=DEVICE).manual_seed(2), 4, 4.0)
assert img.shape == (1, 3, RES, RES), img.shape
assert b.generate("a cat", seed=42, num_steps=4).size == (RES, RES)
# verify.before_after_grid passes num_steps/guidance positionally as None
assert b.generate("a cat", 42, None, None, frozen=True).size == (RES, RES)
print("render/generate ok")

# --- one real op step, and the gradient path the flash-KV fast path would eat
params = b.trainable_parameters("lora")
cfg = ops.OpDefaults(**b.training_defaults())
cfg.sample_steps = 4

# '=' must draw z_ctx from the trajectory of the prompt it is training. The
# loss only moves v(a) at the latents it is evaluated on, and on this
# pixel-space backend the frozen a->b direction at b's trajectory is
# near-orthogonal to the same direction at a's (measured cos +0.36 / +0.03 /
# -0.01 at t=0.045 / 0.167 / 0.423 on the real checkpoint), so sampling in b's
# context teaches the edit where the sampler never goes.
assert cfg.write_context == "source", cfg.write_context
_w = ops.StepContext(b, stop_index=1, seed=7, cfg=cfg)
ops.rule_loss(dsl.parse_phrase("cat=dog")[0], _w)
_ctxs = list(_w._z)
assert len(_ctxs) == 1 and "cat" in _ctxs[0] and "dog" not in _ctxs[0], _ctxs
print("write samples z_ctx in", _ctxs)
del _w

# The bare '#' must anchor the prompt CFG actually leans on. _cfg guides every
# render against CFG_UNCOND, so a drift there rides on every rendered prompt
# with weight -(g - 1) = -3 at guidance 4 -- the one contaminant no other rule
# in a composite phrase can see. '' is a real prompt here and the sampler
# never evaluates it, so anchoring '' would constrain nothing.
assert b.cfg_negative_prompt() == CFG_UNCOND
_f = ops.StepContext(b, stop_index=1, seed=7, cfg=cfg)
ops.rule_loss(dsl.parse_phrase("#")[0], _f)
assert list(_f._z) == [CFG_UNCOND], list(_f._z)
assert {k[0] for k in _f._v} == {CFG_UNCOND}
print("'#' anchors the CFG negative, not ''")
del _f

ctx = ops.StepContext(b, stop_index=2, seed=7, cfg=cfg)
loss = sum(r.alpha * ops.rule_loss(r, ctx)
           for r in dsl.parse_phrase("vibrant colors++"))
loss.backward()
print("op step loss", float(loss.detach()), "params", len(params))

# Check lora_B, not lora_A: lora_B is zero-initialised, so dL/dA is exactly
# zero on the first step for every module and would prove nothing.
kinds: dict[str, list[int]] = {}
for n, p in b.transformer.named_parameters():
    if p.requires_grad and "lora_B" in n:
        k = n.split(".lora_B")[0].split(".")[-1]
        kinds.setdefault(k, [0, 0])
        kinds[k][0] += 1
        kinds[k][1] += int(p.grad is not None and p.grad.abs().sum().item() > 0)
for k, (tot, live) in sorted(kinds.items()):
    print(f"  {k}: {live}/{tot} nonzero grad")
    assert live == tot, (
        f"{k} lost its gradient path -- the preallocated flash KV cache "
        f"detaches k/v; prepare_flash_kv_cache must not run during training")

with tempfile.TemporaryDirectory() as d:
    b.save_trained(os.path.join(d, "lora"))
    assert "adapter_model.safetensors" in os.listdir(os.path.join(d, "lora"))
    # the reload path smoke_sensenova.py --adapter uses for its
    # control-preservation check must accept what save_trained wrote
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(os.path.join(d, "lora", "adapter_model.safetensors"))
    res = set_peft_model_state_dict(b.transformer, sd)
    assert not getattr(res, "unexpected_keys", []), res
    print("adapter round-trip ok:", len(sd), "tensors")

try:
    b.attach_encoder_lora(8)
    raise SystemExit("attach_encoder_lora must refuse: there is no separable encoder")
except NotImplementedError:
    pass

print("max vram GiB", torch.cuda.max_memory_allocated(DEVICE) / 2**30)
print("ALL TINY CHECKS PASSED")
