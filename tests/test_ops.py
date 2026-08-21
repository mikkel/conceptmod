import torch
import pytest

from conceptmod import dsl, ops


class MockBackend:
    """Deterministic fake: v(prompt) = hash-derived constant field; the
    'trained' model adds a learnable delta so gradients can be checked."""

    device = "cpu"
    latent_shape = (4, 8, 8)

    def __init__(self):
        self.delta = torch.nn.Parameter(torch.zeros(1, *self.latent_shape))
        self.partial_calls = 0

    def _base_v(self, prompt, z):
        g = torch.Generator().manual_seed(abs(hash(prompt)) % (2**31))
        return torch.randn(z.shape, generator=g) * 0.1

    def predict_v(self, prompt, z, timestep, frozen):
        v = self._base_v(prompt, z)
        if not frozen:
            v = v + self.delta
        return v

    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        self.partial_calls += 1
        z = torch.randn(1, *self.latent_shape, generator=generator)
        return z, torch.tensor(500.0)

    def velocity_loss_scale(self, timestep):
        return 1.0

    def cfg_negative_prompt(self):
        return ""

    def render(self, prompt, generator, num_steps, guidance, grad_steps=0,
               frozen=False):
        z = torch.randn(1, *self.latent_shape, generator=generator)
        img = self._base_v(prompt, z)
        if not frozen and grad_steps > 0:
            img = img + self.delta
        return img


@pytest.fixture
def ctx():
    backend = MockBackend()
    return backend, ops.StepContext(backend, stop_index=3, seed=7,
                                    cfg=ops.OpDefaults())


def loss_for(phrase, ctx):
    rules = dsl.parse_phrase(phrase)
    assert len(rules) == 1
    return ops.rule_loss(rules[0], ctx)


ALL_OPS = ["cat++", "cat--", "cat=dog", "=dog", "#", "cat#cat", "cat%dog",
           "photo^painting"]


@pytest.mark.parametrize("phrase", ALL_OPS)
def test_ops_compute_and_backprop(phrase, ctx):
    backend, c = ctx
    loss = loss_for(phrase, c)
    assert torch.isfinite(loss)
    loss.backward()
    assert backend.delta.grad is not None
    assert torch.isfinite(backend.delta.grad).all()


def test_freeze_identity_is_near_zero_but_grads_flow(ctx):
    backend, c = ctx
    # trained == frozen + delta; with delta=0 freeze loss is exactly 0
    loss = loss_for("cat#cat", c)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_exaggerate_guidance_option_changes_loss(ctx):
    backend, c = ctx
    l1 = loss_for("cat++:guidance=1", c)
    c2 = ops.StepContext(backend, stop_index=3, seed=7, cfg=ops.OpDefaults())
    l5 = loss_for("cat++:guidance=5", c2)
    assert l5.item() > l1.item()


def test_z_cache_shared_across_rules(ctx):
    backend, c = ctx
    loss_for("cat++", c)
    loss_for("dog++", c)  # same uncond context -> no new partial_denoise
    assert backend.partial_calls == 1
    loss_for("cat--", c)  # 'cat' context -> one more
    assert backend.partial_calls == 2


def test_erase_moves_away_from_concept(ctx, monkeypatch):
    """After a gradient step on erase loss, the trained model's prediction
    for the concept moves toward the negatively-guided target."""
    backend, c = ctx
    monkeypatch.setattr(ops, "ERASE_TEMPLATES", ["{}"])
    c.cfg.erase_guidance = 1.0
    opt = torch.optim.SGD([backend.delta], lr=1.0)
    z, t = c.z_for("cat")
    before = backend.predict_v("cat", z, t, frozen=False)
    v0 = backend.predict_v("", z, t, frozen=True)
    vc = backend.predict_v("cat", z, t, frozen=True)
    target = v0 - 1.0 * (vc - v0)
    d0 = ((before - target) ** 2).mean().item()
    loss = loss_for("cat--", c)
    loss.backward()
    opt.step()
    after = backend.predict_v("cat", z, t, frozen=False)
    d1 = ((after - target) ** 2).mean().item()
    assert d1 < d0


def test_reward_not_implemented(ctx):
    _, c = ctx
    with pytest.raises(NotImplementedError):
        loss_for(";nice image", c)


class DividedMockBackend(MockBackend):
    """SenseNova-shaped: the head predicts the clean sample and the wrapper
    returns ``(x_pred - z) / (1 - t)``, so ``|v|`` blows up toward t=1."""

    def __init__(self, t):
        super().__init__()
        self.t = float(t)

    def predict_v(self, prompt, z, timestep, frozen):
        return super().predict_v(prompt, z, timestep, frozen) / (1.0 - self.t)

    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        z, _ = super().partial_denoise(prompt, stop_index, num_steps,
                                       guidance, generator)
        return z, torch.tensor(self.t)

    def velocity_loss_scale(self, timestep):
        return 1.0 - float(timestep)


def _divided_loss(t, phrase="cat=dog", honest=True):
    backend = DividedMockBackend(t)
    if not honest:
        backend.velocity_loss_scale = lambda timestep: 1.0
    c = ops.StepContext(backend, stop_index=3, seed=7, cfg=ops.OpDefaults())
    return loss_for(phrase, c).item()


@pytest.mark.parametrize("phrase", ["cat=dog", "cat--", "cat#cat2", "cat++"])
def test_velocity_loss_scale_removes_the_schedule_bias(phrase):
    """A backend whose velocity carries a 1/(1-t) factor must not weigh the
    top of the schedule 16x heavier than the bottom -- that bias is what made
    the SenseNova LoRA learn a prompt-independent DC offset (a global
    relight) instead of the concept edit."""
    flat = _divided_loss(0.0, phrase)
    steep = _divided_loss(0.75, phrase)
    assert steep == pytest.approx(flat, rel=1e-4)
    # and the test bites: unreported, the same rule is (1-t)^-2 = 16x louder
    raw = _divided_loss(0.75, phrase, honest=False)
    assert raw == pytest.approx(flat / 0.25 ** 2, rel=1e-4)


def test_orthogonal_is_already_scale_free():
    """`%` is a cosine, so velocity_loss_scale must not change it."""
    assert _divided_loss(0.75, "cat%dog") == pytest.approx(
        _divided_loss(0.75, "cat%dog", honest=False), rel=1e-6)


def test_sensenova_reports_its_velocity_divisor():
    import types

    from conceptmod.backends.sensenova import SenseNovaBackend

    stub = types.SimpleNamespace(
        model=types.SimpleNamespace(
            config=types.SimpleNamespace(t_eps=0.02)))
    scale = SenseNovaBackend.velocity_loss_scale
    assert scale(stub, torch.tensor(0.0)) == pytest.approx(1.0)
    assert scale(stub, torch.tensor(0.75)) == pytest.approx(0.25)
    # clamped exactly like _t2i_predict_v's divisor
    assert scale(stub, torch.tensor(1.0)) == pytest.approx(0.02)


def test_require_cuda_rejects_cpu():
    from conceptmod.backends.base import require_cuda

    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu")


def test_unknown_backend_rejected():
    from conceptmod.backends import BACKENDS, load_backend

    assert BACKENDS == ("sana", "zimage", "anima", "krea", "sensenova")
    with pytest.raises(ValueError, match="unknown backend"):
        load_backend("nope", device="cpu")
    with pytest.raises(ValueError, match="1.x"):
        load_backend("sdxl", device="cpu")


def test_backends_share_the_same_protocol():
    """Adding a model must not fork the trainer contract."""
    from conceptmod.backends.anima import AnimaBackend
    from conceptmod.backends.krea import KreaBackend
    from conceptmod.backends.sana import SanaBackend
    from conceptmod.backends.sensenova import SenseNovaBackend
    from conceptmod.backends.zimage import ZImageBackend

    required = (
        "encode_text", "encode_text_grad", "predict_v", "partial_denoise",
        "render", "generate", "trainable_parameters", "save_trained",
        "training_defaults", "velocity_loss_scale", "attach_encoder_lora",
        "cfg_negative_prompt",
    )
    for cls in (SanaBackend, ZImageBackend, AnimaBackend, KreaBackend,
                SenseNovaBackend):
        for name in required:
            assert callable(getattr(cls, name)), f"{cls.__name__}.{name}"


def test_encoder_pool_flattens_layered_embeds():
    """Krea stacks per-layer hidden states as (B, L, layers, D)."""
    from conceptmod.backends.base import TextEmbeds
    from conceptmod.encoder_train import as_padded, pool

    embeds = torch.randn(1, 6, 12, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    text = TextEmbeds(embeds, mask)
    padded, m = as_padded(text)
    assert padded.shape == (1, 6, 96)
    assert m.shape == (1, 6)
    p = pool(text)
    expected = embeds.float().flatten(-2)[:, :3].mean(dim=1)
    assert p.shape == (1, 96)
    assert torch.allclose(p, expected, atol=1e-5)


def test_write_context_picks_the_trajectory_z_is_drawn_from(monkeypatch):
    """`=` may only teach the edit where the sampler will look for it.

    The loss moves ``v(a)`` at the latents it is evaluated on; at inference
    those are the latents the sampler visits while rendering *a*. Sampling
    z_ctx in b's context instead only works if the a->b direction carries
    from one trajectory to the other, which it does not on a pixel-space
    backend.
    """
    monkeypatch.setattr(ops, "WRITE_TEMPLATES", ["{}"])
    for mode, want in (("target", "dog"), ("source", "cat")):
        backend = MockBackend()
        c = ops.StepContext(backend, stop_index=3, seed=7,
                            cfg=ops.OpDefaults(write_context=mode))
        loss_for("cat=dog", c)
        assert list(c._z) == [want], mode
        assert backend.partial_calls == 1


def test_write_into_the_empty_prompt_keeps_the_target_context(monkeypatch):
    """"=dog" has no source trajectory to sample; b stays the context."""
    monkeypatch.setattr(ops, "WRITE_TEMPLATES", ["{}"])
    for mode in ("target", "source"):
        backend = MockBackend()
        c = ops.StepContext(backend, stop_index=3, seed=7,
                            cfg=ops.OpDefaults(write_context=mode))
        loss_for("=dog", c)
        assert list(c._z) == ["dog"], mode


def test_write_context_rejects_nonsense(ctx):
    backend, c = ctx
    c.cfg.write_context = "elsewhere"
    with pytest.raises(ValueError, match="write_context"):
        loss_for("cat=dog", c)


def test_sensenova_writes_in_the_source_trajectory():
    from conceptmod.backends.sensenova import SenseNovaBackend

    # the historical default stays put for the backends whose proofs landed
    assert ops.OpDefaults().write_context == "target"
    assert SenseNovaBackend.training_defaults(None)["write_context"] == "source"


def test_sensenova_empty_prompt_is_a_prompt_not_the_cfg_negative():
    """t2i_generate templates '' like any other prompt (251 tokens) and keeps
    a separate 9-token bare query as the CFG negative. Routing '' to the bare
    query put a template difference the size of the concept itself into every
    ``v(p) - v('')`` the DSL forms."""
    import types

    from conceptmod.backends import sensenova as sn

    seen = []

    def build_query(prompt_text, system_message=None, append_text=None):
        seen.append((prompt_text, system_message, append_text))
        return "query"

    ids = torch.zeros(1, 3, dtype=torch.long)
    model = types.SimpleNamespace(
        _build_t2i_query=build_query,
        _build_t2i_text_inputs=lambda tok, q: (ids, ids, None),
        _t2i_prefix_forward=lambda i, x, a: ("cache", torch.zeros(1, 3, 4)),
        _build_t2i_image_indexes=lambda h, w, n, device=None: torch.zeros(
            3, h * w, dtype=torch.long),
    )
    stub = types.SimpleNamespace(model=model, tokenizer=None, token_hw=(2, 2))

    sn.SenseNovaBackend._encode_raw(stub, "")
    sn.SenseNovaBackend._encode_raw(stub, sn.CFG_UNCOND)
    empty, negative = seen
    assert empty[1] is not None and "<think>" in empty[2]
    assert negative[1] is None and negative[2] == sn.IMG_START_TOKEN


def test_bare_hash_freezes_the_backends_own_cfg_negative(monkeypatch):
    """`#` means "freeze the unconditional prompt", and the unconditional
    prompt is whatever CFG anchors on -- not necessarily ''.

    Under ``v_u + g (v_c - v_u)`` a drift in the negative lands on every
    rendered prompt with weight ``-(g - 1)``, so leaving it unanchored is a
    prompt-independent contaminant no other op penalises."""
    backend = MockBackend()
    backend.cfg_negative_prompt = lambda: "<NEG>"
    c = ops.StepContext(backend, stop_index=3, seed=7, cfg=ops.OpDefaults())
    loss_for("#", c)
    assert list(c._z) == ["<NEG>"]
    assert {k[0] for k in c._v} == {"<NEG>"}

    # a named freeze is untouched -- only the bare '#' resolves
    c2 = ops.StepContext(backend, stop_index=3, seed=7, cfg=ops.OpDefaults())
    loss_for("cat#cat", c2)
    assert list(c2._z) == ["cat"]


def test_bare_hash_is_unchanged_where_the_negative_is_the_empty_prompt(ctx):
    backend, c = ctx
    loss_for("#", c)
    assert list(c._z) == [""]


def test_sensenova_declares_the_bare_query_as_its_cfg_negative():
    from conceptmod.backends import sensenova as sn

    assert sn.SenseNovaBackend.cfg_negative_prompt(None) == sn.CFG_UNCOND
    # ...and everything else keeps the empty prompt
    from conceptmod.backends.base import Backend

    assert Backend.cfg_negative_prompt(None) == ""


def test_sensenova_accumulates_because_its_loss_is_not_flat_in_t():
    """One draw per optimizer step is not a sampling detail on this backend:
    the `=` loss spans 3700x across the 16 training timesteps, and AdamW is
    scale invariant, so the quiet 13/16 of draws take full-size steps along
    noise."""
    from conceptmod.backends.sensenova import SenseNovaBackend

    assert ops.OpDefaults().accumulation_steps == 1
    assert SenseNovaBackend.training_defaults(None)["accumulation_steps"] == 8


@pytest.mark.parametrize("accum", [2, 4, 8, 16, 32])
def test_every_optimizer_step_sees_both_ends_of_the_schedule(accum):
    """Stratified draws, so an accumulation window cannot miss the handful of
    timesteps that carry the signal (i.i.d. drawing misses all three of
    SenseNova's top indices in 17% of 8-draw windows)."""
    import random

    from conceptmod.model_train import stop_index_for

    rng = random.Random(0)
    steps = 16
    for window in range(20):
        drawn = [stop_index_for(rng, window * accum + j, accum, steps)
                 for j in range(accum)]
        assert min(drawn) < steps // accum + 1, drawn
        assert max(drawn) >= steps - max(1, steps // accum), drawn
        assert all(0 <= d < steps for d in drawn)


def test_single_step_training_keeps_the_uniform_draw():
    """accumulation_steps=1 must behave exactly as before, so the backends
    whose proofs already landed are untouched."""
    import random

    from conceptmod.model_train import stop_index_for

    r1, r2 = random.Random(0), random.Random(0)
    a = [stop_index_for(r1, i, 1, 16) for i in range(50)]
    b = [r2.randrange(0, 16) for _ in range(50)]
    assert a == b
    assert len(set(a)) > 1


def test_sensenova_cfg_guides_the_empty_prompt_too():
    import types

    from conceptmod.backends import sensenova as sn

    seen = []

    def predict_v(prompt, z, timestep, frozen):
        seen.append(prompt)
        return torch.ones(2) if prompt == sn.CFG_UNCOND else torch.zeros(2)

    stub = types.SimpleNamespace(predict_v=predict_v)
    v = sn.SenseNovaBackend._cfg(stub, "", None, None, 4.0, False)
    assert seen == ["", sn.CFG_UNCOND]
    assert torch.allclose(v, torch.full((2,), -3.0))
    # the negative itself is never guided against itself
    seen.clear()
    sn.SenseNovaBackend._cfg(stub, sn.CFG_UNCOND, None, None, 4.0, False)
    assert seen == [sn.CFG_UNCOND]
