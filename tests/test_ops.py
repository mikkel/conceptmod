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


def test_require_cuda_rejects_cpu():
    from conceptmod.backends.base import require_cuda

    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu")


def test_unknown_backend_rejected():
    from conceptmod.backends import load_backend

    with pytest.raises(ValueError, match="unknown backend"):
        load_backend("nope", device="cpu")


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
