import torch

from src.adapter.residual_mlp import ResidualMLPAdapter


def test_identity_at_init():
    adapter = ResidualMLPAdapter(d_in=768)
    x = torch.randn(32, 768)
    y = adapter(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_shape():
    adapter = ResidualMLPAdapter(d_in=512)
    x = torch.randn(8, 512)
    assert adapter(x).shape == (8, 512)


def test_gradients_flow():
    adapter = ResidualMLPAdapter(d_in=128)
    x = torch.randn(4, 128, requires_grad=False)
    y = adapter(x).sum()
    y.backward()
    for name, p in adapter.named_parameters():
        assert p.grad is not None, name
