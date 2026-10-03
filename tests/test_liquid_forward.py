import torch

from src.models.liquid_network import LiquidECGClassifier


def test_forward_shape():
    model = LiquidECGClassifier(seq_len=187, hidden_size=32, output_size=5)
    x = torch.randn(4, 187)
    out = model(x)
    assert out.shape == (4, 5)


def test_gradient_flow():
    model = LiquidECGClassifier(seq_len=187, hidden_size=16, output_size=5)
    x = torch.randn(2, 187, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
