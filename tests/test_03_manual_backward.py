import torch
import torch.nn.functional as F

from minillm.functional import layer_norm_backward, layer_norm_forward, linear_backward


def test_linear_backward():
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, 3, dtype=torch.float64)
    F.linear(x, w, b).backward(g)
    dx, dw, db = linear_backward(x.detach(), w.detach(), g)
    assert torch.allclose(dx, x.grad) and torch.allclose(dw, w.grad) and torch.allclose(db, b.grad)


def test_layer_norm_forward():
    x = torch.randn(5, 8, dtype=torch.float64)
    gamma, beta = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)
    y, _ = layer_norm_forward(x, gamma, beta)
    assert torch.allclose(y, F.layer_norm(x, (8,), gamma, beta, eps=1e-5))


def test_layer_norm_backward():
    x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(8, dtype=torch.float64, requires_grad=True)
    beta = torch.randn(8, dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, 8, dtype=torch.float64)
    F.layer_norm(x, (8,), gamma, beta, eps=1e-5).backward(g)
    _, cache = layer_norm_forward(x.detach(), gamma.detach(), beta.detach())
    dx, dgamma, dbeta = layer_norm_backward(g, cache)
    assert torch.allclose(dx, x.grad) and torch.allclose(dgamma, gamma.grad) and torch.allclose(dbeta, beta.grad)
