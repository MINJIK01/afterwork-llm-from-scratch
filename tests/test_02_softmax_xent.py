import torch
import torch.nn.functional as F

from minillm.functional import cross_entropy, cross_entropy_backward, log_softmax, softmax


def test_softmax_and_log_softmax(no_builtin_softmax):
    x = torch.randn(4, 7, dtype=torch.float64)
    ref_p, ref_lp = F.softmax(x, -1), F.log_softmax(x, -1)
    with no_builtin_softmax():
        p, lp = softmax(x), log_softmax(x)
    assert torch.allclose(p, ref_p)
    assert torch.allclose(lp, ref_lp)


def test_numerical_stability(no_builtin_softmax):
    x = torch.tensor([[1000.0, 1000.0, -1000.0]])
    with no_builtin_softmax():
        p, lp = softmax(x), log_softmax(x)
    assert torch.isfinite(p).all() and torch.allclose(p, torch.tensor([[0.5, 0.5, 0.0]]))
    assert torch.isfinite(lp[:, :2]).all()


def test_cross_entropy_with_ignore_index():
    logits = torch.randn(6, 5, dtype=torch.float64)
    targets = torch.tensor([0, 4, -100, 2, -100, 1])
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(cross_entropy(logits, targets), ref)


def test_backward_matches_autograd():
    logits = torch.randn(6, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0, 4, -100, 2, -100, 1])
    F.cross_entropy(logits, targets, ignore_index=-100).backward()
    got = cross_entropy_backward(logits.detach(), targets)
    assert torch.allclose(got, logits.grad)
