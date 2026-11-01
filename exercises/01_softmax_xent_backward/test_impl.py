import numpy as np
import torch
import torch.nn.functional as F


def test_softmax_matches_torch(impl):
    z = np.array([2.0, 1.0, -1.0, 0.5])
    got = impl.softmax(z)
    expected = F.softmax(torch.tensor(z), dim=0).numpy()
    assert np.allclose(got, expected, atol=1e-6)


def test_softmax_numerically_stable(impl):
    z = np.array([1000.0, 1000.0, 1000.0])
    p = impl.softmax(z)
    assert np.all(np.isfinite(p))
    assert np.allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_cross_entropy_matches_torch(impl):
    z = np.array([2.0, 1.0, -1.0, 0.5])
    y = 2
    p = impl.softmax(z)
    got = impl.cross_entropy(p, y)
    expected = F.cross_entropy(torch.tensor(z).unsqueeze(0), torch.tensor([y])).item()
    assert np.isclose(got, expected, atol=1e-6)


def test_backward_matches_autograd(impl):
    rng = np.random.default_rng(0)
    for _ in range(10):
        z_np = rng.normal(size=5)
        y = int(rng.integers(0, 5))

        z_t = torch.tensor(z_np, requires_grad=True)
        loss = F.cross_entropy(z_t.unsqueeze(0), torch.tensor([y]))
        loss.backward()
        expected_grad = z_t.grad.numpy()

        got_grad = impl.softmax_xent_backward(z_np, y)
        assert np.allclose(got_grad, expected_grad, atol=1e-6)
