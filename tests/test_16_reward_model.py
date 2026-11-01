import torch
import torch.nn.functional as F

from minillm.align import RewardModel, bradley_terry_loss


def test_bradley_terry_loss():
    rc, rr = torch.randn(10), torch.randn(10)
    assert torch.allclose(bradley_terry_loss(rc, rr), -F.logsigmoid(rc - rr).mean(), atol=1e-6)


def test_bradley_terry_stable():
    loss = bradley_terry_loss(torch.tensor([0.0, 1e4]), torch.tensor([1e4, 0.0]))
    assert torch.isfinite(loss) and torch.allclose(loss, torch.tensor(5e3))


def test_bradley_terry_gradient_at_zero():
    rc = torch.zeros(1, requires_grad=True)
    bradley_terry_loss(rc, torch.zeros(1)).backward()
    assert torch.allclose(rc.grad, torch.tensor([-0.5]))


def test_reward_model_uses_last_valid_token(tiny_model):
    rm = RewardModel(tiny_model)
    idx = torch.randint(0, 64, (2, 10))
    lengths = torch.tensor([6, 10])
    r = rm(idx, lengths)
    assert r.shape == (2,)
    idx2 = idx.clone()
    idx2[0, 6:] = 0
    assert torch.allclose(rm(idx2, lengths)[0], r[0], atol=1e-6)
    assert torch.allclose(r[0], rm(idx[:1, :6], torch.tensor([6]))[0], atol=1e-5)
