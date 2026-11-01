import torch

from minillm.align import group_advantages, kl_k3, ppo_clip_loss


def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    a = group_advantages(r, 4).view(2, 4)
    assert torch.allclose(a[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4)
    assert torch.allclose(a[1], torch.zeros(4))


def test_ppo_clip_on_policy_equals_policy_gradient():
    logp = torch.randn(3, 5, requires_grad=True)
    adv = torch.tensor([1.0, -2.0, 0.5])
    mask = torch.ones(3, 5)
    loss = ppo_clip_loss(logp, logp.detach(), adv, mask)
    assert torch.allclose(loss, -adv.mean())
    loss.backward()
    assert torch.allclose(logp.grad, -adv[:, None].expand(3, 5) / 15)


def test_ppo_clip_blocks_gradient_outside_trust_region():
    old = torch.zeros(2, 1)
    logp = torch.tensor([[0.5], [-0.5]], requires_grad=True)  # ρ = 1.65, 0.61
    ppo_clip_loss(logp, old, torch.tensor([1.0, -1.0]), torch.ones(2, 1), clip_eps=0.2).backward()
    assert torch.equal(logp.grad, torch.zeros(2, 1))


def test_ppo_clip_respects_mask():
    logp = torch.randn(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = torch.tensor([1.0, 1.0])
    assert torch.allclose(ppo_clip_loss(logp, logp, adv, mask), torch.tensor(-1.0))


def test_kl_k3_unbiased_estimate():
    g = torch.Generator().manual_seed(0)
    pi = torch.softmax(torch.randn(6, generator=g), -1)
    ref = torch.softmax(torch.randn(6, generator=g), -1)
    true_kl = (pi * (pi.log() - ref.log())).sum()
    tok = torch.multinomial(pi, 200000, replacement=True, generator=g)
    est = kl_k3(pi.log()[tok], ref.log()[tok], torch.ones(len(tok)))
    assert torch.allclose(est, true_kl, atol=0.01)
    assert kl_k3(pi.log()[tok], pi.log()[tok], torch.ones(len(tok))) == 0
