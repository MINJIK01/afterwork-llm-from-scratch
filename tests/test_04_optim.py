import copy
import math

import torch

from minillm.optim import AdamW, clip_grad_norm_, cosine_lr


def test_adamw_matches_torch():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    ref = copy.deepcopy(model)
    opt = AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    for _ in range(20):
        for m, o in [(model, opt), (ref, ref_opt)]:
            o.zero_grad()
            ((m(x) - y) ** 2).mean().backward()
            o.step()
    for p, rp in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p, rp, atol=1e-6)


def test_cosine_lr():
    kw = dict(max_lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert math.isclose(cosine_lr(0, **kw), 0.1)
    assert math.isclose(cosine_lr(9, **kw), 1.0)
    assert math.isclose(cosine_lr(10, **kw), 1.0)
    assert math.isclose(cosine_lr(60, **kw), 0.55)
    assert math.isclose(cosine_lr(110, **kw), 0.1)
    assert math.isclose(cosine_lr(500, **kw), 0.1)


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)]
    rs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    for p, r in zip(ps, rs):
        p.grad = torch.randn(3, 3) * 10
        r.grad = p.grad.clone()
    norm = clip_grad_norm_(ps, 1.0)
    ref_norm = torch.nn.utils.clip_grad_norm_(rs, 1.0)
    assert torch.allclose(norm, ref_norm)
    for p, r in zip(ps, rs):
        assert torch.allclose(p.grad, r.grad)


def test_clip_no_op_below_threshold():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([0.3, 0.4])
    clip_grad_norm_([p], 1.0)
    assert torch.allclose(p.grad, torch.tensor([0.3, 0.4]))
