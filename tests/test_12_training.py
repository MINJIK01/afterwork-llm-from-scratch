import math

import numpy as np
import torch

from minillm.optim import AdamW
from minillm.trainer import evaluate, train_step


def test_grad_accumulation_equals_full_batch(tiny_model):
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    opt = AdamW(tiny_model.parameters(), lr=0.0, weight_decay=0.0)
    train_step(tiny_model, opt, [(x[:2], y[:2]), (x[2:], y[2:])], grad_clip=1e9)
    acc = [p.grad.clone() for p in tiny_model.parameters()]

    tiny_model.zero_grad()
    tiny_model(x, y)[1].backward()
    for a, p in zip(acc, tiny_model.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


def test_overfit_single_batch(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = AdamW(tiny_model.parameters(), lr=3e-3, weight_decay=0.0)
    first, _ = train_step(tiny_model, opt, [(x, y)])
    for _ in range(150):
        last, norm = train_step(tiny_model, opt, [(x, y)])
    assert last < 0.5 < first and norm > 0


def test_evaluate(tiny_model):
    data = np.random.default_rng(0).integers(0, 64, 2000).astype(np.uint16)
    loss, ppl = evaluate(tiny_model, data, 4, 16, iters=3, generator=torch.Generator().manual_seed(0))
    assert math.isclose(ppl, math.exp(loss), rel_tol=1e-6)
    assert tiny_model.training
