import copy

import torch

from minillm.model import GPT
from minillm.optim import AdamW
from minillm.precision import LossScaler, bf16_round, enable_checkpointing, train_step_amp


def test_bf16_round_matches_torch():
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(-(2**31), 2**31 - 1, (200000,), generator=g).to(torch.int32)
    x = bits.view(torch.float32)
    x = x[torch.isfinite(x)]
    ties = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8), 65504.0, 3.39e38, 1e-40])
    x = torch.cat([x, torch.randn(10000), ties])
    assert torch.equal(bf16_round(x), x.to(torch.bfloat16).float())


def test_loss_scaler():
    s = LossScaler(init_scale=8.0, growth_interval=2)
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([8.0, 16.0])
    assert s.unscale_([p]) is False and torch.equal(p.grad, torch.tensor([1.0, 2.0]))
    s.update(False)
    assert s.scale == 8.0
    s.update(False)
    assert s.scale == 16.0
    p.grad = torch.tensor([float("inf"), 1.0])
    assert s.unscale_([p]) is True
    s.update(True)
    assert s.scale == 8.0 and s.good_steps == 0


def test_train_step_amp_bf16(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, 1)
    ref = tiny_model(x, y)[1].item()
    opt = AdamW(tiny_model.parameters(), lr=3e-3, weight_decay=0.0)
    first, _ = train_step_amp(tiny_model, opt, [(x[:2], y[:2]), (x[2:], y[2:])], dtype=torch.bfloat16)
    assert abs(first - ref) < 0.05
    assert all(p.dtype == torch.float32 and p.grad.dtype == torch.float32 for p in tiny_model.parameters())
    for _ in range(40):
        last, _ = train_step_amp(tiny_model, opt, [(x, y)], dtype=torch.bfloat16)
    assert last < first - 1.0


def test_train_step_amp_skips_step_on_overflow(tiny_model):
    x = torch.randint(0, 64, (2, 8))
    opt = AdamW(tiny_model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in tiny_model.parameters()]
    scaler = LossScaler(init_scale=2.0**200)
    _, norm = train_step_amp(tiny_model, opt, [(x, x)], dtype=torch.bfloat16, scaler=scaler)
    assert norm == float("inf") and scaler.scale == 2.0**199
    assert all(torch.equal(a, b) for a, b in zip(before, tiny_model.parameters()))


def test_checkpointing_same_grads_less_memory(tiny_cfg):
    torch.manual_seed(0)
    a = GPT(tiny_cfg)
    b = enable_checkpointing(copy.deepcopy(a))
    x, y = torch.randint(0, 64, (4, 32)), torch.randint(0, 64, (4, 32))

    def run(model):
        saved = [0]

        def pack(t):
            saved[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = model(x, y)[1]
        loss.backward()
        return loss, saved[0]

    la, bytes_a = run(a)
    lb, bytes_b = run(b)
    assert torch.allclose(la, lb)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)
    assert bytes_b < bytes_a / 2
    assert list(a.state_dict()) == list(b.state_dict())


def test_checkpointing_under_autocast_recomputes_same_ops(tiny_cfg):
    from minillm.functional import cross_entropy

    torch.manual_seed(0)
    a = GPT(tiny_cfg)
    b = enable_checkpointing(copy.deepcopy(a))
    x, y = torch.randint(0, 64, (4, 32)), torch.randint(0, 64, (4, 32))
    for m in (a, b):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, _ = m(x)
        cross_entropy(logits.float().flatten(0, 1), y.flatten()).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-7)
