import torch
import torch.nn as nn

from minillm.lora import LoRALinear, apply_lora, merge_lora, trainable_params
from minillm.model import GPT
from minillm.optim import AdamW


def test_init_equals_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x))
    assert [n for n, p in lora.named_parameters() if p.requires_grad] == ["A", "B"]
    assert trainable_params(lora) == 4 * (16 + 24)


def test_forward_formula_and_merge():
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = LoRALinear(base, r=4, alpha=8)
    lora.B.data = torch.randn(24, 4)
    x = torch.randn(5, 16)
    ref = base(x) + 2.0 * x @ (lora.B @ lora.A).T
    assert torch.allclose(lora(x), ref, atol=1e-5)
    assert torch.allclose(lora.merge()(x), lora(x), atol=1e-5)


def test_gradients_only_to_adapters():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=2)
    lora(torch.randn(3, 8)).sum().backward()
    assert base.weight.grad is None
    assert torch.equal(lora.A.grad, torch.zeros_like(lora.A))  # B = 0이므로 초기 dL/dA = 0
    assert lora.B.grad.abs().sum() > 0


def test_apply_train_merge_gpt(tiny_cfg):
    torch.manual_seed(0)
    model = GPT(tiny_cfg)
    keys = list(model.state_dict())
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, 1)
    ref, _ = model(x)

    apply_lora(model, r=4, alpha=8)
    assert isinstance(model.blocks[0].attn.qkv, LoRALinear) and isinstance(model.lm_head, nn.Linear)
    assert all(n.endswith((".A", ".B")) for n, p in model.named_parameters() if p.requires_grad)
    assert torch.allclose(model(x)[0], ref, atol=1e-6)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    first = model(x, y)[1].item()
    for _ in range(30):
        opt.zero_grad()
        model(x, y)[1].backward()
        opt.step()
    tuned, last = model(x, y)
    assert last.item() < first - 0.3

    merge_lora(model)
    assert list(model.state_dict()) == keys
    assert torch.allclose(model(x)[0], tuned, atol=1e-4)
