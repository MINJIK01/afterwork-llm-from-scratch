import torch

from minillm.model import GPT
from minillm.quant import QuantLinear, dequantize, quantize, quantize_model, weight_bytes


def test_int8_per_channel():
    w = torch.randn(16, 64)
    q, scale = quantize(w, bits=8)
    assert q.dtype == torch.int8 and scale.shape == (16, 1, 1)
    assert q.abs().max() == 127
    err = (dequantize(q, scale) - w).abs()
    assert (err <= scale.reshape(16, 1) / 2 + 1e-6).all()


def test_int4_groupwise():
    w = torch.randn(8, 64)
    w[0, 0] = 50.0  # outlier
    q, scale = quantize(w, bits=4, group_size=16)
    assert scale.shape == (8, 4, 1) and q.min() >= -7 and q.max() <= 7
    err = (dequantize(q, scale) - w).abs().reshape(8, 4, 16)
    assert (err <= scale / 2 + 1e-6).all()
    # outlier가 있는 그룹 외에는 오차가 작게 유지된다
    assert err[0, 1:].max() < err[0, 0].max()


def test_quant_model(tiny_cfg):
    torch.manual_seed(0)
    model = GPT(tiny_cfg)
    idx = torch.randint(0, 64, (2, 10))
    ref, _ = model(idx)
    fp_bytes = weight_bytes(model)
    quantize_model(model, bits=8)
    assert isinstance(model.blocks[0].attn.qkv, QuantLinear)
    assert isinstance(model.lm_head, torch.nn.Linear)
    got, _ = model(idx)
    assert (got - ref).abs().max() < 0.05
    assert weight_bytes(model) < fp_bytes
