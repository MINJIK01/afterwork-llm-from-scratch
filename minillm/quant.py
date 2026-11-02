import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(w, bits=8, group_size=None):
    """대칭 균일 양자화. w (out, in)을 입력 축으로 group_size씩 묶어(None이면 행 전체) 그룹별 scale = max|w| / qmax,
    q = clamp(round(w / scale), -qmax, qmax), qmax = 2^(bits-1) - 1.
    (q int8 (out, in), scale float (out, in // group_size, 1)) 반환. (step 19)"""
    # BEGIN TODO (step 19)
    raise NotImplementedError("TODO: step 19 (docs/19_*.md)")
    # END TODO


def dequantize(q, scale):
    """q (out, in), scale (out, n_groups, 1) → float 가중치 (out, in). (step 19)"""
    # BEGIN TODO (step 19)
    raise NotImplementedError("TODO: step 19 (docs/19_*.md)")
    # END TODO


class QuantLinear(nn.Module):
    """weight-only 양자화 Linear. 저장은 int8 + scale, forward 때 역양자화."""

    def __init__(self, linear, bits=8, group_size=None):
        super().__init__()
        q, scale = quantize(linear.weight.data.float(), bits, group_size)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bits = bits
        self.bias = linear.bias

    def forward(self, x):
        return F.linear(x, dequantize(self.q, self.scale).to(x.dtype), self.bias)


def quantize_model(model, bits=8, group_size=None, skip=("lm_head",)):
    """model 안의 nn.Linear를 QuantLinear로 in-place 교체 (이름이 skip에 있는 모듈 제외)."""
    for name, mod in model.named_children():
        if isinstance(mod, nn.Linear) and name not in skip:
            setattr(model, name, QuantLinear(mod, bits, group_size))
        else:
            quantize_model(mod, bits, group_size, skip)
    return model


def weight_bytes(model):
    """가중치 메모리 (이론값). QuantLinear는 bits/8 바이트/원소 + fp32 scale로 계산."""
    q_bytes = sum(m.q.numel() * m.bits // 8 + m.scale.numel() * 4 for m in model.modules() if isinstance(m, QuantLinear))
    return q_bytes + sum(p.numel() * p.element_size() for p in model.parameters())
