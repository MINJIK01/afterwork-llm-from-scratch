import torch
import torch.nn.functional as F

from minillm.layers import Block, RMSNorm, SwiGLU, precompute_rope


def test_rmsnorm():
    norm = RMSNorm(8)
    norm.weight.data = torch.randn(8)
    x = torch.randn(3, 5, 8)
    assert torch.allclose(norm(x), F.rms_norm(x, (8,), norm.weight, eps=1e-6), atol=1e-6)


def test_swiglu():
    torch.manual_seed(0)
    ffn = SwiGLU(8, 12)
    x = torch.randn(3, 8)
    ref = ffn.w_down(F.silu(ffn.w_gate(x)) * ffn.w_up(x))
    assert torch.allclose(ffn(x), ref, atol=1e-6)


def test_block_prenorm_residual():
    torch.manual_seed(0)
    block = Block(16, 4, 32)
    cos, sin = precompute_rope(4, 5)
    x = torch.randn(2, 5, 16)
    h = x + block.attn(block.norm1(x), cos, sin)
    ref = h + block.ffn(block.norm2(h))
    assert torch.allclose(block(x, cos, sin), ref, atol=1e-6)
