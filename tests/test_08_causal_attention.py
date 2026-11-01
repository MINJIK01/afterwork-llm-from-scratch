import torch
import torch.nn.functional as F

from minillm.layers import CausalSelfAttention, apply_rope, precompute_rope


def test_matches_reference():
    torch.manual_seed(0)
    B, T, C, H = 2, 7, 16, 4
    attn = CausalSelfAttention(C, H)
    cos, sin = precompute_rope(C // H, T)
    x = torch.randn(B, T, C)

    q, k, v = attn.qkv(x).split(C, -1)
    q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    ref = attn.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))

    assert torch.allclose(attn(x, cos, sin), ref, atol=1e-5)


def test_causality():
    torch.manual_seed(0)
    attn = CausalSelfAttention(16, 4)
    cos, sin = precompute_rope(4, 6)
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16)
    assert torch.allclose(attn(x, cos, sin)[:, :4], attn(x2, cos, sin)[:, :4], atol=1e-6)
