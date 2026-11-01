import torch
import torch.nn.functional as F

from minillm.layers import scaled_dot_product_attention


def test_matches_reference(no_builtin_softmax):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 5)
    ref = F.scaled_dot_product_attention(q, k, v)
    with no_builtin_softmax():
        got = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, ref, atol=1e-5)


def test_causal_mask(no_builtin_softmax):
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 4)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    with no_builtin_softmax():
        got = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(got, ref, atol=1e-5)
