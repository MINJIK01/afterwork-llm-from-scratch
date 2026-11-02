import pytest
import torch
import torch.nn.functional as F

from minillm.flash import flash_attention, online_logsumexp


def test_online_logsumexp():
    x = torch.randn(3, 37, dtype=torch.float64) * 50
    for chunk in (1, 5, 37, 64):
        assert torch.allclose(online_logsumexp(x, chunk), torch.logsumexp(x, -1))


@pytest.mark.parametrize("block", [1, 7, 64])
def test_matches_sdpa(block, no_builtin_softmax):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 20, 8), torch.randn(2, 3, 20, 8), torch.randn(2, 3, 20, 6)
    ref_c, ref_n = F.scaled_dot_product_attention(q, k, v, is_causal=True), F.scaled_dot_product_attention(q, k, v)
    with no_builtin_softmax():
        got_c, got_n = flash_attention(q, k, v, True, block), flash_attention(q, k, v, False, block)
    assert torch.allclose(got_c, ref_c, atol=1e-5)
    assert torch.allclose(got_n, ref_n, atol=1e-5)


def test_causal_with_cache_offset():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 10, 8), torch.randn(2, 10, 8)
    mask = torch.ones(4, 10, dtype=torch.bool).tril(diagonal=6)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(flash_attention(q, k, v, causal=True, block_size=3), ref, atol=1e-5)
