import torch
import torch.nn.functional as F


def test_matches_reference_no_mask(impl):
    torch.manual_seed(0)
    q = torch.randn(4, 8)
    k = torch.randn(6, 8)
    v = torch.randn(6, 5)

    got = impl.scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(got, expected, atol=1e-5)


def test_matches_reference_with_causal_mask(impl):
    torch.manual_seed(1)
    seq_len = 5
    q = torch.randn(seq_len, 8)
    k = torch.randn(seq_len, 8)
    v = torch.randn(seq_len, 5)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    got = impl.scaled_dot_product_attention(q, k, v, mask=mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(got, expected, atol=1e-5)


def test_banned_functions_not_used(impl):
    # 실행 코드에서 참조하는 이름만 검사 (독스트링 텍스트는 제외)
    names = impl.scaled_dot_product_attention.__code__.co_names
    assert "softmax" not in names
    assert "scaled_dot_product_attention" not in names
