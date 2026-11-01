import torch

from minillm.layers import apply_rope, precompute_rope


def test_matches_complex_rotation():
    hd, T = 8, 10
    cos, sin = precompute_rope(hd, T)
    assert cos.shape == (T, hd // 2)
    x = torch.randn(2, T, hd)
    theta = 10000.0 ** (-torch.arange(0, hd, 2).float() / hd)
    rot = torch.polar(torch.ones(T, hd // 2), torch.arange(T).float()[:, None] * theta)
    ref = torch.view_as_real(torch.view_as_complex(x.reshape(2, T, hd // 2, 2)) * rot).flatten(-2)
    assert torch.allclose(apply_rope(x, cos, sin), ref, atol=1e-5)


def test_norm_preserved_and_relative_position():
    hd = 16
    cos, sin = precompute_rope(hd, 64)
    q, k = torch.randn(hd), torch.randn(hd)

    def rot(x, m):
        return apply_rope(x[None], cos[m : m + 1], sin[m : m + 1])[0]

    assert torch.allclose(rot(q, 5).norm(), q.norm(), atol=1e-5)
    # <R_m q, R_n k>는 m - n에만 의존
    assert torch.allclose(rot(q, 10) @ rot(k, 3), rot(q, 40) @ rot(k, 33), atol=1e-4)
