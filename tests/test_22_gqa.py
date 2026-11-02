import torch
import torch.nn.functional as F

from minillm.layers import GroupedQueryAttention, apply_rope, kv_cache_bytes, precompute_rope, repeat_kv
from minillm.model import GPT, GPTConfig


def test_repeat_kv():
    x = torch.randn(2, 2, 5, 4)
    r = repeat_kv(x, 3)
    assert r.shape == (2, 6, 5, 4)
    for h in range(6):
        assert torch.equal(r[:, h], x[:, h // 3])
    assert repeat_kv(x, 1) is x


def test_matches_reference():
    torch.manual_seed(0)
    B, T, C, H, Hkv = 2, 7, 32, 8, 2
    hd = C // H
    attn = GroupedQueryAttention(C, H, Hkv)
    cos, sin = precompute_rope(hd, T)
    x = torch.randn(B, T, C)

    q = attn.q(x).view(B, T, H, hd).transpose(1, 2)
    kv = attn.kv(x)
    k = kv[..., : Hkv * hd].reshape(B, T, Hkv, hd).transpose(1, 2)
    v = kv[..., Hkv * hd :].reshape(B, T, Hkv, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    ref = attn.proj(y.transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(attn(x, cos, sin), ref, atol=1e-5)


def test_mqa():
    torch.manual_seed(0)
    m = GroupedQueryAttention(32, 8, 1)
    assert sum(p.numel() for p in m.parameters()) == 32 * 32 + 32 * 2 * 4 + 32 * 32
    cos, sin = precompute_rope(4, 6)
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 32)
    assert torch.allclose(m(x, cos, sin)[:, :4], m(x2, cos, sin)[:, :4], atol=1e-6)


def test_gpt_with_gqa_cache_matches_full():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, d_model=32, ffn_hidden=64, n_kv_head=2)
    model = GPT(cfg)
    assert isinstance(model.blocks[0].attn, GroupedQueryAttention)
    idx = torch.randint(0, 64, (2, 12))
    full, _ = model(idx)
    caches = model.new_caches(2)
    assert caches[0].k.shape[1] == 2
    steps = [model(idx[:, :5], caches=caches)[0]] + [model(idx[:, t : t + 1], caches=caches)[0] for t in range(5, 12)]
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
    assert kv_cache_bytes(2, 2, 8, 1, 32, 4) * 2 == kv_cache_bytes(2, 4, 8, 1, 32, 4)
