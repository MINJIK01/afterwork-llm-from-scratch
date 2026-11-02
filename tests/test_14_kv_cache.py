import torch

from minillm.generate import generate
from minillm.layers import KVCache, kv_cache_bytes


def test_cache_update():
    c = KVCache(1, 2, 8, 4)
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k, v = c.update(k1, v1)
    assert c.pos == 3 and torch.equal(k, k1) and torch.equal(v, v1)
    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = c.update(k2, v2)
    assert c.pos == 4 and torch.equal(k, torch.cat([k1, k2], 2)) and torch.equal(v, torch.cat([v1, v2], 2))


def test_incremental_logits_match_full(tiny_model):
    idx = torch.randint(0, 64, (2, 12))
    full, _ = tiny_model(idx)
    caches = tiny_model.new_caches(2)
    steps = [tiny_model(idx[:, :5], caches=caches)[0]]
    for t in range(5, 12):
        steps.append(tiny_model(idx[:, t : t + 1], caches=caches)[0])
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)


def test_generate_cache_equals_no_cache(tiny_model):
    idx = torch.randint(0, 64, (2, 4))
    a = generate(tiny_model, idx, 20, temperature=0, use_cache=True)
    b = generate(tiny_model, idx, 20, temperature=0, use_cache=False)
    assert torch.equal(a, b)


def test_generate_beyond_block_size_falls_back(tiny_model, tiny_cfg):
    idx = torch.randint(0, 64, (1, 4))
    out = generate(tiny_model, idx, tiny_cfg.block_size + 10, temperature=0)
    assert out.shape == (1, 4 + tiny_cfg.block_size + 10)


def test_kv_cache_bytes():
    # Llama-2-7B: 32 layers, 32 heads, head_dim 128, fp16, 4096 tokens → 2 GiB
    assert kv_cache_bytes(32, 32, 128, 1, 4096, 2) == 2 * 1024**3
