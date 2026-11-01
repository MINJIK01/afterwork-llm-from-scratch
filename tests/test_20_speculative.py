import torch

from minillm.generate import generate, speculative_accept, speculative_generate
from minillm.model import GPT, GPTConfig


def test_accept_preserves_target_distribution():
    g = torch.Generator().manual_seed(0)
    V, gamma, n = 5, 2, 20000
    q = torch.softmax(torch.randn(gamma, V, generator=g) * 2, -1)
    p = torch.softmax(torch.randn(gamma + 1, V, generator=g) * 2, -1)
    first = torch.zeros(V)
    for _ in range(n):
        drafts = torch.stack([torch.multinomial(q[i], 1, generator=g)[0] for i in range(gamma)])
        out = speculative_accept(drafts, q, p, g)
        assert 1 <= len(out) <= gamma + 1
        first[out[0]] += 1
    assert torch.allclose(first / n, p[0], atol=0.015)


def test_accept_all_when_identical():
    q = torch.softmax(torch.randn(3, 6), -1)
    p = torch.cat([q, torch.softmax(torch.randn(1, 6), -1)])
    assert len(speculative_accept(torch.tensor([0, 1, 2]), q, p)) == 4


def test_greedy_speculative_equals_greedy_target(tiny_cfg):
    torch.manual_seed(0)
    target = GPT(tiny_cfg)
    draft = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2, d_model=16, ffn_hidden=32))
    idx = torch.randint(0, 64, (1, 4))
    out, accepted = speculative_generate(target, draft, idx, 20, gamma=3, temperature=0)
    ref = generate(target, idx, 20, temperature=0)
    assert out.shape[1] > 10 and torch.equal(out, ref[:, : out.shape[1]])
    assert sum(accepted) >= out.shape[1] - 4
