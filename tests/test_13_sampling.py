import torch

from minillm.generate import filter_probs, generate, sample_next


def test_greedy():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]])
    assert sample_next(logits, temperature=0).tolist() == [1, 0]


def test_top_k():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    p = filter_probs(logits, top_k=2)
    assert (p[0, [0, 3]] == 0).all() and torch.allclose(p.sum(), torch.tensor(1.0))
    assert torch.allclose(p[0, 1] / p[0, 2], torch.exp(torch.tensor(1.0)))


def test_top_p():
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    assert torch.allclose(filter_probs(logits, top_p=0.7), torch.tensor([[0.625, 0.375, 0.0, 0.0]]))
    assert torch.allclose(filter_probs(logits, top_p=0.75), torch.tensor([[0.625, 0.375, 0.0, 0.0]]))
    assert torch.allclose(filter_probs(logits, top_p=0.81), torch.tensor([[0.5, 0.3, 0.15, 0.0]]) / 0.95)
    assert torch.allclose(filter_probs(logits, top_p=0.4), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_sampling_frequency():
    logits = torch.tensor([[0.0, 1.0, 2.0]]).repeat(20000, 1)
    s = sample_next(logits, temperature=1.0, generator=torch.Generator().manual_seed(0))
    freq = torch.bincount(s, minlength=3).float() / len(s)
    assert torch.allclose(freq, torch.softmax(torch.tensor([0.0, 1.0, 2.0]), -1), atol=0.02)


def test_generate_no_cache(tiny_model):
    idx = torch.randint(0, 64, (2, 5))
    out = generate(tiny_model, idx, 10, temperature=0, use_cache=False)
    assert out.shape == (2, 15) and torch.equal(out[:, :5], idx)
    eos = int(out[0, 5])
    out2 = generate(tiny_model, idx[:1], 10, temperature=0, eos_id=eos, use_cache=False)
    assert out2.shape == (1, 6)
