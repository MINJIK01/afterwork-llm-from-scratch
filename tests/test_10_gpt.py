import math

import torch
import torch.nn.functional as F

from minillm.model import GPT, GPTConfig


def test_forward_shapes_and_loss(tiny_model, tiny_cfg):
    idx = torch.randint(0, tiny_cfg.vocab_size, (3, 10))
    tgt = torch.randint(0, tiny_cfg.vocab_size, (3, 10))
    logits, loss = tiny_model(idx, tgt)
    assert logits.shape == (3, 10, tiny_cfg.vocab_size)
    assert torch.allclose(loss, F.cross_entropy(logits.flatten(0, 1), tgt.flatten()), atol=1e-6)
    assert tiny_model(idx)[1] is None


def test_initial_loss_near_uniform(tiny_model, tiny_cfg):
    idx = torch.randint(0, tiny_cfg.vocab_size, (8, 16))
    tgt = torch.randint(0, tiny_cfg.vocab_size, (8, 16))
    _, loss = tiny_model(idx, tgt)
    assert abs(loss.item() - math.log(tiny_cfg.vocab_size)) < 0.3


def test_weight_tying_and_init():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=512, block_size=16, n_layer=4, n_head=4, d_model=128, ffn_hidden=256)
    m = GPT(cfg)
    assert m.lm_head.weight is m.tok_emb.weight
    assert abs(m.tok_emb.weight.std().item() - 0.02) < 0.002
    assert abs(m.blocks[0].attn.qkv.weight.std().item() - 0.02) < 0.002
    resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
    assert abs(m.blocks[0].attn.proj.weight.std().item() - resid_std) < 0.001
    assert abs(m.blocks[0].ffn.w_down.weight.std().item() - resid_std) < 0.001
    d, f, L, V = cfg.d_model, cfg.ffn_hidden, cfg.n_layer, cfg.vocab_size
    assert m.num_params() == V * d + L * (4 * d * d + 3 * d * f + 2 * d) + d


def test_causality(tiny_model):
    idx = torch.randint(0, 64, (1, 12))
    idx2 = idx.clone()
    idx2[0, 8:] = (idx2[0, 8:] + 1) % 64
    assert torch.allclose(tiny_model(idx)[0][:, :8], tiny_model(idx2)[0][:, :8], atol=1e-6)
