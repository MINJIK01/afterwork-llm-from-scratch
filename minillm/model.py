import math
from dataclasses import dataclass

import torch.nn as nn

from .functional import cross_entropy
from .layers import Block, KVCache, RMSNorm, precompute_rope


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    ffn_hidden: int = 1024
    rope_base: float = 10000.0


class GPT(nn.Module):
    """Decoder-only Transformer LM. (step 10)"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_head, cfg.ffn_hidden) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.block_size, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # 1) lm_head와 tok_emb 가중치 공유  2) 모든 Linear/Embedding ~ N(0, 0.02²)
        # 3) residual로 더해지는 출력 투영(attn.proj, ffn.w_down) ~ N(0, (0.02/sqrt(2·n_layer))²)
        raise NotImplementedError("TODO: step 10 (docs/10_*.md)")

    def hidden(self, idx, caches=None):
        """idx (B, T) → 최종 RMSNorm까지 통과한 hidden (B, T, d_model). caches: 레이어별 KVCache 리스트."""
        T = idx.shape[1]
        start = caches[0].pos if caches else 0
        assert start + T <= self.cfg.block_size, "block_size 초과"
        cos, sin = self.rope_cos[start : start + T], self.rope_sin[start : start + T]
        raise NotImplementedError("TODO: step 10 (docs/10_*.md)")

    def forward(self, idx, targets=None, caches=None):
        """logits (B, T, V)와 loss(targets가 있으면 평균 cross-entropy, 없으면 None)."""
        raise NotImplementedError("TODO: step 10 (docs/10_*.md)")

    def new_caches(self, batch_size, max_len=None):
        p = self.tok_emb.weight
        hd = self.cfg.d_model // self.cfg.n_head
        return [
            KVCache(batch_size, self.cfg.n_head, max_len or self.cfg.block_size, hd, p.device, p.dtype)
            for _ in range(self.cfg.n_layer)
        ]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
