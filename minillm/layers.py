import math

import torch
import torch.nn as nn

from .functional import softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(QK^T / sqrt(d_k)) V.
    q (..., Tq, d), k (..., Tk, d), v (..., Tk, dv). mask: bool, (..., Tq, Tk)로 broadcast, True=참조 허용. (step 06)"""
    raise NotImplementedError("TODO: step 06 (docs/06_*.md)")


def precompute_rope(head_dim, max_seq_len, base=10000.0):
    """θ_i = base^(-2i/head_dim), 각도 m·θ_i의 cos, sin. 각각 (max_seq_len, head_dim // 2). (step 07)"""
    raise NotImplementedError("TODO: step 07 (docs/07_*.md)")


def apply_rope(x, cos, sin):
    """x (..., T, head_dim)의 인접 쌍 (x[2i], x[2i+1])을 위치별 각도로 회전. cos/sin (T, head_dim // 2). (step 07)"""
    raise NotImplementedError("TODO: step 07 (docs/07_*.md)")


class KVCache:
    """레이어 하나의 K/V 버퍼. (B, H, max_len, head_dim)를 미리 할당하고 pos까지 채운다. (step 14)"""

    def __init__(self, batch_size, n_head, max_len, head_dim, device=None, dtype=None):
        self.k = torch.zeros(batch_size, n_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.pos = 0

    def update(self, k, v):
        """새 k, v (B, H, T, hd)를 pos 위치부터 기록하고 pos를 T만큼 전진. 지금까지의 전체 K, V (B, H, pos, hd) 반환."""
        raise NotImplementedError("TODO: step 14 (docs/14_*.md)")


def kv_cache_bytes(n_layer, n_head, head_dim, batch_size, seq_len, bytes_per_elem):
    """전체 레이어 KV cache가 차지하는 바이트 수. (step 14)"""
    raise NotImplementedError("TODO: step 14 (docs/14_*.md)")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention + RoPE. (step 08)"""

    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        """x (B, T, C). cos/sin: 이번 입력 T개 위치의 (T, head_dim // 2).
        cache가 있으면 K/V를 cache에 기록하고 과거 K/V 전체를 참조한다.
        query i의 절대 위치는 Tk - T + i (Tk = 참조 가능한 key 수)이므로 key j <= Tk - T + i만 허용."""
        B, T, C = x.shape
        raise NotImplementedError("TODO: step 08 (docs/08_*.md)")
        if cache is not None:
            k, v = cache.update(k, v)
        raise NotImplementedError("TODO: step 08 (docs/08_*.md)")


def repeat_kv(x, n_rep):
    """(B, Hkv, T, hd) → (B, Hkv·n_rep, T, hd). 각 KV head를 n_rep번 연속 반복해 query head h가 KV head h // n_rep를 쓰게 한다. (step 22)"""
    raise NotImplementedError("TODO: step 22 (docs/22_*.md)")


class GroupedQueryAttention(nn.Module):
    """GQA (Ainslie et al. 2023): n_kv_head개의 K/V head를 n_head // n_kv_head개 query head가 공유. n_kv_head=1이면 MQA. (step 22)"""

    def __init__(self, d_model, n_head, n_kv_head):
        super().__init__()
        assert d_model % n_head == 0 and n_head % n_kv_head == 0
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * n_kv_head * self.head_dim, bias=False)  # 앞 절반 K, 뒤 절반 V
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        """CausalSelfAttention과 같은 규약. cache에는 KV head n_kv_head개만 저장한다."""
        B, T, C = x.shape
        raise NotImplementedError("TODO: step 22 (docs/22_*.md)")
        if cache is not None:
            k, v = cache.update(k, v)
        raise NotImplementedError("TODO: step 22 (docs/22_*.md)")


class RMSNorm(nn.Module):
    """Zhang & Sennrich (2019). (step 09)"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        raise NotImplementedError("TODO: step 09 (docs/09_*.md)")


class SwiGLU(nn.Module):
    """FFN(x) = W_down(SiLU(W_gate x) ⊙ W_up x). Shazeer (2020). (step 09)"""

    def __init__(self, d_model, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        raise NotImplementedError("TODO: step 09 (docs/09_*.md)")


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attn(Norm(x)), 이어서 x + FFN(Norm(x)). (step 09)"""

    def __init__(self, d_model, n_head, ffn_hidden, n_kv_head=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if n_kv_head in (None, n_head):
            self.attn = CausalSelfAttention(d_model, n_head)
        else:
            self.attn = GroupedQueryAttention(d_model, n_head, n_kv_head)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_hidden)

    def forward(self, x, cos, sin, cache=None):
        raise NotImplementedError("TODO: step 09 (docs/09_*.md)")
