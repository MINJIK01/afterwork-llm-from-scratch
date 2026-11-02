import math

import torch


def online_logsumexp(x, chunk_size):
    """마지막 축의 log-sum-exp를 chunk_size씩 한 번만 훑으며 계산.
    running max m과 running sum s = Σ exp(x_j - m)를 유지하고, m이 커지면 s를 exp(m_old - m_new)로 보정. (step 21)"""
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
    s = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    for c in x.split(chunk_size, dim=-1):
        # BEGIN SOLUTION (step 21)
        m_new = torch.maximum(m, c.amax(-1))
        s = s * torch.exp(m - m_new) + torch.exp(c - m_new.unsqueeze(-1)).sum(-1)
        m = m_new
        # END SOLUTION
    return m + s.log()


def flash_attention(q, k, v, causal=True, block_size=64):
    """FlashAttention과 같은 online softmax 누적으로 softmax(QK^T/√d)V를 계산. Tq×Tk score 행렬 전체를 만들지 않는다.
    q (..., Tq, d), k (..., Tk, d), v (..., Tk, dv). key 축을 block_size씩 순회하며 m(행 최댓값), l(정규화합), acc(가중합)를 갱신.
    causal이면 query i(절대 위치 Tk - Tq + i)는 key j <= Tk - Tq + i만 참조. (step 21)"""
    Tq, Tk = q.shape[-2], k.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    m = torch.full((*q.shape[:-1], 1), float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)
    q_pos = torch.arange(Tq, device=q.device) + (Tk - Tq)
    for start in range(0, Tk, block_size):
        kb, vb = k[..., start : start + block_size, :], v[..., start : start + block_size, :]
        # BEGIN SOLUTION (step 21)
        s = q @ kb.transpose(-2, -1) * scale
        if causal:
            k_pos = torch.arange(start, start + kb.shape[-2], device=q.device)
            s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        p = torch.exp(s - m_new)
        corr = torch.exp(m - m_new)
        l = l * corr + p.sum(-1, keepdim=True)
        acc = acc * corr + p @ vb
        m = m_new
        # END SOLUTION
    return acc / l
