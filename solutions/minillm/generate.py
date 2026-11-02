import torch

from .functional import softmax


def filter_probs(logits, top_k=None, top_p=None):
    """logits (B, V) → top-k, nucleus(top-p) 필터링 후 재정규화한 확률 (B, V).
    top-p: 확률 내림차순 누적합이 top_p에 처음 도달하는 토큰까지 포함하는 최소 집합. (step 13)"""
    # BEGIN SOLUTION (step 13)
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = softmax(logits, dim=-1)
    if top_p is not None:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        sorted_p = sorted_p.masked_fill(sorted_p.cumsum(-1) - sorted_p >= top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs
    # END SOLUTION


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits (B, V) → 다음 토큰 (B,). temperature == 0이면 greedy(argmax). (step 13)"""
    # BEGIN SOLUTION (step 13)
    if temperature == 0:
        return logits.argmax(-1)
    probs = filter_probs(logits / temperature, top_k, top_p)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    # END SOLUTION


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None, use_cache=True, generator=None):
    """idx (B, T0) 뒤에 최대 max_new_tokens개를 생성해 (B, T0 + n) 반환.
    use_cache=True: 첫 스텝에 프롬프트 전체(prefill), 이후 직전 토큰 1개만 입력. False: 매 스텝 전체(최근 block_size) 재계산.
    eos_id를 낸 시퀀스는 이후 eos_id로 채우고, 모든 시퀀스가 끝나면 조기 종료.
    총 길이가 block_size를 넘으면 RoPE 테이블 범위를 벗어나므로 cache 없이 sliding window로 생성한다."""
    model.eval()
    use_cache = use_cache and idx.shape[1] + max_new_tokens <= model.cfg.block_size
    caches = model.new_caches(idx.shape[0], idx.shape[1] + max_new_tokens) if use_cache else None
    done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    inp = idx
    for _ in range(max_new_tokens):
        # BEGIN SOLUTION (step 13)
        if caches is None:
            inp = idx[:, -model.cfg.block_size :]
        logits, _ = model(inp, caches=caches)
        nxt = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        # END SOLUTION
        if eos_id is not None:
            nxt = nxt.masked_fill(done, eos_id)
            done |= nxt == eos_id
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        inp = nxt[:, None]
        if done.all():
            break
    return idx


def _probs(logits, temperature):
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return softmax(logits.float() / temperature, dim=-1)


def speculative_accept(draft_tokens, q, p, generator=None):
    """Leviathan et al. (2023)의 수락/거절 규칙 (배치 1).
    draft_tokens (γ,): 드래프트 제안, q (γ, V): 각 제안 시점의 드래프트 분포, p (γ+1, V): 같은 시점들 + 다음 1칸의 타깃 분포.
    i번째 제안 x는 확률 min(1, p_i(x)/q_i(x))로 수락. 거절되면 norm(max(0, p_i - q_i))에서 하나 뽑고 종료.
    전부 수락되면 p_γ에서 보너스 토큰 1개를 뽑는다. 확정된 토큰 list[int] (길이 1..γ+1) 반환. (step 20)"""
    # BEGIN SOLUTION (step 20)
    out = []
    for i, t in enumerate(draft_tokens.tolist()):
        if torch.rand((), generator=generator) < p[i, t] / q[i, t]:
            out.append(t)
            continue
        residual = (p[i] - q[i]).clamp(min=0)
        out.append(torch.multinomial(residual / residual.sum(), 1, generator=generator).item())
        return out
    out.append(torch.multinomial(p[-1], 1, generator=generator).item())
    return out
    # END SOLUTION


@torch.no_grad()
def speculative_generate(target, draft, idx, max_new_tokens, gamma=4, temperature=1.0, generator=None):
    """idx (1, T0). 드래프트가 γ개 제안 → 타깃 1회 forward로 검증하는 라운드를 반복.
    불변식: 두 모델의 cache에는 '마지막 확정 토큰을 제외한' 확정 토큰들만 들어 있다(cache.pos <= len - 1).
    매 라운드 cache에 아직 없는 토큰 seq[pos:]부터 입력하고, 라운드 후 pos를 min(pos, len(seq) - 1)로 되돌린다.
    (idx (1, T0 + n), 라운드별 확정 토큰 수 리스트) 반환. (step 20)"""
    target.eval()
    draft.eval()
    T0 = idx.shape[1]
    limit = min(T0 + max_new_tokens, target.cfg.block_size, draft.cfg.block_size)
    tc, dc = target.new_caches(1), draft.new_caches(1)
    seq, accepted = idx[0].tolist(), []
    while len(seq) + gamma + 1 <= limit:
        # BEGIN SOLUTION (step 20)
        inp, drafts, qs = seq[dc[0].pos :], [], []
        for _ in range(gamma):
            logits, _ = draft(torch.tensor([inp], device=idx.device), caches=dc)
            q = _probs(logits[0, -1], temperature)
            t = torch.multinomial(q, 1, generator=generator).item()
            drafts.append(t)
            qs.append(q)
            inp = [t]
        verify = seq[tc[0].pos :] + drafts
        logits, _ = target(torch.tensor([verify], device=idx.device), caches=tc)
        p = _probs(logits[0, -(gamma + 1) :], temperature)
        out = speculative_accept(torch.tensor(drafts), torch.stack(qs), p, generator)
        seq += out
        accepted.append(len(out))
        for c in tc + dc:
            c.pos = min(c.pos, len(seq) - 1)
        # END SOLUTION
    return torch.tensor([seq[: T0 + max_new_tokens]], device=idx.device), accepted
