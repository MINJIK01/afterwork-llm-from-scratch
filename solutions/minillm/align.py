import torch
import torch.nn as nn

from .functional import log_softmax


def build_sft_example(prompt_ids, response_ids, block_size, ignore_index=-100):
    """s = (prompt + response)[: block_size + 1]. input = s[:-1], labels = s[1:].
    단, 예측 대상이 프롬프트 토큰인 위치의 label은 ignore_index. block_size보다 짧으면 input은 0, label은 ignore_index로 오른쪽 padding.
    (input_ids, labels) 각각 (block_size,) int64. (step 15)"""
    # BEGIN SOLUTION (step 15)
    s = (prompt_ids + response_ids)[: block_size + 1]
    x = s[:-1]
    y = [t if j + 1 >= len(prompt_ids) else ignore_index for j, t in enumerate(s[1:])]
    pad = block_size - len(x)
    return torch.tensor(x + [0] * pad), torch.tensor(y + [ignore_index] * pad)
    # END SOLUTION


def bradley_terry_loss(r_chosen, r_rejected):
    """-log σ(r_chosen - r_rejected)의 평균. F.logsigmoid 없이 수치 안정적으로, z=0에서도 gradient가 정확해야 한다. (step 16)"""
    # BEGIN SOLUTION (step 16)
    z = r_chosen - r_rejected
    return -log_softmax(torch.stack([z, torch.zeros_like(z)], -1), -1)[..., 0].mean()
    # END SOLUTION


class RewardModel(nn.Module):
    """GPT 백본 + 스칼라 head. 시퀀스별 마지막 유효 토큰(lengths - 1) 위치의 hidden으로 보상을 낸다. (step 16)"""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg.d_model, 1)

    def forward(self, idx, lengths):
        """idx (B, T), lengths (B,) → 보상 (B,)."""
        # BEGIN SOLUTION (step 16)
        h = self.gpt.hidden(idx)
        return self.head(h[torch.arange(len(idx)), lengths - 1]).squeeze(-1)
        # END SOLUTION


def token_logprobs(logits, targets):
    """logits (B, T, V), targets (B, T) → 각 위치의 log p(target) (B, T). ignore_index(음수) 위치 값은 아무거나 무방. (step 17)"""
    # BEGIN SOLUTION (step 17)
    return log_softmax(logits.float(), dim=-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    # END SOLUTION


def sequence_logprobs(model, input_ids, labels, ignore_index=-100):
    """labels != ignore_index 위치의 log p(label) 합 (B,). (step 17)"""
    # BEGIN SOLUTION (step 17)
    logits, _ = model(input_ids)
    return (token_logprobs(logits, labels) * (labels != ignore_index)).sum(-1)
    # END SOLUTION


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Rafailov et al. (2023). 입력은 모두 시퀀스 log-prob (B,).
    implicit reward r = β(log π - log π_ref). (loss, chosen_reward.detach(), rejected_reward.detach()) 반환. (step 17)"""
    # BEGIN SOLUTION (step 17)
    rc, rr = beta * (pi_chosen - ref_chosen), beta * (pi_rejected - ref_rejected)
    return bradley_terry_loss(rc, rr), rc.detach(), rr.detach()
    # END SOLUTION


def group_advantages(rewards, group_size, eps=1e-6):
    """rewards (N·G,): 같은 프롬프트의 G개 샘플이 연속. A = (r - mean_g) / (std_g + eps), std는 모표준편차. (step 18)"""
    # BEGIN SOLUTION (step 18)
    r = rewards.view(-1, group_size)
    return ((r - r.mean(-1, keepdim=True)) / (r.std(-1, unbiased=False, keepdim=True) + eps)).flatten()
    # END SOLUTION


def ppo_clip_loss(logp, old_logp, advantages, mask, clip_eps=0.2):
    """토큰 단위 PPO-clip. logp/old_logp/mask (B, T), advantages (B,)는 시퀀스의 모든 토큰에 broadcast.
    ρ = exp(logp - old_logp), loss = -mask 평균[min(ρA, clip(ρ, 1-ε, 1+ε)A)]. (step 18)"""
    # BEGIN SOLUTION (step 18)
    ratio = (logp - old_logp).exp()
    adv = advantages.unsqueeze(-1)
    obj = torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    return -(obj * mask).sum() / mask.sum()
    # END SOLUTION


def kl_k3(logp, ref_logp, mask):
    """KL(π || π_ref)의 k3 추정량 exp(d) - d - 1, d = ref_logp - logp. 토큰은 π에서 샘플링된 것. mask 평균. (step 18)"""
    # BEGIN SOLUTION (step 18)
    d = ref_logp - logp
    return ((d.exp() - d - 1) * mask).sum() / mask.sum()
    # END SOLUTION
