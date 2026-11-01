import torch
import torch.nn as nn

from .functional import log_softmax


def build_sft_example(prompt_ids, response_ids, block_size, ignore_index=-100):
    """s = (prompt + response)[: block_size + 1]. input = s[:-1], labels = s[1:].
    단, 예측 대상이 프롬프트 토큰인 위치의 label은 ignore_index. block_size보다 짧으면 input은 0, label은 ignore_index로 오른쪽 padding.
    (input_ids, labels) 각각 (block_size,) int64. (step 15)"""
    raise NotImplementedError("TODO: step 15 (docs/15_*.md)")


def bradley_terry_loss(r_chosen, r_rejected):
    """-log σ(r_chosen - r_rejected)의 평균. F.logsigmoid 없이 수치 안정적으로, z=0에서도 gradient가 정확해야 한다. (step 16)"""
    raise NotImplementedError("TODO: step 16 (docs/16_*.md)")


class RewardModel(nn.Module):
    """GPT 백본 + 스칼라 head. 시퀀스별 마지막 유효 토큰(lengths - 1) 위치의 hidden으로 보상을 낸다. (step 16)"""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg.d_model, 1)

    def forward(self, idx, lengths):
        """idx (B, T), lengths (B,) → 보상 (B,)."""
        raise NotImplementedError("TODO: step 16 (docs/16_*.md)")


def token_logprobs(logits, targets):
    """logits (B, T, V), targets (B, T) → 각 위치의 log p(target) (B, T). ignore_index(음수) 위치 값은 아무거나 무방. (step 17)"""
    raise NotImplementedError("TODO: step 17 (docs/17_*.md)")


def sequence_logprobs(model, input_ids, labels, ignore_index=-100):
    """labels != ignore_index 위치의 log p(label) 합 (B,). (step 17)"""
    raise NotImplementedError("TODO: step 17 (docs/17_*.md)")


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Rafailov et al. (2023). 입력은 모두 시퀀스 log-prob (B,).
    implicit reward r = β(log π - log π_ref). (loss, chosen_reward.detach(), rejected_reward.detach()) 반환. (step 17)"""
    raise NotImplementedError("TODO: step 17 (docs/17_*.md)")


def group_advantages(rewards, group_size, eps=1e-6):
    """rewards (N·G,): 같은 프롬프트의 G개 샘플이 연속. A = (r - mean_g) / (std_g + eps), std는 모표준편차. (step 18)"""
    raise NotImplementedError("TODO: step 18 (docs/18_*.md)")


def ppo_clip_loss(logp, old_logp, advantages, mask, clip_eps=0.2):
    """토큰 단위 PPO-clip. logp/old_logp/mask (B, T), advantages (B,)는 시퀀스의 모든 토큰에 broadcast.
    ρ = exp(logp - old_logp), loss = -mask 평균[min(ρA, clip(ρ, 1-ε, 1+ε)A)]. (step 18)"""
    raise NotImplementedError("TODO: step 18 (docs/18_*.md)")


def kl_k3(logp, ref_logp, mask):
    """KL(π || π_ref)의 k3 추정량 exp(d) - d - 1, d = ref_logp - logp. 토큰은 π에서 샘플링된 것. mask 평균. (step 18)"""
    raise NotImplementedError("TODO: step 18 (docs/18_*.md)")
