import math

import torch


class AdamW(torch.optim.Optimizer):
    """Loshchilov & Hutter (2019) decoupled weight decay Adam. (step 04)"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                # BEGIN SOLUTION (step 04)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad
                p.mul_(1 - lr * wd)
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
                # END SOLUTION


def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    """2차원 이상 파라미터(행렬, 임베딩)에만 weight decay 적용."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=betas)


def cosine_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    """선형 warmup(step=0에서 max_lr/warmup_steps) 후 max_steps까지 cosine으로 min_lr까지 감소. (step 04)"""
    # BEGIN SOLUTION (step 04)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
    # END SOLUTION


def clip_grad_norm_(params, max_norm):
    """전체 gradient L2 norm이 max_norm을 넘으면 비율을 맞춰 축소. 클리핑 전 norm(tensor) 반환. (step 04)"""
    grads = [p.grad for p in params if p.grad is not None]
    # BEGIN SOLUTION (step 04)
    total = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1:
        for g in grads:
            g.mul_(scale)
    return total
    # END SOLUTION
