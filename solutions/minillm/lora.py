import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Hu et al. (2022). y = W x + (α/r)·B A x. W(base)는 고정, A (r, in), B (out, r)만 학습.
    초기화: A ~ Kaiming uniform(a=√5), B = 0 → 학습 시작 시 출력이 base와 동일. (step 24)"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # BEGIN SOLUTION (step 24)
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
        # END SOLUTION

    @torch.no_grad()
    def merge(self):
        """W + (α/r)·B A를 가중치로 갖는 일반 nn.Linear를 반환 (추론 시 추가 비용 0)."""
        # BEGIN SOLUTION (step 24)
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scaling * self.B @ self.A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device)
        # END SOLUTION


TARGETS = ("qkv", "q", "kv", "proj", "w_gate", "w_up", "w_down")


def apply_lora(model, r=8, alpha=16, targets=TARGETS):
    """모든 파라미터를 고정한 뒤, 이름이 targets에 있는 nn.Linear를 LoRALinear로 교체 (lm_head 제외)."""
    model.requires_grad_(False)

    def swap(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
            else:
                swap(child)

    swap(model)
    return model


def merge_lora(model):
    """LoRALinear를 병합된 nn.Linear로 되돌린다. state_dict 키가 원래 모델과 같아진다."""
    for name, child in model.named_children():
        if isinstance(child, LoRALinear):
            setattr(model, name, child.merge())
        else:
            merge_lora(child)
    return model.requires_grad_(True)


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
