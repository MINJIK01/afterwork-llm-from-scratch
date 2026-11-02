import torch

from .functional import cross_entropy
from .layers import Block
from .optim import clip_grad_norm_


def bf16_round(x):
    """fp32 → bf16 → fp32 변환을 비트 연산으로 재현 (round-to-nearest-even). NaN은 고려하지 않는다.
    bf16은 fp32 비트열의 상위 16비트이므로, 하위 16비트를 반올림해 버리면 된다. (step 23)"""
    bits = x.float().contiguous().view(torch.int32)
    # BEGIN SOLUTION (step 23)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & ~0xFFFF
    return bits.view(torch.float32)
    # END SOLUTION


class LossScaler:
    """fp16 학습용 동적 loss scaling (Micikevicius et al. 2018). (step 23)"""

    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=1000):
        self.scale = init_scale
        self.growth_factor, self.backoff_factor, self.growth_interval = growth_factor, backoff_factor, growth_interval
        self.good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    @torch.no_grad()
    def unscale_(self, params):
        """모든 grad를 scale로 나누고, inf/NaN이 하나라도 있으면 True 반환."""
        grads = [p.grad for p in params if p.grad is not None]
        # BEGIN SOLUTION (step 23)
        for g in grads:
            g.div_(self.scale)
        return not torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
        # END SOLUTION

    def update(self, found_inf):
        """inf/NaN이면 scale에 backoff_factor를 곱하고 카운터 초기화.
        아니면 연속 성공 횟수를 세고, growth_interval번마다 scale에 growth_factor를 곱한다."""
        # BEGIN SOLUTION (step 23)
        if found_inf:
            self.scale *= self.backoff_factor
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps % self.growth_interval == 0:
                self.scale *= self.growth_factor
        # END SOLUTION


def train_step_amp(model, optimizer, batches, grad_clip=1.0, dtype=torch.bfloat16, scaler=None):
    """mixed precision 학습 스텝. forward만 torch.autocast(dtype) 아래에서 실행하고, loss는 logits를 float32로 올려 계산.
    scaler(LossScaler)가 있으면 loss를 scale해서 backward → unscale → inf면 step을 건너뛴다(반환 norm = inf).
    파라미터·gradient·optimizer 상태는 fp32 그대로 둔다(master weights). (평균 loss, grad norm) 반환. (step 23)"""
    device_type = next(model.parameters()).device.type
    params = [p for p in model.parameters() if p.requires_grad]
    # BEGIN SOLUTION (step 23)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        with torch.autocast(device_type, dtype=dtype):
            logits, _ = model(x)
        loss = cross_entropy(logits.float().flatten(0, 1), y.flatten()) / len(batches)
        (scaler.scale_loss(loss) if scaler else loss).backward()
        total += loss.item()
    if scaler is not None and scaler.unscale_(params):
        scaler.update(True)
        return total, float("inf")
    norm = clip_grad_norm_(params, grad_clip)
    optimizer.step()
    if scaler is not None:
        scaler.update(False)
    return total, norm.item()
    # END SOLUTION


class CheckpointFunction(torch.autograd.Function):
    """forward는 no_grad로 실행하고 입력만 저장, backward에서 forward를 다시 계산해 gradient를 구한다. (step 23)
    재계산은 원래 forward와 같은 연산이어야 하므로 forward 시점의 autocast 상태를 저장해 backward에서 복원한다."""

    @staticmethod
    def forward(ctx, fn, *args):
        ctx.device_type = args[0].device.type
        ctx.autocast = (torch.is_autocast_enabled(ctx.device_type), torch.get_autocast_dtype(ctx.device_type))
        # BEGIN SOLUTION (step 23)
        ctx.fn = fn
        ctx.save_for_backward(*args)
        with torch.no_grad():
            return fn(*args)
        # END SOLUTION

    @staticmethod
    def backward(ctx, grad_out):
        """저장한 입력을 detach해 leaf로 만든 뒤 enable_grad + 저장된 autocast 상태 아래에서 fn을 재실행하고
        torch.autograd.backward(out, grad_out)로 파라미터 grad를 누적. 반환: (fn 자리 None, 각 입력의 grad 또는 None)."""
        enabled, dtype = ctx.autocast
        # BEGIN SOLUTION (step 23)
        args = [a.detach().requires_grad_(a.requires_grad) for a in ctx.saved_tensors]
        with torch.enable_grad(), torch.autocast(ctx.device_type, dtype=dtype, enabled=enabled):
            out = ctx.fn(*args)
        torch.autograd.backward(out, grad_out)
        return (None, *[a.grad if a.requires_grad else None for a in args])
        # END SOLUTION


def checkpoint(fn, *args):
    """fn(*args)를 activation checkpointing으로 실행. args는 텐서만, 최소 하나는 requires_grad여야 한다."""
    return CheckpointFunction.apply(fn, *args)


class CheckpointedBlock(Block):
    """학습(grad 활성, cache 없음) 중에는 블록 forward를 checkpoint로 감싼다. state_dict 키는 Block과 같다."""

    def forward(self, x, cos, sin, cache=None):
        if cache is not None or not torch.is_grad_enabled():
            return super().forward(x, cos, sin, cache)
        return checkpoint(lambda h: Block.forward(self, h, cos, sin), x)


def enable_checkpointing(model):
    for b in model.blocks:
        b.__class__ = CheckpointedBlock
    return model
