import math

import torch

from .data import get_batch
from .optim import clip_grad_norm_


def train_step(model, optimizer, batches, grad_clip=1.0):
    """batches: [(x, y), ...] 마이크로배치 리스트. gradient accumulation으로 전체 배치 평균 loss의 gradient를 만든 뒤
    clip → optimizer.step(). (마이크로배치 평균 loss, 클리핑 전 grad norm)을 float으로 반환. (step 12)"""
    # BEGIN SOLUTION (step 12)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        (loss / len(batches)).backward()
        total += loss.item()
    norm = clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return total / len(batches), norm.item()
    # END SOLUTION


@torch.no_grad()
def evaluate(model, data, batch_size, block_size, iters, device="cpu", generator=None):
    """eval 모드에서 iters개 배치 평균 loss와 perplexity = exp(loss) 반환. 끝나면 train 모드로 복귀. (step 12)"""
    # BEGIN SOLUTION (step 12)
    model.eval()
    losses = [model(*get_batch(data, batch_size, block_size, device, generator))[1].item() for _ in range(iters)]
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)
    # END SOLUTION
