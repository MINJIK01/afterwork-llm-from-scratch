import numpy as np
import torch


def pack_documents(docs, eos_id):
    """docs: list[list[int]]. 각 문서 뒤에 eos_id를 붙여 하나의 1D uint16 배열로 연결. (step 11)"""
    # BEGIN SOLUTION (step 11)
    return np.concatenate([np.array(d + [eos_id], dtype=np.uint16) for d in docs])
    # END SOLUTION


def get_batch(data, batch_size, block_size, device="cpu", generator=None):
    """data: 1D 토큰 배열. 무작위 시작점 i에서 x = data[i : i+block_size], y = data[i+1 : i+1+block_size].
    (x, y) 각각 (batch_size, block_size) int64. 시작점은 torch.randint(..., generator=generator)로 뽑는다. (step 11)"""
    # BEGIN SOLUTION (step 11)
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
    # END SOLUTION
