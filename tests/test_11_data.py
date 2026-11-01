import numpy as np
import torch

from minillm.data import get_batch, pack_documents


def test_pack_documents():
    out = pack_documents([[1, 2], [3], []], eos_id=9)
    assert out.dtype == np.uint16
    assert out.tolist() == [1, 2, 9, 3, 9, 9]


def test_get_batch_shift_and_determinism():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, 4, 16, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16) and x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(4, 15, dtype=torch.int64))
    assert int(y.max()) < 1000
    x2, _ = get_batch(data, 4, 16, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x, x2)
