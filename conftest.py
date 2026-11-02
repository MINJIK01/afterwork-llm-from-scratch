import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

# 대상 패키지 선택: 기본 minillm/, MINILLM_SOLUTIONS=1 이면 정답, MINILLM_ONLY=NN 이면 해당 step만 본인 구현 (tools/hybrid.py)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "tools"))
from hybrid import minillm_path  # noqa: E402

sys.path.insert(0, str(minillm_path()))


@pytest.fixture
def tiny_cfg():
    from minillm.model import GPTConfig

    return GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32, ffn_hidden=64)


@pytest.fixture
def tiny_model(tiny_cfg):
    from minillm.model import GPT

    torch.manual_seed(0)
    return GPT(tiny_cfg)


@contextmanager
def _no_builtin_softmax():
    def banned(*args, **kwargs):
        raise AssertionError("torch 내장 softmax/attention 사용 금지")

    targets = [
        (torch, "softmax"),
        (torch, "log_softmax"),
        (torch.Tensor, "softmax"),
        (torch.Tensor, "log_softmax"),
        (torch.nn.functional, "softmax"),
        (torch.nn.functional, "log_softmax"),
        (torch.nn.functional, "scaled_dot_product_attention"),
    ]
    with pytest.MonkeyPatch.context() as mp:
        for obj, name in targets:
            mp.setattr(obj, name, banned)
        yield


@pytest.fixture
def no_builtin_softmax():
    """with no_builtin_softmax(): 블록 안에서 torch 내장 softmax/SDPA 호출을 금지한다."""
    return _no_builtin_softmax
