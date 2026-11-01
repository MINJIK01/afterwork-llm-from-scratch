import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """p_i = exp(z_i) / sum_j exp(z_j), max-subtraction trick 필요."""
    raise NotImplementedError


def cross_entropy(p: np.ndarray, y: int) -> float:
    """L = -log(p_y)"""
    raise NotImplementedError


def softmax_xent_backward(z: np.ndarray, y: int) -> np.ndarray:
    """dL/dz_i = p_i - t_i (t: y의 one-hot). softmax(z)를 내부에서 사용해도 된다."""
    raise NotImplementedError
