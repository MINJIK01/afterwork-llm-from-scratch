import torch


def softmax(x, dim=-1):
    """수치 안정적인 softmax. torch.softmax / F.softmax 사용 금지. (step 02)"""
    # BEGIN TODO (step 02)
    raise NotImplementedError("TODO: step 02 (docs/02_*.md)")
    # END TODO


def log_softmax(x, dim=-1):
    """log-sum-exp trick으로 계산한 log softmax. (step 02)"""
    # BEGIN TODO (step 02)
    raise NotImplementedError("TODO: step 02 (docs/02_*.md)")
    # END TODO


def cross_entropy(logits, targets, ignore_index=-100):
    """logits (N, C), targets (N,). targets == ignore_index인 위치를 제외한 평균 NLL. (step 02)"""
    # BEGIN TODO (step 02)
    raise NotImplementedError("TODO: step 02 (docs/02_*.md)")
    # END TODO


def cross_entropy_backward(logits, targets, ignore_index=-100):
    """cross_entropy의 dL/dlogits (N, C)를 autograd 없이 해석적으로 계산. (step 02)"""
    # BEGIN TODO (step 02)
    raise NotImplementedError("TODO: step 02 (docs/02_*.md)")
    # END TODO


def linear_backward(x, w, grad_out):
    """y = x @ w.T + b (nn.Linear 규약). x (N, in), w (out, in), grad_out (N, out) → (dx, dw, db). (step 03)"""
    # BEGIN TODO (step 03)
    raise NotImplementedError("TODO: step 03 (docs/03_*.md)")
    # END TODO


def layer_norm_forward(x, gamma, beta, eps=1e-5):
    """x (N, D) → (y, cache). cache는 backward에 필요한 값을 담는다. (step 03)"""
    # BEGIN TODO (step 03)
    raise NotImplementedError("TODO: step 03 (docs/03_*.md)")
    # END TODO


def layer_norm_backward(grad_out, cache):
    """grad_out (N, D), cache → (dx, dgamma, dbeta). (step 03)"""
    # BEGIN TODO (step 03)
    raise NotImplementedError("TODO: step 03 (docs/03_*.md)")
    # END TODO
