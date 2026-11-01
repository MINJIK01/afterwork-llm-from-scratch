import torch


def softmax(x, dim=-1):
    """수치 안정적인 softmax. torch.softmax / F.softmax 사용 금지. (step 02)"""
    # BEGIN SOLUTION (step 02)
    x = x - x.amax(dim=dim, keepdim=True)
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)
    # END SOLUTION


def log_softmax(x, dim=-1):
    """log-sum-exp trick으로 계산한 log softmax. (step 02)"""
    # BEGIN SOLUTION (step 02)
    x = x - x.amax(dim=dim, keepdim=True)
    return x - x.exp().sum(dim=dim, keepdim=True).log()
    # END SOLUTION


def cross_entropy(logits, targets, ignore_index=-100):
    """logits (N, C), targets (N,). targets == ignore_index인 위치를 제외한 평균 NLL. (step 02)"""
    # BEGIN SOLUTION (step 02)
    mask = targets != ignore_index
    logp = log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (nll * mask).sum() / mask.sum().clamp(min=1)
    # END SOLUTION


def cross_entropy_backward(logits, targets, ignore_index=-100):
    """cross_entropy의 dL/dlogits (N, C)를 autograd 없이 해석적으로 계산. (step 02)"""
    # BEGIN SOLUTION (step 02)
    mask = (targets != ignore_index).to(logits.dtype)
    grad = softmax(logits, dim=-1)
    grad[torch.arange(len(targets)), targets.clamp(min=0)] -= 1
    return grad * mask.unsqueeze(-1) / mask.sum().clamp(min=1)
    # END SOLUTION


def linear_backward(x, w, grad_out):
    """y = x @ w.T + b (nn.Linear 규약). x (N, in), w (out, in), grad_out (N, out) → (dx, dw, db). (step 03)"""
    # BEGIN SOLUTION (step 03)
    return grad_out @ w, grad_out.T @ x, grad_out.sum(0)
    # END SOLUTION


def layer_norm_forward(x, gamma, beta, eps=1e-5):
    """x (N, D) → (y, cache). cache는 backward에 필요한 값을 담는다. (step 03)"""
    # BEGIN SOLUTION (step 03)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    rstd = (var + eps).rsqrt()
    xhat = (x - mu) * rstd
    return xhat * gamma + beta, (xhat, rstd, gamma)
    # END SOLUTION


def layer_norm_backward(grad_out, cache):
    """grad_out (N, D), cache → (dx, dgamma, dbeta). (step 03)"""
    # BEGIN SOLUTION (step 03)
    xhat, rstd, gamma = cache
    g = grad_out * gamma
    dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
    return dx, (grad_out * xhat).sum(0), grad_out.sum(0)
    # END SOLUTION
