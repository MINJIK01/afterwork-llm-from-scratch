import torch


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    torch.nn.functional.scaled_dot_product_attention / softmax 사용 금지.
    mask[i, j] == False인 위치는 softmax 전에 -inf로 채운다.
    """
    raise NotImplementedError
