# Scaled Dot-Product Attention

출처: Vaswani et al., "Attention Is All You Need" (2017), Section 3.2.1.

## 수식

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

- Q: shape (seq_len_q, d_k)
- K: shape (seq_len_k, d_k)
- V: shape (seq_len_k, d_v)
- softmax는 마지막 축(각 query row마다 key들에 대한 분포)에 대해 적용

## Mask (옵션, causal attention용)

mask는 shape (seq_len_q, seq_len_k)의 bool 텐서. mask[i, j] == False인 위치는 softmax 전에 -inf로 채워서 해당 attention weight가 0이 되게 한다.

## 제약

- `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.functional.softmax` 사용 금지. `torch.exp`, `torch.sum`, `@` (matmul)만 써서 직접 구현한다.
- 수치 안정성을 위해 softmax는 row별 max-subtraction을 적용한다.

## 표기법 매핑

| 수식 | 코드 |
|---|---|
| Q, K, V | `q, k, v: torch.Tensor` |
| d_k | `q.shape[-1]` |
| mask | `mask: torch.Tensor \| None`, dtype bool |
| 반환값 | shape (seq_len_q, d_v) |
