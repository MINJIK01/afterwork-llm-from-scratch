# Softmax + Cross-Entropy Backward

입력 로짓 벡터 z ∈ R^C (C개 클래스), 정답 클래스 인덱스 y.

## Forward

Softmax:

    p_i = exp(z_i) / sum_j exp(z_j)

Cross-entropy loss (정답 클래스가 y일 때):

    L = -log(p_y)

## Backward

구해야 하는 것: dL/dz_i (모든 i에 대해).

정답 클래스의 one-hot 벡터를 t (t_y = 1, 그 외 0)라 하면, 아래가 성립한다.

    dL/dz_i = p_i - t_i

이 결과를 chain rule로 직접 유도해서 코드로 옮기는 것이 이번 연습이다. `np.log`, `np.exp`의 수치 불안정성(overflow)을 막기 위해 forward의 softmax는 max-subtraction trick을 써야 한다.

## 표기법 매핑

| 수식 | 코드 |
|---|---|
| z ∈ R^C | `z: np.ndarray`, shape `(C,)` |
| p ∈ R^C | `p: np.ndarray`, shape `(C,)` |
| y (정답 인덱스) | `y: int` |
| dL/dz | 반환값, shape `(C,)` |
