# Step 02 확인 문제 해설

[← 본문](../02_softmax_xent.md)

## 1. $p_y \to 1$, $p_y \to 0$일 때 $\partial L/\partial z_y$

$\partial L/\partial z_y = p_y - 1$.

- $p_y \to 1$: $\to 0$. 이미 확신을 가지고 맞힌 샘플은 학습 신호가 거의 없다.
- $p_y \to 0$: $\to -1$. 크게 틀린 샘플은 크기 1의 최대 gradient를 받는다.

학습 초기에는 출력이 거의 균등($p_y \approx 1/V$)이므로 정답 logit의 gradient가 약 $-1$로 크고, 학습이 진행되어 $p_y$가 커질수록 줄어든다. 확률에 MSE를 쓰면 gradient에 softmax Jacobian이 곱해져 $p_y \to 0$에서 오히려 gradient가 사라지는데, cross-entropy는 log가 이를 상쇄해 틀린 예측에서 gradient가 소실되지 않는다.

## 2. fp16에서 max-subtraction 없이 softmax가 overflow하는 $z$

fp16 최댓값은 65504이므로 $e^z > 65504 \iff z > \ln 65504 \approx 11.09$. 분모는 여러 항의 합이므로 그보다 작은 logit에서도 합이 65504를 넘으면 overflow한다(예: $V = 4096$개의 logit이 모두 3.0이면 $4096 \cdot e^3 \approx 82{,}000 > 65504$). fp16 학습에서 softmax/loss를 fp32로 계산하는 이유이다(step 23).

## 3. Label smoothing을 적용한 gradient

목표 분포 $q = (1-\epsilon)\delta_y + \epsilon/C$에 대해 $L = -\sum_i q_i \log p_i$. $\partial \log p_i/\partial z_j = \delta_{ij} - p_j$이므로

$$\frac{\partial L}{\partial z_j} = -\sum_i q_i(\delta_{ij} - p_j) = -q_j + p_j\sum_i q_i = p_j - q_j.$$

($\sum_i q_i = 1$.) 형태는 그대로 "예측 − 목표"이고, 목표가 one-hot에서 $q$로 바뀐다. $p_y$가 $1 - \epsilon + \epsilon/C$에 도달하면 gradient가 0이 되므로, logit이 무한히 커지는 것을 막는 정규화 효과가 있다.
