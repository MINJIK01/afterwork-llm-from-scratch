# Step 03 확인 문제 해설

[← 본문](../03_manual_backward.md)

## 1. $Y = XW$ (전치 없음) 규약

$W \in \mathbb R^{d_{in}\times d_{out}}$, $Y_{nk} = \sum_i X_{ni}W_{ik}$:

$$\frac{\partial L}{\partial X} = GW^\top,\qquad \frac{\partial L}{\partial W} = X^\top G,\qquad \frac{\partial L}{\partial b} = \sum_n G_{n,:}.$$

shape 검산: $GW^\top$: $(N, d_{out})(d_{out}, d_{in})$ ✓, $X^\top G$: $(d_{in}, N)(N, d_{out})$ ✓. `nn.Linear` 규약($W$가 $(d_{out}, d_{in})$)과는 $W$의 전치 관계만 다르다.

## 2. RMSNorm backward

$r = \sqrt{\frac1D\sum_k x_k^2 + \epsilon}$, $\hat x = x/r$, $y = \gamma\odot\hat x$, $h := G\odot\gamma$.

$\partial r/\partial x_j = x_j/(Dr)$이므로

$$\frac{\partial \hat x_i}{\partial x_j} = \frac{\delta_{ij}}{r} - \frac{x_i x_j}{D r^3} = \frac1r\left(\delta_{ij} - \frac{\hat x_i\hat x_j}{D}\right),\qquad
\frac{\partial L}{\partial x_j} = \frac1r\left(h_j - \hat x_j\,\overline{h\odot\hat x}\right).$$

LayerNorm 식에서 **평균 제거 항 $-\overline{h}$가 사라진다**. RMSNorm은 평균을 빼지 않으므로 입력에 상수를 더하는 변환에 불변이 아니고, 따라서 gradient가 $\mathbf 1$ 방향에 수직일 필요가 없다. 스케일 불변성은 남아 있으므로 $\hat x$ 방향 성분은 여전히 제거된다. $\partial L/\partial\gamma = \sum_n G\odot\hat x$.

## 3. BatchNorm

정규화 축이 배치(및 공간) 축이고 특성별 통계를 쓴다. 유도는 같고 $D$ 대신 배치 크기 $N$, 행 평균 대신 배치 평균을 쓴다. 차이점:

- 한 샘플의 출력이 같은 배치의 다른 샘플에 의존한다. 그래서 gradient도 배치 내 샘플 간에 결합된다.
- 추론 시에는 배치가 1이거나 분포가 다를 수 있고, 같은 입력의 출력이 함께 들어온 다른 샘플에 따라 달라지면 안 된다. 그래서 학습 중 지수이동평균으로 모은 running mean/var를 고정값으로 쓴다. 학습(배치 통계)과 추론(running 통계)의 동작 차이 때문에 작은 배치나 시퀀스 모델에서 불안정하며, Transformer가 토큰 단위 LayerNorm/RMSNorm을 쓰는 이유이다.
