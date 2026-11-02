# Step 03. Linear / LayerNorm의 해석적 역전파

- 구현: `minillm/functional.py` — `linear_backward`, `layer_norm_forward`, `layer_norm_backward`
- 테스트: `pytest tests/test_03_manual_backward.py`

## 1. 목표

행렬 미분을 손으로 유도해 Linear와 LayerNorm의 backward를 autograd 없이 구현한다. 논문의 수식을 텐서 연산으로 옮기는 연습의 핵심 단계이다.

## 2. 이론

### 2.1 규약

스칼라 $L$과 행렬 $X$에 대해 $\partial L/\partial X$는 $X$와 같은 shape이고 $(\partial L/\partial X)_{ij} = \partial L / \partial X_{ij}$이다. 출력 쪽에서 들어오는 gradient를 $G = \partial L / \partial Y$라 쓴다.

### 2.2 Linear

PyTorch `nn.Linear` 규약: $X \in \mathbb{R}^{N \times d_{in}}$, $W \in \mathbb{R}^{d_{out} \times d_{in}}$, $Y = XW^\top + \mathbf{1}b^\top$.

성분으로 쓰면 $Y_{nk} = \sum_i X_{ni} W_{ki} + b_k$이므로

$$\frac{\partial L}{\partial X_{ni}} = \sum_k G_{nk} W_{ki} \Rightarrow \frac{\partial L}{\partial X} = GW, \qquad
\frac{\partial L}{\partial W_{ki}} = \sum_n G_{nk} X_{ni} \Rightarrow \frac{\partial L}{\partial W} = G^\top X, \qquad
\frac{\partial L}{\partial b} = \sum_n G_{n,:}.$$

**shape 검산**: $GW$: $(N, d_{out})(d_{out}, d_{in}) = (N, d_{in})$ ✓, $G^\top X$: $(d_{out}, N)(N, d_{in})$ ✓. 유도가 헷갈리면 결과 shape이 맞도록 전치·곱셈 순서를 정하는 것이 실용적인 검산법이다.

**FLOPs**: forward $2Nd_{in}d_{out}$, backward는 $dX$, $dW$ 각각 $2Nd_{in}d_{out}$이므로 forward의 2배. 합계 $6Nd_{in}d_{out}$. "학습 FLOPs ≈ 토큰당 6 × 파라미터 수"(Kaplan et al. 2020; step 10, 12)의 출처가 이것이다.

### 2.3 LayerNorm forward

행(토큰) 하나 $x \in \mathbb{R}^D$에 대해

$$\mu = \frac{1}{D}\sum_k x_k,\quad \sigma^2 = \frac{1}{D}\sum_k (x_k - \mu)^2,\quad \hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},\quad y = \gamma \odot \hat x + \beta.$$

분산은 biased($1/D$)이다. 이하 $s = \sqrt{\sigma^2+\epsilon}$, $\mathrm{rstd} = 1/s$.

### 2.4 LayerNorm backward 유도

$g := \partial L / \partial \hat x = G \odot \gamma$로 두면 $\partial L/\partial \gamma = \sum_n G \odot \hat x$, $\partial L/\partial \beta = \sum_n G$는 바로 나온다. 어려운 부분은 $\hat x$의 각 성분이 행의 모든 $x_j$에 의존한다는 점이다.

1. $\partial \mu / \partial x_j = 1/D$.
2. $\partial \sigma^2 / \partial x_j = \frac{2}{D}(x_j - \mu)$ ($\sum_k (x_k - \mu) = 0$이므로 $\mu$를 통한 항은 사라진다).
3. $\partial s / \partial x_j = \frac{1}{2s} \cdot \frac{2}{D}(x_j - \mu) = \hat x_j / D$.
4. 몫의 미분:
$$\frac{\partial \hat x_i}{\partial x_j} = \frac{(\delta_{ij} - \frac{1}{D})s - (x_i - \mu)\frac{\hat x_j}{D}}{s^2} = \frac{1}{s}\left(\delta_{ij} - \frac{1}{D} - \frac{\hat x_i \hat x_j}{D}\right).$$
5. 따라서
$$\frac{\partial L}{\partial x_j} = \sum_i g_i \frac{\partial \hat x_i}{\partial x_j} = \frac{1}{s}\left(g_j - \overline{g} - \hat x_j \, \overline{g \odot \hat x}\right),$$
여기서 $\overline{\cdot}$는 행 내 평균이다.

**해석**: LayerNorm 출력은 입력에 상수를 더하거나($x + c\mathbf 1$) 양수를 곱해도($ax$) 변하지 않는다. 따라서 입력 gradient는 그 두 방향($\mathbf 1$, $\hat x$)에 수직이어야 한다. 실제로 $\sum_j \partial L/\partial x_j = 0$이고, $\epsilon = 0$이면 $\sum_j \hat x_j \partial L / \partial x_j = 0$이다. 위 식은 $g$에서 그 두 성분을 제거한 사영이다.

backward에 필요한 값은 $\hat x$, rstd, $\gamma$뿐이므로 forward에서 이것만 cache에 저장하면 된다. 실제 fused LayerNorm 커널도 같은 방식으로 activation 메모리를 줄인다.

## 3. 구현 가이드

- 입력은 `(N, D)`, 정규화 축은 마지막 축, `keepdim=True`로 broadcast 유지.
- `x.var(-1, unbiased=False)`로 biased 분산을 써야 `F.layer_norm`과 일치한다.
- `layer_norm_forward`는 `(y, cache)`를 반환하고 `layer_norm_backward(grad_out, cache)`는 `(dx, dgamma, dbeta)`를 반환한다. cache 구성은 자유지만 위 유도에 필요한 최소값을 담는 것을 권장한다.

## 4. 검증

`F.linear`, `F.layer_norm`의 autograd gradient와 float64로 비교한다.

## 5. 확인 문제

> 해설: [answers/03_manual_backward.md](answers/03_manual_backward.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. $Y = XW$ (전치 없음) 규약에서는 $dX$, $dW$가 어떻게 바뀌는가?
2. RMSNorm(step 09) $y = \gamma \odot x / \mathrm{rms}(x)$의 backward를 같은 방식으로 유도하라. 어떤 항이 사라지는가?
3. BatchNorm은 정규화 축이 배치 축이다. 같은 유도에서 무엇이 달라지며, 추론 시 왜 running statistics가 필요한가?

## 참고문헌

- J. L. Ba, J. R. Kiros, G. E. Hinton, "Layer Normalization," arXiv:1607.06450, 2016.
- K. B. Petersen, M. S. Pedersen, *The Matrix Cookbook*, 2012.
- J. Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020.
