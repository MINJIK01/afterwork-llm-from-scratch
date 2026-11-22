# Step 04. AdamW, Learning-rate 스케줄, Gradient Clipping

- 구현: `minillm/optim.py` — `AdamW.step`, `cosine_lr`, `clip_grad_norm_` (`configure_optimizer`는 제공됨)
- 테스트: `pytest tests/test_04_optim.py`

## 1. 목표

LLM 학습의 표준 최적화 구성(AdamW + warmup/cosine 스케줄 + global norm clipping)을 직접 구현하고, `torch.optim.AdamW`와 동일한 궤적을 내는지 확인한다.

## 2. 이론

### 2.1 SGD에서 Adam까지

- SGD: $\theta_{t} = \theta_{t-1} - \eta g_t$.
- Momentum: gradient의 지수이동평균(EMA) $m_t$로 업데이트해 고주파 노이즈를 줄인다.
- RMSProp: $g^2$의 EMA $v_t$로 좌표별로 나눠, 좌표마다 gradient 스케일이 달라도 비슷한 크기로 업데이트한다.
- Adam은 두 가지를 결합한다:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1 - \beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad \theta_t = \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

### 2.2 Bias correction 유도

$m_0 = 0$에서 시작하면 $m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i$. $g_i$의 기댓값이 $\mathbb E[g]$로 일정하다고 가정하면

$$\mathbb E[m_t] = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}\,\mathbb E[g] = (1 - \beta_1^t)\,\mathbb E[g].$$

따라서 $1 - \beta_1^t$로 나누면 초기 스텝의 0 쪽 편향이 제거된다. $t = 1$에서 $\hat m_1 = g_1$, $\hat v_1 = g_1^2$이므로 첫 업데이트는 좌표별로 $\eta \cdot g/(|g| + \epsilon) \approx \eta\,\mathrm{sign}(g)$이다. $\beta_2 = 0.999$이면 $v$의 bias는 수천 스텝 동안 남으므로 보정 없이는 초기 업데이트가 과도하게 커진다.

### 2.3 L2 정규화 vs decoupled weight decay (AdamW)

SGD에서는 loss에 $\frac{\lambda}{2}\|\theta\|^2$을 더하는 것(L2)과 매 스텝 $\theta \leftarrow (1-\eta\lambda)\theta$로 줄이는 것(weight decay)이 같다. Adam에서는 다르다. L2 항의 gradient $\lambda\theta$가 $\sqrt{\hat v}$로 나눠지므로, gradient가 컸던 좌표일수록 정규화가 약해진다. Loshchilov & Hutter는 decay를 adaptive 업데이트와 분리했다:

$$\theta_t = \theta_{t-1} - \eta\lambda\theta_{t-1} - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

PyTorch 구현 순서: `p.mul_(1 - lr*wd)` → $m, v$ 갱신 → 업데이트. 이 순서를 따라야 테스트의 `torch.optim.AdamW`와 일치한다.

관례상 weight decay는 2차원 이상 텐서(행렬, 임베딩)에만 적용하고, bias와 norm의 gain에는 적용하지 않는다(`configure_optimizer`). LLM에서는 $\beta_2 = 0.95$를 흔히 쓴다. 드물게 등장하는 토큰의 gradient가 갑자기 커질 때 $v$가 더 빨리 반응해 업데이트 폭발을 줄인다(GPT-3, LLaMA 설정).

### 2.4 Warmup + cosine decay

- **Warmup**: 초기 몇 백 스텝 동안 lr을 0 근처에서 선형 증가시킨다. 초기에는 $\hat v$ 추정에 쓰인 샘플이 적어 분산이 크고, 무작위 초기화 상태에서 큰 업데이트가 학습을 불안정하게 만든다(Liu et al. 2020은 이를 adaptive lr의 분산 문제로 분석했다).
- **Cosine decay**: warmup 이후 $\eta_t = \eta_{min} + \frac12(\eta_{max} - \eta_{min})(1 + \cos(\pi r))$, $r = (t - t_w)/(T - t_w)$. 보통 $\eta_{min} = 0.1\eta_{max}$.

이 레포의 규약: step $t < t_w$이면 $\eta_{max}(t+1)/t_w$, $t \ge T$이면 $\eta_{min}$.

### 2.5 Global norm gradient clipping

모든 파라미터의 gradient를 하나의 벡터로 보고 $\|g\|_2 > c$이면 $g \leftarrow g \cdot c/\|g\|_2$. 방향은 보존하고 크기만 제한하므로, 간헐적인 loss spike가 파라미터를 크게 망가뜨리는 것을 막는다(Pascanu et al. 2013). 클리핑 전 norm은 학습 상태 모니터링 지표로 로그에 남긴다.

## 3. 예제로 따라가기

### 3.1 Adam 세 스텝을 표로

파라미터 하나, $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, gradient가 $0.5, 0.5, -0.2$로 들어온다(weight decay 0).

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | 업데이트 크기 $\eta\hat m/\sqrt{\hat v}$ | bias correction 없이 |
|---|---|---|---|---|---|---|---|
| 1 | 0.5 | 0.05 | 0.00025 | 0.5 | 0.25 | **0.1** | 0.3162 |
| 2 | 0.5 | 0.095 | 0.0004998 | 0.5 | 0.25 | **0.1** | 0.4250 |
| 3 | −0.2 | 0.0655 | 0.0005393 | 0.2417 | 0.1799 | **0.0570** | 0.2821 |

읽는 법:

- $t = 1$에서 $m_1 = 0.05$는 실제 gradient 0.5의 1/10밖에 안 된다. $1 - \beta_1^1 = 0.1$로 나누면 $\hat m_1 = 0.5$로 복원된다.
- 같은 gradient가 계속 오면 업데이트 크기는 $\eta$(0.1)이다. gradient가 0.5든 500이든 같다. 이것이 Adam의 스케일 불변성이다.
- bias correction 없이 계산하면 첫 스텝들의 업데이트가 3배 이상 커진다. $v$의 과소추정이 $\sqrt{\ }$를 거쳐 분모를 크게 줄이기 때문이다.
- $t = 3$에서 gradient 부호가 바뀌었지만 $m$은 여전히 양수다. momentum이 노이즈성 반전을 완화한다.

### 3.2 AdamW의 weight decay를 한 스텝으로

$\theta = 1.0$, $\eta = 0.1$, $\lambda = 0.1$, gradient 0.

| 방식 | 계산 | 결과 |
|---|---|---|
| AdamW (decoupled) | $\theta(1 - \eta\lambda)$ | $1.0 \times 0.99 = 0.99$ |
| L2를 gradient에 더함 | $g = \lambda\theta = 0.1$을 Adam에 넣음 → 첫 스텝 업데이트 $\eta\cdot\mathrm{sign}(g)$ | $1.0 - 0.1 = 0.9$ |

L2 방식은 Adam의 정규화 때문에 감쇠량이 $\lambda$와 무관하게 거의 $\eta$가 된다. AdamW는 의도한 비율($\eta\lambda$)만큼 정확히 줄인다.

### 3.3 Warmup + cosine 스케줄

`cosine_lr(step, max_lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)`:

| step | 0 | 4 | 9 | 10 | 35 | 60 | 85 | 109 | 110 이후 |
|---|---|---|---|---|---|---|---|---|---|
| lr | 0.1 | 0.5 | 1.0 | 1.0 | 0.8682 | 0.55 | 0.2318 | 0.1002 | 0.1 |

```
lr
1.0 |          ●●
    |        ●     ●
    |      ●          ●
0.55|    ●               ●          ← 구간 중간(step 60)에서 정확히 (max+min)/2
    |  ●                    ●
0.1 |●                          ●●●●
    +----------------------------------- step
     0   10        60          110
     warmup   cosine decay       min_lr 유지
```

### 3.4 Global norm clipping

두 파라미터의 gradient가 $g_1 = [3]$, $g_2 = [4]$이고 `max_norm = 1`.

| 단계 | 값 |
|---|---|
| 전체 norm $\sqrt{3^2 + 4^2}$ | 5 |
| 스케일 $1/5$ | 0.2 |
| 클리핑 후 | $g_1 = [0.6]$, $g_2 = [0.8]$ (norm 1) |
| 반환값 | 5 (클리핑 **전** norm, 로그용) |

두 gradient의 비율 3:4가 유지된다. 파라미터마다 따로 자르면(예: 각각 1로) 방향이 바뀐다.

### 3.5 확인 코드

```python
import torch
from minillm.optim import AdamW, cosine_lr, clip_grad_norm_

p = torch.nn.Parameter(torch.tensor([0.0]))
opt = AdamW([p], lr=0.1, betas=(0.9, 0.999), weight_decay=0.0)
for g in [0.5, 0.5, -0.2]:
    p.grad = torch.tensor([g]); opt.step(); print(p.item())  # -0.1, -0.2, -0.257 (3.1의 업데이트 누적)

print([round(cosine_lr(s, 1.0, 0.1, 10, 110), 4) for s in [0, 9, 60, 110]])  # [0.1, 1.0, 0.55, 0.1]
```

## 4. 구현 가이드

- `step()`은 `@torch.no_grad()` 아래에서 in-place 연산(`mul_`, `add_`, `addcmul_`, `addcdiv_`)으로 작성한다.
- 상태 `t`는 파라미터별로 1부터 센다. $\epsilon$은 $\sqrt{\hat v}$ **바깥**에 더한다.
- `clip_grad_norm_`: norm은 float으로 계산하고, 스케일 계수 $c/(\|g\| + 10^{-6})$가 1 미만일 때만 곱한다(PyTorch와 동일). 반환값은 클리핑 **전** norm.

## 5. 검증

같은 초기값의 MLP를 20 스텝 학습시켜 `torch.optim.AdamW`와 파라미터가 일치하는지, `cosine_lr`의 경계값, `torch.nn.utils.clip_grad_norm_`과의 일치를 확인한다.

## 6. 확인 문제

> 해설: [answers/04_optimizer.md](answers/04_optimizer.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. $\epsilon$을 $\sqrt{\hat v + \epsilon}$처럼 안쪽에 넣으면 gradient가 매우 작은 좌표에서 업데이트 크기가 어떻게 달라지는가?
2. Adam의 업데이트 크기는 loss를 상수배($L \to kL$)해도 거의 변하지 않는다. 왜인가? SGD는 어떤가?
3. Gradient accumulation(step 12)과 clipping을 함께 쓸 때 clipping은 각 마이크로배치마다 해야 하는가, 누적 후 한 번 해야 하는가?

## 참고문헌

- D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
- I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019.
- I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017.
- R. Pascanu, T. Mikolov, Y. Bengio, "On the difficulty of training recurrent neural networks," ICML 2013.
- L. Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond," ICLR 2020.
- T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020 (학습 하이퍼파라미터, Appendix B).
