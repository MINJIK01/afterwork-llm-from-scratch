# Step 17. Direct Preference Optimization (DPO)

- 구현: `minillm/align.py` — `token_logprobs`, `sequence_logprobs`, `dpo_loss`
- 테스트: `pytest tests/test_17_dpo.py`
- 실행: `python scripts/dpo.py`
- 선행: step 15, 16

## 1. 목표

KL 제약이 있는 보상 최대화 문제의 해를 닫힌 형태로 유도하고, 그 결과로 얻어지는 DPO loss를 구현해 SFT 모델을 선호 데이터로 직접 최적화한다.

## 2. 이론

### 2.1 RLHF 목적함수

$$\max_\pi\; \mathbb E_{x\sim\mathcal D,\,y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big)$$

$\pi_{ref}$는 SFT 모델, $\beta$는 참조 모델에서 벗어나는 정도를 제어한다.

### 2.2 최적 정책의 닫힌 형태

$x$를 고정하고 $Z(x) = \sum_y \pi_{ref}(y\mid x)\exp(r(x,y)/\beta)$로 정의하면

$$
\begin{aligned}
\mathbb E_\pi[r] - \beta\,\mathbb E_\pi\left[\log\frac{\pi}{\pi_{ref}}\right]
&= -\beta\,\mathbb E_\pi\left[\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)\,e^{r(x,y)/\beta}}\right] \\
&= -\beta\,\mathbb E_\pi\left[\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)\,e^{r(x,y)/\beta}/Z(x)}\right] + \beta\log Z(x) \\
&= -\beta\,\mathrm{KL}\big(\pi\,\|\,\pi^*\big) + \beta\log Z(x),
\end{aligned}
$$

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{ref}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right).$$

$\beta\log Z(x)$는 $\pi$와 무관하고 KL은 $\pi = \pi^*$일 때만 0이므로 $\pi^*$가 유일한 최적해이다.

### 2.3 보상의 재매개변수화

위 식을 $r$에 대해 풀면

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)} + \beta\log Z(x).$$

이를 BT 모델(step 16)에 대입하면 $\beta\log Z(x)$는 두 응답에 공통이므로 소거된다:

$$P(y_w\succ y_l\mid x) = \sigma\!\left(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right).$$

### 2.4 DPO loss

$\pi^*$ 자리에 학습할 정책 $\pi_\theta$를 넣고 선호 데이터의 음의 로그우도를 최소화한다:

$$\mathcal L_{DPO}(\theta) = -\mathbb E\left[\log\sigma\big(\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)\big)\right],\qquad \hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.$$

$\hat r_\theta$를 implicit reward라 한다. 별도의 reward model과 강화학습 루프 없이, 분류 문제 형태의 loss 하나로 2.1의 문제를 푼다. 구현상으로는 step 16의 `bradley_terry_loss(r̂_w, r̂_l)` 그대로다.

### 2.5 Gradient 해석

$$\nabla_\theta\mathcal L_{DPO} = -\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(y_l) - \hat r_\theta(y_w)\big)\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].$$

chosen의 우도를 높이고 rejected의 우도를 낮추며, implicit reward가 순서를 틀리게 매긴 샘플일수록 큰 가중치($\sigma(\hat r_l - \hat r_w)$)를 받는다. 학습 시작 시 $\pi_\theta = \pi_{ref}$이므로 $\hat r = 0$, loss $= \ln 2$, 가중치 $= 1/2$이다.

### 2.6 시퀀스 로그확률

자기회귀 분해에 의해 $\log\pi(y\mid x) = \sum_t\log\pi(y_t\mid x, y_{<t})$. step 15의 `build_sft_example`로 프롬프트를 마스킹한 라벨을 만들고, 응답 토큰 위치의 로그확률만 합산한다.

### 2.7 실용상 주의점

- $\pi_{ref}$의 로그확률은 학습 중 변하지 않으므로 한 번 계산해 캐시한다(`dpo.py`).
- 이 레포의 쌍은 SFT 정책에서 샘플링해 한 번 만든 고정 데이터이다(offline). 학습이 진행되면 정책 분포와 데이터 분포가 멀어진다. 반복적으로 쌍을 다시 만드는 iterative/online DPO가 이를 완화한다.
- chosen과 rejected의 로그우도가 함께 감소하면서 margin만 커지는 현상이 보고되어 있다(Razin et al. 2025). 로그에서 margin과 함께 hit rate를 확인해야 하는 이유이다.
- $\beta$가 작으면 참조에서 멀리 벗어나는 것을 허용하고, 크면 보수적으로 움직인다.

### 2.8 이 레포의 실험

SFT 모델로 학습용 키워드 프롬프트마다 $k = 4$개를 샘플링하고, 규칙 보상(키워드 포함 + EOS 종료)이 1인 응답과 0인 응답이 모두 있으면 하나씩 골라 (chosen, rejected) 쌍을 만든다. 학습 후 평가 키워드 적중률이 SFT보다 오르는지 확인한다.

## 3. 구현 가이드

- `token_logprobs(logits, targets)`: step 02 `log_softmax` 후 `gather`. 음수 라벨은 `clamp(min=0)`으로 인덱싱(값은 이후 마스크로 버린다).
- `sequence_logprobs(model, input_ids, labels)`: `(token_logprobs * (labels != -100)).sum(-1)`.
- `dpo_loss`: `rc = β(π_c − ref_c)`, `rr = β(π_r − ref_r)`, `bradley_terry_loss(rc, rr)`와 detach된 `rc, rr` 반환.

## 4. 검증

참조 구현과 로그확률 비교, loss 값, $\pi = \pi_{ref}$에서 loss $= \ln 2$, chosen gradient 음수/rejected gradient 양수를 확인한다. `dpo.py`는 reward accuracy, margin, 학습 전후 적중률을 출력한다.

## 5. 확인 문제

> 해설: [answers/17_dpo.md](answers/17_dpo.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. $\beta\to 0$과 $\beta\to\infty$에서 DPO의 최적 정책은 각각 어떻게 되는가?
2. 2.2의 유도에서 KL 대신 다른 $f$-divergence를 쓰면 닫힌 형태가 어떻게 바뀌는가?
3. chosen과 rejected 응답의 길이 차이가 크면 시퀀스 로그확률 합에 어떤 편향이 생기는가?

## 참고문헌

- R. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023.
- D. M. Ziegler et al., "Fine-Tuning Language Models from Human Preferences," arXiv:1909.08593, 2019.
- M. G. Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences," AISTATS 2024 (IPO).
- K. Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization," ICML 2024.
- N. Razin et al., "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization," ICLR 2025.
