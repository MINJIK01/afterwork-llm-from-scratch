# Step 18. Policy Gradient, PPO-clip, GRPO

- 구현: `minillm/align.py` — `group_advantages`, `ppo_clip_loss`, `kl_k3`
- 테스트: `pytest tests/test_18_grpo.py`
- 실행: `python scripts/grpo.py`
- 선행: step 13, 15, 17(`token_logprobs`)

## 1. 목표

정책이 직접 샘플을 생성하고 보상을 받아 개선하는 온라인 강화학습을 구현한다. REINFORCE에서 출발해 baseline, importance sampling, PPO clipping, KL 정규화를 거쳐 GRPO(DeepSeekMath)를 완성한다.

## 2. 이론

### 2.1 문제 설정

프롬프트 $x$, 응답 $y\sim\pi_\theta(\cdot\mid x)$, 시퀀스 단위 보상 $R(x,y)$. 목표는 $J(\theta) = \mathbb E_{x,\,y\sim\pi_\theta}[R(x,y)]$의 최대화이다. 보상은 응답 끝에서 한 번 주어진다.

### 2.2 Policy gradient (REINFORCE)

$$\nabla_\theta J = \sum_y R(y)\nabla_\theta\pi_\theta(y) = \sum_y R(y)\,\pi_\theta(y)\nabla_\theta\log\pi_\theta(y) = \mathbb E_{y\sim\pi_\theta}\big[R(y)\nabla_\theta\log\pi_\theta(y)\big].$$

($\nabla\pi = \pi\nabla\log\pi$; Williams 1992.) 보상은 미분할 필요가 없으므로 규칙 기반 보상, 외부 채점기 등 무엇이든 쓸 수 있다. 자기회귀 정책이면 $\log\pi_\theta(y\mid x) = \sum_t\log\pi_\theta(y_t\mid x, y_{<t})$.

### 2.3 Baseline과 advantage

$\mathbb E_{y\sim\pi}[b\,\nabla\log\pi(y)] = b\,\nabla\sum_y\pi(y) = b\,\nabla 1 = 0$이므로 $y$에 의존하지 않는 $b(x)$를 빼도 기댓값은 그대로이고 분산은 줄어든다. $A = R - b$를 advantage라 한다. PPO 기반 RLHF는 학습된 value 함수로 $b$를 추정한다.

### 2.4 GRPO: 그룹 상대 advantage

GRPO는 같은 프롬프트로 $G$개 응답을 샘플링해 그룹 통계로 baseline을 만든다:

$$A_i = \frac{R_i - \mathrm{mean}(R_{1:G})}{\mathrm{std}(R_{1:G}) + \epsilon}.$$

value 네트워크가 필요 없어 메모리와 구현이 단순하다. 그룹 내 보상이 모두 같으면 $A_i = 0$이 되어 그 프롬프트는 학습 신호를 주지 않는다(DAPO는 이런 그룹을 걸러내는 dynamic sampling을 제안했다). 한 응답의 모든 토큰은 같은 $A_i$를 공유한다. 이 레포는 모표준편차(`unbiased=False`)를 쓴다.

### 2.5 Importance sampling과 PPO-clip

한 번 생성한 샘플로 여러 번 업데이트하려면, 샘플을 생성한 정책 $\pi_{old}$와 현재 정책 $\pi_\theta$의 차이를 보정해야 한다. 토큰 단위 비율 $\rho_t = \pi_\theta(y_t\mid\cdot)/\pi_{old}(y_t\mid\cdot)$로 surrogate $\mathbb E[\rho_t A]$를 최대화한다. $\rho$가 1에서 멀어지면 추정 분산이 커지므로 PPO는 다음과 같이 자른다:

$$\mathcal L^{CLIP} = -\mathbb E_t\Big[\min\big(\rho_t A,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)A\big)\Big].$$

| 경우 | 목적함수 | gradient |
|---|---|---|
| $A > 0$, $\rho > 1+\varepsilon$ | $(1+\varepsilon)A$ (상수) | 0 |
| $A < 0$, $\rho < 1-\varepsilon$ | $(1-\varepsilon)A$ (상수) | 0 |
| 그 외 | $\rho A$ 또는 더 작은 쪽 | $\nabla\rho\cdot A$ |

즉 목적함수를 더 개선하는 방향으로 $\rho$가 신뢰 구간을 벗어나면 gradient가 0이 되어 한 배치에서 정책이 과도하게 움직이지 않는다. min을 취하므로 목적함수의 비관적(하한) 추정이다. 첫 업데이트에서는 $\pi_\theta = \pi_{old}$이므로 $\rho = 1$이고 gradient는 2.2–2.3의 policy gradient와 정확히 같다(테스트로 확인).

### 2.6 참조 모델에 대한 KL 정규화

보상만 최대화하면 정책이 보상 규칙의 허점을 이용하거나(예: 키워드만 반복) 언어 능력을 잃는다. 참조 모델(SFT)과의 KL을 penalty로 더한다. 샘플 $y_t\sim\pi_\theta$에 대한 KL 추정량(Schulman 2020), $r = \pi_{ref}(y_t)/\pi_\theta(y_t)$:

| 추정량 | 식 | 비편향 | 항상 $\ge 0$ |
|---|---|---|---|
| $k_1$ | $-\log r$ | O | X |
| $k_2$ | $\frac12(\log r)^2$ | X | O |
| $k_3$ | $(r - 1) - \log r$ | O | O |

$k_3$의 비편향성: $\mathbb E_{\pi_\theta}[r] = \sum_y\pi_\theta\frac{\pi_{ref}}{\pi_\theta} = 1$이므로 $\mathbb E[r - 1] = 0$, 따라서 $\mathbb E[k_3] = \mathbb E[-\log r] = \mathrm{KL}(\pi_\theta\|\pi_{ref})$. 또 $\log r \le r - 1$이므로 $k_3 \ge 0$. GRPO는 KL을 보상에 섞지 않고 loss에 직접 더한다:

$$\mathcal L = \mathcal L^{CLIP} + \beta_{KL}\,\hat{\mathrm{KL}}_{k_3}.$$

### 2.7 토큰 평균 방식

원 GRPO는 시퀀스마다 토큰 평균을 낸 뒤 그룹 평균을 낸다. 이 방식은 긴 응답의 토큰 하나당 가중치를 줄여 길이 편향을 만든다는 분석이 있다(Dr. GRPO, DAPO). 이 레포는 배치 내 전체 유효 토큰에 대한 평균(token-level mean)을 쓴다.

### 2.8 이 레포의 실험 루프 (`grpo.py`)

1. 학습 키워드 $P$개를 뽑고 각 프롬프트로 $G$개 응답을 샘플링한다(rollout).
2. 규칙 보상 → 그룹 advantage.
3. $\pi_{old}$(rollout 시점 정책)와 $\pi_{ref}$의 응답 토큰 로그확률을 `no_grad`로 계산한다.
4. 같은 샘플로 `inner_epochs`번 PPO-clip + KL 업데이트.
5. 주기적으로 평가 키워드 적중률을 측정한다.

응답 마스크는 첫 EOS까지(포함)이다. EOS 이후 위치는 생성 루프가 EOS로 채운 값이므로 학습에서 제외한다.

## 3. 구현 가이드

- `group_advantages(rewards, G)`: `view(-1, G)`, 행별 평균·모표준편차, `flatten()`.
- `ppo_clip_loss(logp, old_logp, advantages, mask, clip_eps)`: `ratio = exp(logp − old_logp)`, advantage를 `(B, 1)`로 broadcast, `min(...)`의 마스크 평균에 음수.
- `kl_k3(logp, ref_logp, mask)`: `d = ref_logp − logp`, `exp(d) − d − 1`의 마스크 평균.

## 4. 검증

그룹 정규화, $\rho = 1$에서 policy gradient와 일치, 신뢰 구간 밖에서 gradient 0, 마스크 처리, 20만 샘플에서 $k_3$ 평균이 정확한 KL과 일치하는지 확인한다. 이 레포는 개발 중 보상을 "250자 이내에서 EOS로 끝맺기"로 바꿔 0.17M 파라미터 모델에서 평균 보상이 30 iteration 만에 0.44 → 1.0으로 오르는 것을 확인했다. 키워드 과제에서 보상이 계속 0이면(작은 모델·짧은 학습) 그룹 advantage가 전부 0이 되어 학습이 일어나지 않으니, 먼저 SFT 단계의 적중률을 확인하라.

## 5. 확인 문제

1. $G = 1$이면 GRPO advantage는 어떻게 되는가? $G$를 키우는 것의 비용과 이점은?
2. 보상이 0/1이고 그룹 내 성공 비율이 $q$일 때, 성공 샘플과 실패 샘플의 advantage를 $q$로 표현하라. $q$가 작을 때 성공 샘플의 가중치는?
3. KL penalty를 loss에 더하는 방식과 보상에서 빼는 방식($R - \beta\log\frac{\pi}{\pi_{ref}}$)의 gradient 차이를 설명하라.

## 참고문헌

- R. J. Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning," Machine Learning 1992.
- J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR 2016.
- L. Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022.
- Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300, 2024 (GRPO).
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025.
- Q. Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," arXiv:2503.14476, 2025.
- Z. Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective," arXiv:2503.20783, 2025 (Dr. GRPO).
- J. Schulman, "Approximating KL Divergence," blog post, 2020 (joschu.net).
