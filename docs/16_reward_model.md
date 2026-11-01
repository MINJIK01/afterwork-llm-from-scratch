# Step 16. Bradley–Terry 선호 모델과 Reward Model

- 구현: `minillm/align.py` — `bradley_terry_loss`, `RewardModel.forward`
- 테스트: `pytest tests/test_16_reward_model.py`
- 선행: step 02, 10

## 1. 목표

쌍별 선호 데이터로 보상 함수를 학습하는 Bradley–Terry(BT) 모델의 loss를 수치적으로 안정되게 구현하고, LM 백본 위에 스칼라 보상 head를 얹는다. BT loss는 step 17의 DPO에서 그대로 재사용된다.

## 2. 이론

### 2.1 Bradley–Terry 모델

프롬프트 $x$에 대한 두 응답 중 $y_w$가 $y_l$보다 선호될 확률을 잠재 보상 $r$로 모델링한다:

$$P(y_w \succ y_l \mid x) = \frac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}} = \sigma\big(r(x,y_w) - r(x,y_l)\big).$$

이는 두 응답의 효용에 독립적인 Gumbel 노이즈를 더해 더 큰 쪽을 고르는 확률 선택 모델과 동치이다. 선호 데이터셋 $\mathcal D$에 대한 음의 로그우도:

$$\mathcal L_{BT}(r) = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\log\sigma\big(r(x,y_w) - r(x,y_l)\big).$$

$z = r_w - r_l$에 대한 로지스틱 회귀이다.

### 2.2 Gradient와 식별성

$$\frac{\partial}{\partial z}\big(-\log\sigma(z)\big) = -(1 - \sigma(z)) = -\sigma(-z).$$

$z = 0$(두 응답의 보상이 같음)에서 $-1/2$, 순서를 잘 맞힐수록 0으로 간다. 

$r(x, y) + c(x)$처럼 프롬프트에만 의존하는 상수를 더해도 loss가 변하지 않는다. 즉 보상은 프롬프트별 상수만큼의 자유도를 가진다. step 17의 DPO 유도에서 분배함수 $Z(x)$가 소거되는 것이 바로 이 성질 때문이다.

### 2.3 수치 안정성과 $z = 0$에서의 gradient

$-\log\sigma(z) = \log(1 + e^{-z}) = \mathrm{softplus}(-z)$. $z \ll 0$이면 $e^{-z}$가 overflow하므로 그대로 계산할 수 없다.

흔히 쓰는 안정화 $\max(-z, 0) + \log(1 + e^{-|z|})$는 값은 정확하지만, autograd는 $\max$와 $|\cdot|$의 꺾인 점 $z = 0$에서 각 조각의 subgradient를 따로 고른다. PyTorch에서는 두 조각 모두 0이 선택되어 $z = 0$에서 gradient가 $-1/2$이 아니라 **0**이 된다. DPO는 학습 시작 시 정책 = 참조 모델이라 모든 샘플이 정확히 $z = 0$이므로 학습이 시작되지 않는다.

이 레포의 해법은 step 02의 `log_softmax`를 재사용하는 것이다:

$$-\log\sigma(z) = -\log\frac{e^z}{e^z + e^0} = -\mathrm{log\_softmax}([z, 0])_0.$$

max-subtraction에서 $\max$를 통한 gradient 기여는 log-softmax의 shift 불변성 때문에 정확히 상쇄되므로 모든 $z$에서 gradient가 정확하다. 테스트가 $z = 0$에서 gradient $-0.5$를 확인한다.

### 2.4 Reward model 구조

LM 백본(`GPT.hidden`)의 마지막 hidden state에 선형 head를 붙여 스칼라를 출력한다. causal 모델에서 시퀀스의 마지막 유효 토큰 위치는 프롬프트와 응답 전체를 본 유일한 위치이므로 그 위치의 hidden을 쓴다. 오른쪽 padding이면 마지막 유효 위치는 `lengths - 1`이다.

### 2.5 이 레포에서의 위치

InstructGPT식 RLHF는 (1) 사람 선호 라벨로 RM 학습 → (2) RM을 보상으로 PPO 단계로 진행한다. 이 레포의 과제(키워드 포함 여부)는 보상을 규칙으로 정확히 계산할 수 있어 파이프라인에서는 학습된 RM 대신 규칙 보상을 쓴다. 이 단계는 RM의 구성 요소와 BT loss를 구현해 DPO 유도의 기초로 삼는다. 학습된 RM을 정책 최적화에 쓰면 RM의 오차를 정책이 이용하는 reward overoptimization(Gao et al. 2023)이 발생한다는 점도 기억해 둔다.

## 3. 구현 가이드

- `bradley_terry_loss`: `torch.stack([z, zeros_like(z)], -1)`에 step 02 `log_softmax`, 인덱스 0의 음수 평균. `F.logsigmoid`, `F.softplus`는 쓰지 않는다.
- `RewardModel.forward(idx, lengths)`: `h = self.gpt.hidden(idx)`, `h[torch.arange(B), lengths - 1]` → `self.head` → `squeeze(-1)`.

## 4. 검증

`-F.logsigmoid`와 값 비교, $|z| = 10^4$에서 유한성, $z = 0$에서 gradient, 마지막 유효 토큰 이후를 바꿔도 보상이 불변인지 확인한다.

## 5. 확인 문제

1. 선호 쌍이 아니라 $K$개 응답의 순위가 주어지면 BT를 어떻게 확장하는가? (Plackett–Luce)
2. RM 학습 데이터에서 항상 긴 응답이 선호되었다면 RM은 무엇을 학습할 위험이 있는가?
3. 2.3의 $\max(-z,0) + \log(1+e^{-|z|})$ 구현을 직접 작성해 $z = 0$에서 gradient를 확인해 보라.

## 참고문헌

- R. A. Bradley, M. E. Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons," Biometrika 1952.
- P. Christiano et al., "Deep Reinforcement Learning from Human Preferences," NeurIPS 2017.
- N. Stiennon et al., "Learning to summarize from human feedback," NeurIPS 2020.
- L. Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022.
- L. Gao, J. Schulman, J. Hilton, "Scaling Laws for Reward Model Overoptimization," ICML 2023.
