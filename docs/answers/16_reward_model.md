# Step 16 확인 문제 해설

[← 본문](../16_reward_model.md)

## 1. $K$개 응답의 순위: Plackett–Luce

순위 $y_{(1)} \succ y_{(2)} \succ \dots \succ y_{(K)}$의 확률을 "남은 후보 중에서 다음 순위를 softmax로 고르는 과정"으로 모델링한다:

$$P(\text{ranking}) = \prod_{k=1}^{K}\frac{\exp r(x, y_{(k)})}{\sum_{j=k}^{K}\exp r(x, y_{(j)})},\qquad \mathcal L = -\log P.$$

$K = 2$이면 BT와 같다. InstructGPT는 $K$개(4–9개) 응답의 순위를 $\binom K2$개의 BT 쌍으로 분해하고, 같은 프롬프트의 쌍들을 한 배치에 넣어 과적합을 줄였다. DPO 논문 부록도 Plackett–Luce 버전의 DPO를 유도한다.

## 2. 항상 긴 응답이 선호된 데이터로 학습하면?

RM은 품질 대신 **길이**를 보상의 주요 특징으로 학습할 수 있다. 길이는 쉽게 계산되는 특징이라 모델이 이를 먼저 이용한다. 이 RM으로 정책을 최적화하면 정책은 길이를 늘리는 방향으로 보상을 올리고, 실제 품질은 개선되지 않는다(reward hacking). Singhal et al. (2023, "A Long Way to Go: Investigating Length Correlations in RLHF")은 RLHF 보상 향상의 상당 부분이 길이 증가로 설명됨을 보였다. 대응: 선호 데이터의 길이 분포 균형, 길이 페널티, 길이에 대한 보상 정규화, RM 학습 시 길이 특징의 영향 제거 등.

## 3. 순진한 안정화 공식의 $z = 0$ gradient 직접 확인

```python
import torch
z = torch.zeros(1, requires_grad=True)
(torch.clamp(-z, min=0) + torch.log1p(torch.exp(-z.abs()))).sum().backward()
print(z.grad)  # tensor([-0.])  — 정답은 -0.5
z = torch.zeros(1, requires_grad=True)
(-torch.nn.functional.logsigmoid(z)).sum().backward()
print(z.grad)  # tensor([-0.5000])
```

`clamp`와 `abs` 모두 꺾인 점 $z = 0$에서 gradient 0을 선택한다(이 레포에서 실행해 확인). 두 함수의 합은 매끄러운 함수이지만 autograd는 조각별 미분을 합할 뿐, 합이 매끄럽다는 사실을 모른다. 같은 함수라도 **어떻게 분해해 쓰느냐**에 따라 특정 점의 gradient가 달라질 수 있다는 것이 이 문제의 요점이다.
