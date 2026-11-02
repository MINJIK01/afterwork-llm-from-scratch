# Step 17 확인 문제 해설

[← 본문](../17_dpo.md)

## 1. $\beta \to 0$과 $\beta \to \infty$

$\pi^*(y\mid x) \propto \pi_{ref}(y\mid x)\exp(r(x,y)/\beta)$.

- $\beta \to 0$: 지수항이 지배해 $\pi^*$가 $\arg\max_y r(x,y)$(단, $\pi_{ref}$의 support 안)에 질량을 모두 몰아주는 결정적 정책이 된다. KL 제약이 사라지므로 보상 모델/데이터의 오차를 그대로 이용하게 된다. DPO loss 관점에서는 implicit reward $\beta\log(\pi/\pi_{ref})$로 같은 margin을 만들려면 log-ratio가 $1/\beta$배 커져야 하므로, 정책이 참조에서 매우 멀어진다.
- $\beta \to \infty$: $\exp(r/\beta) \to 1$이므로 $\pi^* = \pi_{ref}$. 아무것도 바뀌지 않는다. DPO gradient에도 $\beta$가 곱해지지만 $\sigma(\cdot)$ 안의 margin이 매우 민감해져, 실제로는 log-ratio가 조금만 움직여도 loss가 포화된다.

## 2. KL 대신 다른 $f$-divergence

$D_f(\pi\|\pi_{ref}) = \mathbb E_{\pi_{ref}}[f(\pi/\pi_{ref})]$로 정규화하면, 정규화 조건의 Lagrange 승수 $\lambda(x)$를 포함한 정상점 조건에서

$$r(x, y) = \beta f'\!\left(\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}\right) + \lambda(x).$$

$\lambda(x)$는 프롬프트에만 의존하므로 BT의 차이에서 소거되고, $f'$이 가역이면 DPO와 같은 형태의 loss를 얻는다: implicit reward $= \beta f'(\pi_\theta/\pi_{ref})$. reverse KL은 $f(u) = u\log u$, $f'(u) = \log u + 1$로 원래 DPO를 준다(상수 1은 소거). Wang et al. (2024, "Beyond Reverse KL")이 이 일반화를 다룬다.

## 3. 응답 길이 차이와 시퀀스 로그확률

시퀀스 log-ratio는 토큰별 log-ratio의 합이므로 응답이 길수록 절댓값이 커지기 쉽다. chosen이 rejected보다 체계적으로 길면, 정책은 긴 응답의 우도를 조금씩 올리는 것만으로 margin을 크게 만들 수 있다. 그 결과 길이 증가가 보상 향상처럼 보이는 편향이 생긴다(Park et al. 2024, "Disentangling Length from Quality in DPO"). 대응: 길이 정규화(SimPO는 평균 토큰 log-prob을 implicit reward로 쓴다), 길이 차이에 대한 정규화 항, chosen/rejected 길이 균형 데이터 구성. 이 레포의 과제에서는 rejected에 EOS 없이 잘린 긴 응답이 많을 수 있으므로, 학습 후 응답 길이 분포도 확인해 보라.
