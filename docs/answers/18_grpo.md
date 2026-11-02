# Step 18 확인 문제 해설

[← 본문](../18_grpo.md)

## 1. $G = 1$이면?

그룹 평균이 보상 자신이고 표준편차가 0이므로 $A = 0/(0 + \epsilon) = 0$. 학습 신호가 전혀 없다. GRPO의 baseline은 같은 프롬프트의 다른 샘플에서 오므로 $G \ge 2$가 필요하다. $G$를 키우면 baseline 추정이 정확해지고 0/1 보상에서 "전부 같은 보상"인 그룹(신호 0)의 비율이 줄어들지만, 프롬프트당 생성 비용이 $G$배가 된다. DeepSeekMath는 $G = 64$를 썼다.

## 2. 0/1 보상에서 성공 비율 $q$일 때의 advantage

그룹 평균 $q$, 모표준편차 $\sqrt{q(1-q)}$:

$$A_{success} = \frac{1 - q}{\sqrt{q(1-q)}} = \sqrt{\frac{1-q}{q}},\qquad A_{fail} = \frac{-q}{\sqrt{q(1-q)}} = -\sqrt{\frac{q}{1-q}}.$$

$q = 0.1$이면 성공 $+3$, 실패 $-1/3$. $q$가 작을수록 드문 성공 샘플이 $\approx 1/\sqrt q$의 큰 가중치를 받는다. 어려운 프롬프트에서 우연한 성공을 강하게 강화하는 효과가 있지만, 그 성공이 보상의 허점을 이용한 것이라면 그것도 강하게 강화된다. 그룹 합은 항상 $\sum A_i = 0$이다.

## 3. KL을 loss에 더하는 것과 보상에서 빼는 것의 차이

**보상에서 빼기**(PPO-RLHF 방식): $R' = R - \beta\sum_t\log\frac{\pi_\theta(y_t)}{\pi_{ref}(y_t)}$를 advantage에 넣는다. KL 항이 score function($\nabla\log\pi$)과 곱해지는 REINFORCE 형태로 미분되고, 이는 reverse KL $\mathrm{KL}(\pi_\theta\|\pi_{ref})$ gradient의 비편향 추정이다. 다만 GRPO처럼 advantage를 그룹 정규화하면 KL 항의 스케일도 함께 정규화된다.

**loss에 더하기**(GRPO 방식): $k_3$를 샘플된 토큰에서 직접 미분한다. $r = \pi_{ref}/\pi_\theta$로 두면

$$\nabla_\theta k_3 = \nabla_\theta(r - 1 - \log r) = (1 - r)\nabla_\theta\log\pi_\theta,$$

$$\mathbb E_{\pi_\theta}[(1 - r)\nabla\log\pi_\theta] = \underbrace{\mathbb E_{\pi_\theta}[\nabla\log\pi_\theta]}_{=0} - \sum_y\pi_{ref}\nabla\log\pi_\theta = \nabla_\theta\,\mathrm{KL}(\pi_{ref}\|\pi_\theta).$$

즉 $k_3$의 **값**은 reverse KL의 비편향 추정이지만, 샘플링 분포가 $\theta$에 의존한다는 점을 무시하고 미분한 **gradient**는 forward KL $\mathrm{KL}(\pi_{ref}\|\pi_\theta)$의 gradient이다(이 레포에서 수치로 확인함). forward KL은 $\pi_{ref}$가 확률을 주는 곳에 $\pi_\theta$도 확률을 주도록(mass-covering) 하므로, 정책이 참조의 일부 모드에만 집중하는 것을 막는 쪽으로 작용한다. 어느 쪽이 더 나은지는 연구가 진행 중이며, 같은 "KL penalty"라도 구현에 따라 최적화하는 대상이 다르다는 점을 알고 써야 한다.
