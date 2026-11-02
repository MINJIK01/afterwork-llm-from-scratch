# Step 07 확인 문제 해설

[← 본문](../07_rope.md)

## 1. $v$에도 RoPE를 적용하면?

attention 가중치 $\alpha_{ij}$는 $q, k$로만 계산되므로 그대로 상대 위치에만 의존한다. 출력은 $o_i = \sum_j \alpha_{ij} R_j v_j$가 되어, 각 value가 **자신의 절대 위치** $j$로 회전된 채 섞인다. 출력 쪽에서 $R_i^{-1}$을 곱하지 않으면 $R_j$는 상쇄되지 않으므로, 같은 상대 배치라도 절대 위치에 따라 출력이 달라진다. 즉 출력의 상대 위치 성질이 깨진다. (출력에 $R_i^{-1}$을 곱하면 $\sum_j\alpha_{ij}R_{j-i}v_j$가 되어 상대 위치 형태가 되며, 이런 변형을 시도한 연구도 있다.)

## 2. base를 키우면? 문맥 확장 기법과의 관계

가장 느린 주파수는 $\theta_{d/2-1} = \mathrm{base}^{-(d-2)/d} \approx 1/\mathrm{base}$이므로 파장 $2\pi/\theta \approx 2\pi\cdot\mathrm{base}$가 base에 비례해 길어진다.

학습 길이 $L$을 넘는 위치에서는 저주파 차원의 각도 $m\theta_i$가 학습 중 본 적 없는 범위에 들어간다(고주파 차원은 이미 여러 주기를 돌았으므로 문제가 적다). 

- Position Interpolation: $m$을 $m \cdot L/L'$로 줄여 모든 각도를 학습 범위 안으로 압축한다. 고주파 차원의 해상도도 같이 줄어드는 단점이 있다.
- NTK-aware scaling: base를 키워 저주파 차원은 많이 늘리고 고주파 차원은 거의 그대로 두어, 외삽이 필요한 저주파 차원만 학습 범위 안으로 들어오게 한다.
- YaRN: 주파수 대역별로 보간 정도를 다르게 하고 attention temperature를 보정한다.

## 3. Sinusoidal 절대 위치를 더할 때 $q_m^\top k_n$의 전개

$q_m = W_Q(x_m + p_m)$, $k_n = W_K(x_n + p_n)$, $A = W_Q^\top W_K$:

$$q_m^\top k_n = x_m^\top A x_n + x_m^\top A p_n + p_m^\top A x_n + p_m^\top A p_n.$$

- $x_m^\top A x_n$: 내용끼리의 항(위치 무관).
- $x_m^\top A p_n$, $p_m^\top A x_n$: 내용과 **절대 위치**의 교차 항. 상대 위치만의 함수가 아니다.
- $p_m^\top A p_n$: $A$가 단위행렬이면 sinusoidal의 성질상 $\sum_i\cos((m-n)\omega_i)$로 상대 위치 함수지만, 학습된 일반 $A$에서는 그렇지 않다.

RoPE는 회전을 곱하는 방식이라 이런 교차 항 없이 $q^\top R_{n-m}k$ 하나로 정리된다. Transformer-XL과 T5의 상대 위치 방식은 교차 항을 직접 제거하거나 bias로 대체한 것이다.
