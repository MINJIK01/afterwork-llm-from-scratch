# Step 10 확인 문제 해설

[← 본문](../10_gpt.md)

## 1. Tying 시 $E$의 gradient

$E \in \mathbb R^{V\times d}$가 두 곳에 쓰인다.

1. **입력 경로**: $h^{(0)}_t = E[x_t]$. 입력에 등장한 토큰의 행에만 해당 위치의 gradient가 더해진다(scatter-add).
2. **출력 경로**: $\mathrm{logits} = HE^\top$ ($H$: final norm 출력, $(BT, d)$). $\partial L/\partial E = (\partial L/\partial\,\mathrm{logits})^\top H$, shape $(V, d)$. softmax gradient($p - y$)는 모든 어휘에 대해 0이 아니므로 **모든 행**이 매 스텝 gradient를 받는다.

전체 gradient는 두 경로의 합이다. 입력에 거의 나오지 않는 드문 토큰도 출력 경로로 계속 학습된다.

## 2. $L = 24$일 때 residual 출력 투영의 초기 표준편차

$0.02/\sqrt{2\times24} = 0.02/\sqrt{48} \approx 0.00289$.

스케일링을 하지 않으면 각 서브레이어 기여의 분산이 $48$배($= 2L$배) 크므로, residual stream에 쌓이는 기여의 총분산 $2L\sigma_f^2$가 스케일링했을 때($\approx \sigma^2$ 수준)보다 48배 커진다. 초기 residual stream의 크기가 깊이에 비례해 커지고 출력 분포가 초기부터 날카로워져 학습 초반이 불안정해진다.

## 3. m5 프리셋에서 attention 항의 비율

$N = 12{,}194{,}688$, $6N \approx 73.2$M FLOPs/토큰. attention 항 $6LTd = 6\times6\times256\times384 \approx 3.54$M.

$$\frac{3.54}{73.2 + 3.54} \approx 4.6\%.$$

Kaplan et al.처럼 임베딩을 뺀 $N$(10.62M)을 쓰면 $3.54/(63.7 + 3.54) \approx 5.3\%$. 이 규모($T = 256$, $d = 384$)에서는 행렬곱이 지배적이고, $T$가 $d$보다 훨씬 커지는 긴 문맥에서 attention 비중이 커진다.
