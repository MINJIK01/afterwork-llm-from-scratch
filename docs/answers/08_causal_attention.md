# Step 08 확인 문제 해설

[← 본문](../08_causal_attention.md)

## 1. Causal mask 없이 학습하면?

위치 $t$가 정답 $x_{t+1}$을 attention으로 직접 볼 수 있으므로, 모델은 "다음 위치의 토큰을 복사"하는 해를 쉽게 찾아 학습 loss가 0에 가깝게 떨어진다. 생성 시에는 미래 토큰이 존재하지 않으므로(아직 생성되지 않음) 학습 때 의존하던 정보가 없어, 출력이 무의미해진다. 학습과 추론의 입력 분포가 다른 전형적인 정보 누수(leakage)이다. 학습 loss가 비정상적으로 낮으면 마스크를 먼저 의심해야 한다.

## 2. MQA/GQA의 파라미터 수와 KV cache

KV head 수 $G$, head 차원 $d_h$, $d = Hd_h$:

| | 파라미터 (레이어당) | KV cache (토큰·레이어당 원소 수) |
|---|---|---|
| MHA ($G = H$) | $4d^2$ | $2Hd_h = 2d$ |
| GQA | $2d^2 + 2dGd_h$ | $2Gd_h$ |
| MQA ($G = 1$) | $2d^2 + 2dd_h$ | $2d_h$ |

($W_Q$, $W_O$ 각 $d^2$, $W_K$, $W_V$ 각 $d \times Gd_h$.) 구현은 step 22, 수치 예시는 step 22 해설 참고.

## 3. diagonal 오프셋을 0으로 두면?

decode 단계는 $T = 1$, $T_k$ = 지금까지의 길이이다. `torch.ones(1, Tk).tril(diagonal=0)`은 첫 행에서 $j \le 0$만 True이므로 새 토큰이 **key 0(첫 토큰)만** 참조한다. 올바른 마스크는 `tril(diagonal=Tk-1)`로 모든 과거 key를 허용해야 한다. 오류가 나지 않고 조용히 틀린 결과를 내므로, step 14의 "증분 logits = 전체 logits" 테스트로만 잡힌다. 그 테스트가 왜 중요한지 보여 주는 사례다.
