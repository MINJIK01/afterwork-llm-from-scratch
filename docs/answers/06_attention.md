# Step 06 확인 문제 해설

[← 본문](../06_attention.md)

## 1. fp16에서 $-\infty$ 대신 $-10^4$, 그리고 $-\infty$의 주의점

**큰 음수(-1e4 등)를 쓸 때**
- `-1e9`처럼 fp16 범위(±65504) 밖의 값은 변환 시 $-\infty$가 되어 의도와 달라진다. fp16에서 쓸 수 있는 값은 $-10^4$ 수준이다.
- 가려야 할 위치의 가중치가 정확히 0이 아니라 $e^{-10^4 - m}$이다. 실제로는 0으로 underflow하므로 대부분 문제없다.
- 한 행이 **전부** 가려지면 모든 logit이 같은 $-10^4$이 되어 균등분포가 나온다. NaN 대신 padding 위치를 조용히 균등하게 참조하게 되어, 오류가 드러나지 않는다.

**$-\infty$를 쓸 때**
- 전부 가려진 행에서 max-subtraction이 $-\infty - (-\infty) = \mathrm{NaN}$을 만들고, NaN이 gradient를 통해 전체로 퍼진다. padding이 있는 배치에서는 전부 가려진 query 행이 생길 수 있으므로 해당 행을 0으로 처리하는 등의 보호가 필요하다.
- causal mask는 대각선이 항상 허용되므로 이 문제가 없다.

## 2. $d_k$를 키울 때 vs 헤드 수를 늘릴 때

- $d_k$ 증가(모델 폭 고정 시 head 수 감소, 또는 폭 자체 증가): score 계산 $O(T^2 d_k)$와 투영 파라미터가 늘어난다. head 하나의 logit 행렬 rank 상한($\le d_k$)이 커진다.
- $d$ 고정, head 수 $H$ 증가: $d_h = d/H$가 줄어 총 FLOPs와 파라미터는 같고, 서로 다른 attention 패턴의 수가 늘어난다. 대신 head당 rank가 낮아진다. $d_h$가 너무 작으면(예: 16 이하) 각 head의 표현력이 부족해 성능이 떨어진다는 분석이 있다(Bhojanapalli et al. 2020, "Low-Rank Bottleneck in Multi-head Attention Models"). 대부분의 LLM이 $d_h = 64$–$128$을 쓴다.

## 3. Online softmax가 score 행렬 없이 정확한 이유

running 최댓값 $m$, 정규화합 $l = \sum e^{s_j - m}$, 가중합 $a = \sum e^{s_j - m}v_j$를 유지하고, 새 블록에서 최댓값이 커지면 기존 $l$, $a$에 $e^{m_{old} - m_{new}}$를 곱해 기준을 맞춘다. 지수함수의 곱셈 성질($e^{s - m_{old}}e^{m_{old} - m_{new}} = e^{s - m_{new}}$) 때문에 이 보정은 정확하고, 모든 블록을 본 뒤 $a/l$은 전체 softmax 가중합과 같다. 상태 크기가 key 수와 무관한 $O(d)$이므로 score 행렬이 필요 없다. 유도와 구현은 step 21에서 다룬다.
