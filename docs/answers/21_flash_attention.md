# Step 21 확인 문제 해설

[← 본문](../21_flash_attention.md)

## 1. 두 부분 상태의 병합과 Flash-Decoding

key 집합 $J_1$, $J_2$에 대한 부분 상태 $(m_k, l_k, a_k)$ (정규화하지 않은 가중합 $a_k = \sum_{J_k}e^{s_j - m_k}v_j$):

$$m = \max(m_1, m_2),\quad l = l_1e^{m_1 - m} + l_2e^{m_2 - m},\quad a = a_1e^{m_1 - m} + a_2e^{m_2 - m},\quad o = a/l.$$

이 연산은 결합적·교환적이므로 key 축을 임의로 나눠 병렬 계산한 뒤 트리 형태로 합칠 수 있다. decode 단계는 query가 1개라 query 축 병렬성이 없고, 긴 문맥에서는 한 스레드 블록이 모든 key를 순회하는 것이 병목이다. Flash-Decoding은 KV를 여러 조각으로 나눠 각 스레드 블록이 부분 상태를 계산하고, 마지막에 위 식으로 병합한다. 출력과 함께 LSE($m + \log l$)를 저장하는 이유도 이 병합에 있다.

## 2. $T = 4096$, $d = 64$, fp16에서의 크기

- 표준 구현의 $S$ (head 하나, 배치 하나): $4096^2\times2$ B $= 32$ MiB. $P$도 같은 크기이고, 배치 × head 수만큼 곱해진다.
- SRAM 100 KB에 $Q$ 블록 $B_r\times d$, $K$/$V$ 블록 각 $B_c\times d$ (fp16), score 블록 $B_r\times B_c$, 출력 누적 $B_r\times d$ (fp32)를 둔다고 하자. $B_r = B_c = B$이면 대략 $128B + 256B + 256B + 4B^2 \le 10^5$ → $B \lesssim 97$. 2의 거듭제곱으로 $B = 64$가 현실적이다.

FlashAttention 논문의 규칙 $B_c = \lceil M/(4d)\rceil$, $B_r = \min(B_c, d)$에 $M = 100\text{KB}/2\text{B} = 5\times10^4$ 원소를 넣으면 $B_c \approx 195$, $B_r = 64$이다. 실제 커널은 레지스터·shared memory 제약에 맞춰 64–128을 쓴다.

## 3. Backward 재계산이 FLOPs를 늘리는데도 빠른 이유

표준 backward는 forward에서 저장한 $P$($T^2$ 크기)를 HBM에서 다시 읽고, 중간 gradient($dP$, $dS$)도 HBM에 쓰고 읽는다. attention의 softmax·마스킹 부분은 산술 강도가 낮아 HBM 대역폭이 시간을 결정한다. FlashAttention backward는 저장된 LSE로 $P$를 블록마다 SRAM 안에서 다시 계산하므로, 행렬곱 FLOPs는 늘지만 $O(T^2)$의 HBM 읽기·쓰기가 사라진다. on-chip 연산은 HBM 접근보다 훨씬 싸서 전체 시간이 줄어든다. 메모리도 $O(T^2)$에서 $O(T)$로 줄어든다. step 23의 gradient checkpointing과 같은 "계산으로 메모리 접근을 대체"하는 원리이다.
