# Step 21. Online Softmax와 FlashAttention 방식의 블록 Attention

- 구현: `minillm/flash.py` — `online_logsumexp`, `flash_attention`
- 테스트: `pytest tests/test_21_flash.py`
- 선행: step 02, 06 (step 06 직후에 진행해도 된다)

## 1. 목표

$T_q \times T_k$ score 행렬 전체를 만들지 않고, key 축을 블록 단위로 한 번만 훑으면서 정확한 attention 출력을 계산한다. FlashAttention의 핵심인 online softmax 갱신식을 유도하고 구현한다.

## 2. 이론

### 2.1 표준 attention의 메모리 병목

step 06 구현은 $S = QK^\top/\sqrt d$ ($T_q \times T_k$)와 $P = \mathrm{softmax}(S)$를 메모리에 만든다. GPU는 용량이 큰 HBM과 작고 빠른 on-chip SRAM으로 나뉘는데, $S$와 $P$를 HBM에 쓰고 다시 읽는 비용이 $O(T^2)$이다. attention의 행렬곱 FLOPs는 많지만 softmax·마스킹 같은 원소별 연산이 HBM 접근에 묶여 있어서, 긴 시퀀스에서 attention은 계산량보다 메모리 접근(IO)이 병목이 된다(Dao et al. 2022).

### 2.2 Online softmax: 한 번에 훑는 log-sum-exp

수치적으로 안정된 softmax는 최댓값 $m$을 알아야 하므로 순진하게는 두 번 훑어야 한다(최댓값, 그다음 합). Milakov & Gimelshein(2018)은 running 상태 $(m, s)$를 유지하면 한 번에 계산할 수 있음을 보였다. 지금까지 본 원소 집합을 $J$라 할 때 불변식은

$$m = \max_{j \in J} x_j,\qquad s = \sum_{j \in J} e^{x_j - m}.$$

새 chunk $C$가 들어오면

$$m' = \max\big(m, \max_{j\in C} x_j\big),\qquad s' = s\,e^{m - m'} + \sum_{j \in C} e^{x_j - m'}.$$

첫 항은 기존 합을 새 기준 $m'$으로 다시 맞춘 것이다($e^{x_j - m} e^{m - m'} = e^{x_j - m'}$). 마지막에 $\mathrm{LSE} = m + \log s$. 초기값 $m = -\infty$, $s = 0$이면 첫 chunk에서 $e^{-\infty - m'} = 0$이 되어 그대로 동작한다.

### 2.3 Attention 출력으로 확장

query 하나에 대해, 정규화하지 않은 누적 출력 $a = \sum_{j \in J} e^{s_j - m} v_j$와 $l = \sum_{j\in J} e^{s_j - m}$를 유지한다. key 블록 $C$의 score $s_C$가 들어오면

$$m' = \max(m, \max s_C),\quad c = e^{m - m'},\quad P = e^{s_C - m'},$$
$$l' = c\,l + \textstyle\sum P,\qquad a' = c\,a + P V_C.$$

모든 블록을 본 뒤 출력은 $o = a / l$이다. 불변식 $a = \sum_J e^{s_j - m}v_j$, $l = \sum_J e^{s_j - m}$가 매 단계 유지되므로 $a/l = \sum_j \mathrm{softmax}(s)_j v_j$로 정확히 같다(근사가 아니다).

### 2.4 Tiling과 IO 복잡도

FlashAttention은 $Q$도 블록으로 나눠, $Q$ 블록 하나를 SRAM에 올린 채 $K, V$ 블록을 순회하며 위 갱신을 SRAM 안에서 수행한다. $S$, $P$를 HBM에 쓰지 않으므로 HBM 접근은 SRAM 크기 $M$에 대해 $O(T^2 d^2 / M)$으로, 표준 구현의 $\Theta(Td + T^2)$보다 작다. 메모리 사용량은 $O(T)$(출력과 행별 통계)이다.

backward에서는 $P$를 저장하지 않고, forward에서 저장한 행별 $\mathrm{LSE} = m + \log l$로 $P = \exp(S - \mathrm{LSE})$를 블록마다 다시 계산한다. `online_logsumexp`가 계산하는 값이 바로 이 통계다.

### 2.5 Causal 마스크와 블록 건너뛰기

causal이면 query $i$(절대 위치 $T_k - T_q + i$)보다 뒤에 있는 key는 $-\infty$로 가린다(step 08의 오프셋 규약과 같다). 첫 key 블록에는 항상 key 0이 있고 모든 query가 이를 참조할 수 있으므로, 첫 블록 이후 $m$은 유한하다. 이후 블록에서 어떤 행이 전부 가려져도 $P = 0$, $c = 1$이 되어 상태가 그대로 유지된다. 실제 커널은 전부 가려지는 블록을 아예 계산하지 않아 연산량을 약 절반으로 줄인다.

### 2.6 이 구현의 의미와 한계

PyTorch 수준의 Python 루프는 SRAM을 직접 제어하지 못하므로 각 블록 연산이 여전히 HBM(또는 통합 메모리)을 거친다. 속도 이득은 없고, 얻는 것은 (1) 알고리즘의 정확한 이해와 (2) score 메모리가 $O(T_q T_k)$에서 $O(T_q \times \text{block})$으로 줄어드는 것이다. 실제 이득은 Triton/CUDA/Metal 커널로 구현했을 때 나온다. FlashAttention-2는 루프 순서를 바꾸고 비-행렬곱 연산을 줄여 GPU 활용도를 더 높였다.

## 3. 구현 가이드

- `online_logsumexp`: 2.2의 갱신식. `torch.maximum(m, c.amax(-1))`, `m_new.unsqueeze(-1)`로 broadcast.
- `flash_attention`: 상태 `m`, `l`은 `(..., Tq, 1)`, `acc`는 `(..., Tq, dv)`. 블록마다 score → (causal이면) 마스크 → 2.3의 갱신. torch 내장 softmax는 테스트에서 금지된다.
- 마지막 블록은 `block_size`보다 짧을 수 있다(`kb.shape[-2]` 사용).

## 4. 검증

`torch.logsumexp`와 비교(값 ±50 범위, 다양한 chunk 크기), `F.scaled_dot_product_attention`과 causal/non-causal 비교(block 1, 7, 64), $T_q < T_k$(KV cache 상황)에서 오프셋 마스크를 확인한다.

## 5. 확인 문제

1. 두 부분 상태 $(m_1, l_1, a_1)$, $(m_2, l_2, a_2)$를 합치는 식을 쓰라. 이 식이 key 축을 여러 GPU/스레드 블록에 나눠 병렬 계산할 때(Flash-Decoding) 어떻게 쓰이는가?
2. $T = 4096$, $d = 64$, fp16일 때 표준 구현의 $S$ 행렬 크기(head 하나)와, SRAM 100KB에 $Q$ 블록과 $K, V$ 블록을 올릴 때 가능한 블록 크기를 추정하라.
3. backward에서 $P$를 저장하는 대신 다시 계산하면 FLOPs는 늘어나는데도 전체 시간이 줄어드는 이유를 설명하라.

## 참고문헌

- M. Milakov, N. Gimelshein, "Online normalizer calculation for softmax," arXiv:1805.02867, 2018.
- M. N. Rabe, C. Staats, "Self-attention Does Not Need O(n²) Memory," arXiv:2112.05682, 2021.
- T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022.
- T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," ICLR 2024.
