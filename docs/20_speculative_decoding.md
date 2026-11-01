# Step 20. Speculative Decoding

- 구현: `minillm/generate.py` — `speculative_accept`, `speculative_generate`
- 테스트: `pytest tests/test_20_speculative.py`
- 실행: `python scripts/pretrain.py --preset draft` 후 `python scripts/speculative.py`
- 선행: step 13, 14

## 1. 목표

작은 드래프트 모델이 여러 토큰을 제안하고 큰 타깃 모델이 한 번의 forward로 검증하는 speculative decoding을 구현한다. 출력 분포가 타깃 모델 단독 샘플링과 **정확히 같음**을 증명하고 테스트로 확인한다.

## 2. 이론

### 2.1 동기

decode 단계는 memory-bound여서(step 14) 토큰 1개를 처리하는 forward와 $\gamma + 1$개를 처리하는 forward의 시간이 비슷하다. 다음 $\gamma$개 토큰의 후보를 싸게 얻을 수 있다면, 타깃 모델 한 번의 forward로 여러 토큰을 확정할 수 있다.

### 2.2 알고리즘 (한 라운드)

1. 드래프트가 자기회귀로 $\tilde x_1, \dots, \tilde x_\gamma$를 샘플링하고 각 시점의 분포 $q_1, \dots, q_\gamma$를 기록한다.
2. 타깃이 접두사 + $\tilde x_{1:\gamma}$를 한 번에 처리해 $p_1, \dots, p_{\gamma+1}$을 얻는다($p_i$는 $\tilde x_i$ 자리의 분포, $p_{\gamma+1}$은 그 다음 자리).
3. $i = 1, \dots, \gamma$ 순서로 $\tilde x_i$를 확률 $\min\big(1, p_i(\tilde x_i)/q_i(\tilde x_i)\big)$로 수락한다.
4. 처음 거절된 위치 $i$에서는 잔여분포 $p'_i = \mathrm{norm}\big(\max(0, p_i - q_i)\big)$에서 하나를 뽑고 라운드를 끝낸다.
5. 모두 수락되면 $p_{\gamma+1}$에서 보너스 토큰을 하나 뽑는다.

한 라운드에서 최소 1개, 최대 $\gamma + 1$개의 토큰이 확정된다.

### 2.3 정확성 증명 (한 위치)

$\beta = \sum_x\min(p(x), q(x))$를 수락 확률이라 하자. 출력이 $x$일 확률은 "드래프트가 $x$를 제안하고 수락" + "거절 후 잔여분포에서 $x$"이다:

$$
P(\text{out} = x) = q(x)\min\!\left(1,\frac{p(x)}{q(x)}\right) + (1 - \beta)\,\frac{\max(0, p(x) - q(x))}{\sum_{x'}\max(0, p(x') - q(x'))}.
$$

첫 항은 $\min(p(x), q(x))$이다. 잔여분포의 정규화 상수는 $\sum_x\max(0, p - q) = \sum_x\big(p - \min(p,q)\big) = 1 - \beta$이므로 둘째 항은 $\max(0, p(x) - q(x))$이다. 따라서

$$P(\text{out} = x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x).$$

각 위치는 앞선 확정 토큰을 조건으로 같은 논리가 적용되므로, 귀납적으로 전체 시퀀스가 타깃 분포를 따른다.

**Greedy 특수 경우**: $\tau = 0$에서 $p, q$를 argmax의 one-hot으로 두면 수락 조건은 "드래프트 토큰 = 타깃 argmax"이고, 거절 시 잔여분포는 타깃 argmax의 one-hot이다. 따라서 출력은 타깃 단독 greedy와 토큰 단위로 동일하다(테스트로 확인).

### 2.4 기대 토큰 수와 속도

위치별 수락이 독립이고 수락률이 $\alpha$($= 1 - \mathrm{TV}(p, q)$의 평균)라고 하면, 한 라운드의 확정 토큰 수 기댓값은

$$\mathbb E[\#\text{tokens}] = \sum_{i=0}^{\gamma}\alpha^i = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.$$

예: $\alpha = 0.8$, $\gamma = 4$이면 $(1 - 0.8^5)/0.2 = 3.36$. 드래프트 forward 비용을 $c_d$, 타깃 forward 비용을 $c_t$라 하면 자기회귀 대비 속도비는

$$\text{speedup} = \frac{\mathbb E[\#\text{tokens}]\cdot c_t}{\gamma c_d + c_t}.$$

$c_d/c_t$가 작고 $\alpha$가 클수록 유리하다. 대형 타깃(예: 70B)과 소형 드래프트(예: 7B)에서는 $c_d/c_t$가 작아 2–3배 속도 향상이 보고된다.

### 2.5 이 레포에서의 기대치

12M 타깃과 1M 수준 드래프트는 둘 다 매우 작아서, forward 시간이 연산량보다 커널 실행·Python 오버헤드에 좌우된다. 그러면 $c_d/c_t$가 레이어 수 비(2/6) 정도로 크고, 수락 판정 루프의 Python 비용도 더해져 실제 tok/s는 오히려 낮을 수 있다. 그래서 `speculative.py`는 하드웨어와 무관한 지표인 **타깃 forward 횟수**와 **라운드당 평균 확정 토큰 수**를 함께 출력한다. 이 값들이 2.4의 이론과 맞는지 확인하는 것이 이 단계의 핵심이다.

### 2.6 KV cache와 rollback

두 모델 모두 cache를 쓰고 다음 불변식을 유지한다: **cache에는 마지막 확정 토큰을 제외한 확정 토큰들만 들어 있다**(`pos ≤ len(seq) − 1`).

- 라운드 시작 시 각 모델에 `seq[cache.pos:]`(cache에 아직 없는 확정 토큰들, 보통 1–2개)를 먼저 넣는다.
- 타깃은 `seq[pos:] + drafts`를 한 번에 넣고 마지막 $\gamma + 1$개 위치의 분포를 쓴다.
- 확정 후 `pos = min(pos, len(seq) − 1)`로 되돌린다. 거절된 드래프트 토큰의 K/V는 이후 덮어쓰이므로 지울 필요가 없다(step 14의 사전할당 버퍼 덕분에 $O(1)$).
- 모든 드래프트가 수락되면 드래프트 cache에는 마지막 드래프트 토큰이 없으므로(입력한 적이 없음) 다음 라운드에 `seq[pos:]`로 2개가 들어간다. 위 규칙이 이 경우를 자동으로 처리한다.

## 3. 구현 가이드

- `speculative_accept(draft_tokens, q, p, generator)`: `torch.rand((), generator=generator) < p[i, t] / q[i, t]`로 수락 판정. 거절 시 `(p[i] − q[i]).clamp(min=0)`을 정규화해 `torch.multinomial`. 반환은 `list[int]`.
- `speculative_generate`: 2.6의 불변식을 지키도록 작성. `_probs(logits, temperature)`(제공됨)는 $\tau = 0$이면 one-hot, 아니면 softmax를 반환한다.
- 배치 크기 1만 지원한다(시퀀스마다 수락 길이가 달라지므로 배치화에는 추가 처리가 필요).

## 4. 검증

- 분포 보존: 고정된 $p, q$로 2만 번 실행해 첫 출력 토큰의 경험분포가 $p_1$과 일치하는지.
- $p = q$이면 항상 $\gamma + 1$개 확정.
- greedy speculative 출력 = 타깃 단독 greedy 출력.

## 5. 확인 문제

1. 드래프트와 타깃의 temperature가 다르면 정확성 증명의 어느 부분이 깨지는가? (힌트: $q$는 실제로 드래프트가 샘플링한 분포여야 한다.)
2. $c_d/c_t = 0.1$, $\alpha = 0.7$일 때 speedup을 최대화하는 $\gamma$를 구하라.
3. 드래프트 모델 없이 타깃 모델에 여러 개의 예측 head를 붙이는 Medusa, 또는 n-gram 기반 후보를 쓰는 방법과 비교해 장단점을 정리하라.

## 참고문헌

- Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding," ICML 2023.
- C. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," arXiv:2302.01318, 2023.
- M. Stern, N. Shazeer, J. Uszkoreit, "Blockwise Parallel Decoding for Deep Autoregressive Models," NeurIPS 2018.
- T. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," ICML 2024.
- Y. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," ICML 2024.
