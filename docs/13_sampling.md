# Step 13. 디코딩: temperature, top-k, nucleus sampling

- 구현: `minillm/generate.py` — `filter_probs`, `sample_next`, `generate`의 샘플링 구간
- 테스트: `pytest tests/test_13_sampling.py`
- 선행: step 10

## 1. 목표

다음 토큰 분포에서 토큰을 고르는 전략들을 구현하고, 자기회귀 생성 루프를 완성한다(KV cache 없는 버전).

## 2. 이론

### 2.1 탐색 vs 샘플링

- **Greedy**: $\arg\max_i p_i$. 결정적이며, 개방형 생성에서는 같은 구절을 반복하는 퇴화가 잘 일어난다.
- **Beam search**: 높은 우도의 시퀀스를 찾는다. 번역처럼 정답이 좁은 과제에 적합하지만, 개방형 생성에서는 사람이 쓴 텍스트보다 우도가 지나치게 높은 단조로운 텍스트를 낸다(Holtzman et al. 2020).
- **Sampling**: 분포에서 추출한다. 꼬리의 저확률 토큰이 누적되면 문맥에서 벗어나므로 분포를 잘라내는 기법과 함께 쓴다.

### 2.2 Temperature

$$p_i^{(\tau)} = \frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}$$

$\tau \to 0$이면 argmax로 수렴하고 $\tau \to \infty$이면 균등분포로 수렴한다. 엔트로피는 $\tau$에 대해 단조 증가한다. $\tau = 0$은 0 나눗셈이 되므로 greedy로 특수 처리한다.

### 2.3 Top-k (Fan et al. 2018)

logit 상위 $k$개만 남기고 나머지를 $-\infty$로 둔 뒤 softmax. 분포가 뾰족할 때도 평평할 때도 같은 $k$를 쓴다는 한계가 있다.

### 2.4 Nucleus / top-p (Holtzman et al. 2020)

확률 내림차순으로 정렬했을 때 누적합이 $p$ 이상이 되는 **최소** 집합 $V_p$만 남기고 재정규화한다:

$$V_p = \arg\min_{V'} |V'| \quad \text{s.t.}\quad \sum_{i \in V'} p_i \ge p.$$

분포가 뾰족하면 후보가 적고, 평평하면 많아져 분포 모양에 적응한다.

구현: 정렬된 확률 $\tilde p$와 누적합 $c$에 대해, **자기 자신을 제외한** 앞선 누적합 $c_i - \tilde p_i$가 이미 $p$ 이상인 토큰을 제거한다. 그러면 임계값을 처음 넘게 만드는 토큰까지 포함된다. 예: $[0.5, 0.3, 0.15, 0.05]$, $p = 0.7$ → 앞선 누적합 $[0, 0.5, 0.8, 0.95]$ → 앞의 두 개 유지 → $[0.625, 0.375, 0, 0]$.

### 2.5 적용 순서

`logits / τ` → top-k(logit 공간) → softmax → top-p(확률 공간) → 재정규화 → `torch.multinomial`.

### 2.6 생성 루프와 비용

매 스텝 지금까지의 시퀀스(최근 `block_size`개)를 모델에 넣어 마지막 위치의 logits로 다음 토큰을 뽑는다. 길이 $t$의 시퀀스를 매번 다시 계산하므로 $n$개 토큰 생성 비용은 $O(\sum_t t) = O(n^2)$ 토큰-forward이다. step 14의 KV cache가 이를 $O(n)$으로 줄인다.

EOS 처리: 배치 중 EOS를 낸 시퀀스는 이후 EOS로 채우고(`done` 마스크), 모두 끝나면 조기 종료한다(제공됨).

## 3. 구현 가이드

- `filter_probs`: top-k는 `logits.topk(k).values[..., -1:]`(k번째 값)보다 작은 logit을 $-\infty$로. top-p는 `sort(descending=True)` → `cumsum` → 마스크 → `scatter`로 원래 순서 복원 → 재정규화. softmax는 step 02 구현을 쓴다.
- `sample_next`: `temperature == 0`이면 `argmax`, 아니면 `torch.multinomial(probs, 1, generator=generator)`.
- `generate`의 마커 구간: cache가 없으면 `idx[:, -block_size:]`를 입력, 있으면 `inp`(제공되는 루프가 첫 스텝에 프롬프트 전체, 이후 직전 토큰을 넣어 준다)를 입력한다.

## 4. 검증

greedy, top-k 비율 보존, top-p 경계 사례, 20000회 샘플링 빈도가 softmax 확률과 일치하는지, 생성 shape과 EOS 조기 종료를 확인한다.

## 5. 확인 문제

> 해설: [answers/13_sampling.md](answers/13_sampling.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. top-p = 1, top-k = None, τ = 1이면 어떤 분포에서 샘플링하는가?
2. τ < 1과 top-p를 함께 쓸 때 순서를 바꾸면(top-p 먼저) 결과가 달라지는가?
3. repetition penalty(이미 나온 토큰의 logit을 낮춤)를 `filter_probs` 앞단에 추가한다면 어떤 형태가 되는가?

## 참고문헌

- A. Holtzman et al., "The Curious Case of Neural Text Degeneration," ICLR 2020.
- A. Fan, M. Lewis, Y. Dauphin, "Hierarchical Neural Story Generation," ACL 2018.
- D. H. Ackley, G. E. Hinton, T. J. Sejnowski, "A Learning Algorithm for Boltzmann Machines," Cognitive Science 1985 (temperature).
