# Step 14. KV Cache와 추론 비용 분석

- 구현: `minillm/layers.py` — `KVCache.update`, `kv_cache_bytes`
- 테스트: `pytest tests/test_14_kv_cache.py`
- 실행: `python scripts/bench_inference.py` (cache 유무 처리량 비교)
- 선행: step 08, 13

## 1. 목표

이미 계산한 key/value를 재사용하는 KV cache를 구현해, 증분 생성의 logits가 전체 재계산과 정확히 같음을 확인하고 메모리 비용을 정량화한다.

## 2. 이론

### 2.1 왜 재사용이 정확한가

causal 모델에서 레이어 $l$, 위치 $j$의 표현은 $x_{\le j}$에만 의존한다(step 08). 따라서 토큰을 뒤에 추가해도 기존 위치의 $K_j^{(l)}, V_j^{(l)}$는 바뀌지 않는다. 새 토큰의 query만 계산해 저장된 K, V 전체에 attention하면 전체 재계산과 수학적으로 동일한 결과를 얻는다(근사가 아니다). RoPE는 절대 위치 $m$으로 회전하므로 새 토큰은 cache 위치(`pos`)에 해당하는 각도를 써야 한다(`GPT.hidden`이 처리).

### 2.2 Prefill과 decode

- **Prefill**: 프롬프트 $T_0$개를 한 번에 처리해 cache를 채운다. 큰 행렬곱이라 계산량 위주(compute-bound).
- **Decode**: 토큰을 1개씩 처리한다. 토큰 하나당 모든 가중치($N$개 파라미터)와 KV cache를 메모리에서 읽지만 FLOPs는 $2N$ 정도라, 산술 강도(FLOPs/byte)가 fp32 기준 약 0.5로 매우 낮다. 따라서 메모리 대역폭이 속도를 결정한다(memory-bound):
$$\text{tokens/s} \lesssim \frac{\text{메모리 대역폭}}{\text{가중치 바이트} + \text{KV cache 바이트}}.$$
이 관계가 양자화(step 19)와 speculative decoding(step 20)의 동기이다.

### 2.3 비용 비교 (토큰 $n$개 생성, 프롬프트 무시)

| | cache 없음 | cache 있음 |
|---|---|---|
| 스텝 $t$의 MLP·투영 계산 | $O(tN)$ | $O(N)$ |
| 스텝 $t$의 attention | $O(t^2 d L)$ | $O(t d L)$ |
| 총합 | $O(n^2 N)$ | $O(nN + n^2 dL)$ |

### 2.4 메모리

$$\text{KV bytes} = 2 \cdot L \cdot B \cdot H \cdot d_h \cdot T \cdot \text{bytes/elem}$$

(2 = K와 V). LLaMA-2-7B($L = 32$, $H = 32$, $d_h = 128$), fp16, $T = 4096$이면 시퀀스 하나당 $2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2$ GiB, 배치 8이면 16 GiB로 가중치(13 GiB)보다 커진다. 이를 줄이는 방법:

- MQA/GQA: K/V head 수를 $H \to G$로 줄여 cache를 $G/H$배로(step 08 확인 문제).
- PagedAttention(vLLM): cache를 고정 크기 블록으로 관리해 단편화를 줄이고 공유를 가능하게 한다.
- KV cache 양자화.

### 2.5 구현 방식: 사전 할당 버퍼

매 스텝 `torch.cat`으로 늘리면 매번 전체를 복사한다. `(B, H, max_len, d_h)` 버퍼를 미리 할당하고 `pos`부터 써 넣으면 복사가 새 토큰 분량뿐이다. 또한 `pos`를 줄이기만 하면 이후 토큰을 "되돌릴" 수 있어 speculative decoding(step 20)의 rollback이 $O(1)$이 된다.

## 3. 예제로 따라가기

### 3.1 cache가 채워지는 과정

프롬프트 3토큰 `[a, b, c]` 뒤에 2토큰을 생성한다. 레이어 하나, head 하나만 본다.

| 스텝 | 모델 입력 (T) | 이번에 계산하는 K, V | 호출 전 `pos` | 호출 후 cache 내용 | attention 대상 key | 마스크 diagonal ($T_k - T$) |
|---|---|---|---|---|---|---|
| prefill | `[a, b, c]` (3) | $k_a, k_b, k_c$ | 0 | $k_a\ k_b\ k_c$ | a, b, c (causal) | 0 |
| decode 1 | `[d]` (1) | $k_d$ | 3 | $k_a\ k_b\ k_c\ k_d$ | a, b, c, d | 3 |
| decode 2 | `[e]` (1) | $k_e$ | 4 | $k_a \ldots k_e$ | a … e | 4 |

```
사전할당 버퍼 (max_len = 8)      pos
prefill 후   [ka kb kc  .  .  .  .  .]   3
decode 1 후  [ka kb kc kd  .  .  .  .]   4
decode 2 후  [ka kb kc kd ke  .  .  .]   5
update()가 반환하는 것 = 버퍼[:, :, :pos]
```

decode 단계의 RoPE 각도도 `pos` 기준이다. `d`는 위치 3, `e`는 위치 4로 회전되어야 cache 없이 전체를 계산한 결과와 같아진다(`GPT.hidden`이 `rope_cos[start:start+T]`로 처리).

### 3.2 계산량 비교 (256토큰 생성)

| | 모델에 넣는 토큰 수 합계 |
|---|---|
| cache 없음: 매 스텝 전체 재계산 | $1 + 2 + \cdots + 256 = 32{,}896$ |
| cache 있음: 매 스텝 새 토큰 1개 | 256 |

MLP와 투영 계산이 약 128배 줄어든다. 단, attention은 매 스텝 과거 key 전체를 보므로 여전히 길이에 비례한다.

### 3.3 메모리를 숫자로

$2 \cdot L \cdot B \cdot H_{kv} \cdot d_h \cdot T \cdot \text{bytes}$:

| 설정 | $L$ | $H_{kv}$ | $d_h$ | $B$ | $T$ | dtype | KV cache |
|---|---|---|---|---|---|---|---|
| m5 프리셋 | 6 | 6 | 64 | 1 | 256 | fp32 | 4.5 MiB |
| m5 프리셋, 배치 32 | 6 | 6 | 64 | 32 | 256 | fp32 | 144 MiB |
| LLaMA-2-7B | 32 | 32 | 128 | 1 | 4096 | fp16 | 2,048 MiB |
| 같은 모델, GQA ($H_{kv}$=8) | 32 | 8 | 128 | 1 | 4096 | fp16 | 512 MiB |

m5 모델 가중치는 46.5 MiB이므로 배치 32에서는 cache가 가중치의 3배가 된다.

### 3.4 decode가 memory-bound인 이유를 숫자로

m5 모델, 배치 1로 토큰 하나를 만들 때:

| | 값 |
|---|---|
| 읽어야 하는 가중치 | 12.2M × 4 B = 48.8 MB |
| 연산량 | 약 $2N$ = 24.4 MFLOPs |
| 산술 강도 | 24.4M / 48.8M = 0.5 FLOP/byte |

GPU의 연산/대역폭 비율은 보통 수십 FLOP/byte 이상이므로, 이 상태에서는 연산 유닛이 대부분 메모리를 기다린다. 배치를 키우면 같은 가중치 읽기로 여러 토큰을 처리해 산술 강도가 배치 크기만큼 올라간다.

### 3.5 확인 코드

```python
import torch
from minillm.model import GPT, GPTConfig

m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, ffn_hidden=64))
idx = torch.randint(0, 64, (1, 6))
full, _ = m(idx)
caches = m.new_caches(1)
step = [m(idx[:, :3], caches=caches)[0]] + [m(idx[:, t:t + 1], caches=caches)[0] for t in range(3, 6)]
print(torch.allclose(torch.cat(step, 1), full, atol=1e-5), caches[0].pos)   # True 6
```

## 4. 구현 가이드

- `update(k, v)`: `self.k[:, :, pos:pos+T] = k`, 같은 방식으로 v, `pos += T`, `self.k[:, :, :pos]`와 `self.v[:, :, :pos]`를 반환.
- 마스크의 diagonal 오프셋(step 08)이 올바르면 attention 쪽은 수정할 필요가 없다.
- `kv_cache_bytes`: 2.4의 공식.

## 5. 검증

프롬프트 5토큰 prefill 후 1토큰씩 decode한 logits를 전체 forward logits와 비교하고, greedy 생성 결과가 cache 유무와 무관하게 같은지 확인한다. `bench_inference.py`는 두 방식의 tok/s를 비교한다.

## 6. 확인 문제

> 해설: [answers/14_kv_cache.md](answers/14_kv_cache.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. m5 프리셋 모델($L$=6, $H$=6, $d_h$=64, $T$=256, fp32)의 KV cache 크기를 계산하고 가중치 크기(12.2M × 4B)와 비교하라.
2. 배치 생성에서 프롬프트 길이가 서로 다르면 이 구현(공통 `pos`)에 어떤 문제가 생기는가? left padding과 attention mask로 어떻게 해결하는가?
3. 이 모델에서 cache 사용 시 속도 향상이 이론값보다 작게 나올 수 있는 이유는? (힌트: 작은 모델에서 커널 실행 오버헤드)

## 참고문헌

- R. Pope et al., "Efficiently Scaling Transformer Inference," MLSys 2023.
- N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.
- J. Ainslie et al., "GQA," EMNLP 2023.
- W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
