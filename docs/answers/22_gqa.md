# Step 22 확인 문제 해설

[← 본문](../22_gqa.md)

## 1. m5 프리셋의 `n_kv_head`별 KV cache와 파라미터 수

KV cache ($B$=1, $T$=256, fp32) $= 2\cdot6\cdot G\cdot64\cdot256\cdot4 = 786{,}432\,G$ B. attention 파라미터(레이어당) $= 2d^2 + 2\cdot d\cdot G\cdot64 = 294{,}912 + 49{,}152\,G$.

| $G$ (`n_kv_head`) | KV cache | attention 파라미터/레이어 | 모델 전체 파라미터 |
|---|---|---|---|
| 6 (MHA) | 4.50 MiB | 589,824 | 12,194,688 |
| 3 | 2.25 MiB | 442,368 | 11,309,952 |
| 2 | 1.50 MiB | 393,216 | 11,015,040 |
| 1 (MQA) | 0.75 MiB | 344,064 | 10,720,128 |

(전체 파라미터는 `GPT(GPTConfig(n_kv_head=G)).num_params()`로 확인한 값.)

## 2. 배치 1 decode에서 GQA가 이득이 되는 조건

토큰 하나를 생성할 때 읽는 바이트 ≈ 가중치 $N b$ + KV cache $2LGd_hTb$ (같은 dtype $b$ 가정). GQA의 속도비는 대략

$$\frac{N + 2LHd_hT}{N + 2LGd_hT}.$$

KV 항이 가중치와 비슷하거나 커야 의미 있는 차이가 난다: $2LHd_hT \gtrsim N$, 즉 $T \gtrsim N/(2Ld)$ ($Hd_h = d$). m5 프리셋은 $12.2\text{M}/(2\times6\times384) \approx 2{,}650$토큰인데 block_size가 256이므로, **배치 1에서는 GQA 이득이 작다**. 배치가 $B$이면 KV 항이 $B$배가 되므로 $BT \gtrsim 2{,}650$이면 의미가 생긴다. 대형 모델은 $N$이 크지만 $L$, $d$, 문맥 길이, 배치도 함께 커서 서빙 시 KV가 지배적이 된다.

## 3. MHA → GQA 변환 시 head 하나만 고르면?

그룹 안 다른 head들이 학습한 K/V 정보를 버리게 되어, 그 head들에 의존하던 query head들의 attention 패턴이 크게 바뀐다. 평균(mean pooling)은 그룹 내 정보를 모두 반영한 초기값을 주어 uptraining 시작점이 원래 모델에 더 가깝다. Ainslie et al.은 mean pooling이 첫 head 선택이나 랜덤 초기화보다 좋다고 보고했다.
