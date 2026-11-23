# Step 22. Grouped-Query Attention (GQA)와 Multi-Query Attention (MQA)

- 구현: `minillm/layers.py` — `repeat_kv`, `GroupedQueryAttention.forward`
- 테스트: `pytest tests/test_22_gqa.py`
- 실행: `python scripts/pretrain.py --preset m5 --n_kv_head 2` 후 `python scripts/bench_inference.py`
- 선행: step 08, 14

## 1. 목표

여러 query head가 K/V head를 공유하는 GQA를 구현하고 GPT에 통합한다. KV cache 메모리와 decode 단계 산술 강도에 미치는 영향을 정량화한다.

## 2. 이론

### 2.1 MHA → MQA → GQA

| 방식 | Q head | K/V head | KV cache (토큰·레이어당 원소) |
|---|---|---|---|
| MHA | $H$ | $H$ | $2Hd_h$ |
| GQA | $H$ | $G$ ($1 < G < H$) | $2Gd_h$ |
| MQA | $H$ | 1 | $2d_h$ |

GQA에서 query head $h$는 KV head $\lfloor h / (H/G) \rfloor$를 쓴다. 즉 연속된 $H/G$개 query head가 한 그룹으로 K/V를 공유한다.

### 2.2 동기: decode는 KV cache 읽기에 묶여 있다

step 14에서 보았듯 decode 단계는 메모리 대역폭이 병목이다. 긴 문맥·큰 배치에서는 KV cache 크기가 가중치보다 커지고(LLaMA-2-7B, $T$=4096, 배치 8에서 16GiB), 매 토큰 이를 전부 읽어야 한다. K/V head 수를 $H \to G$로 줄이면 cache 크기와 읽는 바이트가 $G/H$배가 된다. 같은 데이터로 $H/G$개 query head를 계산하므로 KV 읽기 1바이트당 FLOPs(산술 강도)는 $H/G$배 높아진다.

### 2.3 품질 trade-off

MQA(Shazeer 2019)는 cache를 최대한 줄이지만 품질 저하와 학습 불안정이 보고되었다. GQA(Ainslie et al. 2023)는 그 사이를 택해, $G = 8$ 정도에서 MHA에 가까운 품질과 MQA에 가까운 속도를 얻었다. LLaMA-2 70B, Mistral 7B, LLaMA-3 등이 GQA를 쓴다. Ainslie et al.은 기존 MHA 체크포인트의 K/V head를 그룹별로 평균내어 GQA로 변환한 뒤 원래 학습량의 5% 정도 추가 학습(uptraining)하는 방법도 제안했다.

### 2.4 파라미터 수

$W_Q$: $d^2$, $W_{KV}$: $2 \cdot d \cdot G d_h$, $W_O$: $d^2$. MHA의 $4d^2$보다 $2d^2(1 - G/H)$만큼 적다. 같은 파라미터 예산을 맞추려면 FFN을 키우는 것이 일반적이다.

### 2.5 구현: repeat_kv

K, V를 $(B, G, T, d_h)$로 계산·저장하고, attention 직전에 각 head를 $H/G$번 **연속 반복**해 $(B, H, T, d_h)$로 만든다(`repeat_interleave(n_rep, dim=1)`). `repeat`(전체 블록 반복)를 쓰면 head 대응이 $h \bmod G$로 바뀌어 `F.scaled_dot_product_attention(enable_gqa=True)`와 규약이 달라진다. 반복은 attention 계산에서만 일어나고 cache에는 $G$개 head만 저장한다. 실제 커널은 복사 없이 인덱스로 같은 K/V를 공유한다.

### 2.6 이 레포에서의 통합

`GPTConfig.n_kv_head`가 None이거나 `n_head`와 같으면 기존 `CausalSelfAttention`, 아니면 `GroupedQueryAttention`이 쓰인다. `GPT.new_caches`는 `n_kv_head`개 head로 cache를 할당한다. `--n_kv_head 2`(m5 프리셋 $H = 6$이므로 그룹 크기 3)로 학습해 MHA 모델과 val loss와 KV cache 크기를 비교해 보라.

## 3. 예제로 따라가기

### 3.1 head 대응표 ($H = 6$, $G = 2$)

`repeat_interleave(3, dim=1)`로 KV head를 3번씩 연속 반복한다.

| query head | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 사용하는 KV head ($h // 3$) | 0 | 0 | 0 | 1 | 1 | 1 |

```
K (저장)     : [K0, K1]                         (B, 2, T, d_h)
repeat_kv(3) : [K0, K0, K0, K1, K1, K1]         (B, 6, T, d_h)
```

`repeat(1, 3, 1, 1)`을 쓰면 `[K0, K1, K0, K1, K0, K1]`이 되어 대응이 $h \bmod 2$로 바뀐다. 학습한 모델을 다른 구현에서 불러올 때 결과가 달라진다.

### 3.2 같은 모델에서 MHA, GQA, MQA 비교 (m5 프리셋, $H = 6$, $d_h = 64$)

| | KV head | `kv` 가중치 shape | attention 파라미터/레이어 | KV cache (B=1, T=256, fp32) |
|---|---|---|---|---|
| MHA | 6 | (768, 384) | 589,824 | 4.50 MiB |
| GQA | 3 | (384, 384) | 442,368 | 2.25 MiB |
| GQA | 2 | (256, 384) | 393,216 | 1.50 MiB |
| MQA | 1 | (128, 384) | 344,064 | 0.75 MiB |

(MHA 행의 `kv` shape은 비교를 위해 K, V를 합쳐 계산한 값이다. 실제 MHA 구현은 Q까지 합친 `qkv`를 쓴다.)

### 3.3 decode 한 토큰에서 읽는 바이트

7B급 모델(가중치 fp16 약 13 GiB, $L = 32$, $H = 32$, $d_h = 128$), 배치 8, 문맥 4096:

| | KV cache | 가중치 + KV | MHA 대비 |
|---|---|---|---|
| MHA ($G = 32$) | 16 GiB | 29 GiB | 1.0 |
| GQA ($G = 8$) | 4 GiB | 17 GiB | 0.59 |
| MQA ($G = 1$) | 0.5 GiB | 13.5 GiB | 0.47 |

decode는 memory-bound이므로 토큰당 시간은 대략 읽는 바이트에 비례한다. 이 설정에서는 GQA만으로 토큰당 읽는 양이 약 40% 줄어든다.

### 3.4 확인 코드

```python
import torch
from minillm.layers import repeat_kv

k = torch.arange(2).float().view(1, 2, 1, 1)   # KV head 0, 1을 값 0, 1로 표시
print(repeat_kv(k, 3).flatten().tolist())       # [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
```

## 4. 구현 가이드

- `repeat_kv(x, n_rep)`: `n_rep == 1`이면 그대로 반환.
- `forward` 첫 구간: `q`는 `(B, H, T, d_h)`, `self.kv(x)`를 `(B, T, 2G, d_h)`로 view 후 transpose하고 `chunk(2, dim=1)`로 앞 $G$개를 K, 뒤 $G$개를 V로. RoPE를 q, k에 적용.
- 두 번째 구간: step 08과 같은 오프셋 마스크, `repeat_kv`로 확장한 K/V로 attention, 출력 투영.

## 5. 검증

`repeat_kv`의 head 대응, `F.scaled_dot_product_attention(enable_gqa=True)` 참조와의 일치, MQA 파라미터 수와 causality, GQA GPT에서 cache 증분 logits = 전체 logits, cache head 수를 확인한다.

## 6. 확인 문제

> 해설: [answers/22_gqa.md](answers/22_gqa.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. m5 프리셋에서 `n_kv_head` 6, 3, 2, 1일 때 KV cache 크기와 파라미터 수를 계산하라.
2. 배치 1 decode에서 GQA가 MHA보다 빠르려면 KV cache 읽기가 전체 메모리 트래픽에서 어느 정도 비중이어야 하는가? 가중치 크기와 문맥 길이로 표현하라.
3. MHA 체크포인트를 GQA로 변환할 때 K/V head를 평균하는 대신 하나만 고르면 어떤 문제가 생길 수 있는가?

## 참고문헌

- N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv:1911.02150, 2019.
- J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023.
- H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288, 2023.
- A. Q. Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023.
