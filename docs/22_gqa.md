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

## 3. 구현 가이드

- `repeat_kv(x, n_rep)`: `n_rep == 1`이면 그대로 반환.
- `forward` 첫 구간: `q`는 `(B, H, T, d_h)`, `self.kv(x)`를 `(B, T, 2G, d_h)`로 view 후 transpose하고 `chunk(2, dim=1)`로 앞 $G$개를 K, 뒤 $G$개를 V로. RoPE를 q, k에 적용.
- 두 번째 구간: step 08과 같은 오프셋 마스크, `repeat_kv`로 확장한 K/V로 attention, 출력 투영.

## 4. 검증

`repeat_kv`의 head 대응, `F.scaled_dot_product_attention(enable_gqa=True)` 참조와의 일치, MQA 파라미터 수와 causality, GQA GPT에서 cache 증분 logits = 전체 logits, cache head 수를 확인한다.

## 5. 확인 문제

> 해설: [answers/22_gqa.md](answers/22_gqa.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. m5 프리셋에서 `n_kv_head` 6, 3, 2, 1일 때 KV cache 크기와 파라미터 수를 계산하라.
2. 배치 1 decode에서 GQA가 MHA보다 빠르려면 KV cache 읽기가 전체 메모리 트래픽에서 어느 정도 비중이어야 하는가? 가중치 크기와 문맥 길이로 표현하라.
3. MHA 체크포인트를 GQA로 변환할 때 K/V head를 평균하는 대신 하나만 고르면 어떤 문제가 생길 수 있는가?

## 참고문헌

- N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," arXiv:1911.02150, 2019.
- J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023.
- H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288, 2023.
- A. Q. Jiang et al., "Mistral 7B," arXiv:2310.06825, 2023.
