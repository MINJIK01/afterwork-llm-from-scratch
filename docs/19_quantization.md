# Step 19. Weight-only 양자화 (int8 per-channel, int4 group-wise)

- 구현: `minillm/quant.py` — `quantize`, `dequantize` (`QuantLinear`, `quantize_model`, `weight_bytes`는 제공됨)
- 테스트: `pytest tests/test_19_quantization.py`
- 실행: `python scripts/bench_inference.py` (정밀도별 perplexity, 가중치 메모리)

## 1. 목표

대칭 균일 양자화를 구현하고, 양자화 단위(per-channel, per-group)에 따른 오차와 perplexity 변화를 측정한다.

## 2. 이론

### 2.1 균일 양자화

$b$비트 정수로 실수 $w$를 표현한다:

$$q = \mathrm{clamp}\big(\mathrm{round}(w/s) + z,\ q_{min},\ q_{max}\big),\qquad \hat w = s(q - z).$$

**대칭** 양자화는 $z = 0$, $q \in [-q_{max}, q_{max}]$, $q_{max} = 2^{b-1} - 1$(int8: 127, int4: 7), $s = \max|w|/q_{max}$. 가중치 분포는 대체로 0 중심 대칭이라 대칭 방식으로 충분하며, $-2^{b-1}$을 쓰지 않아 부호 대칭을 유지한다.

### 2.2 오차

범위 안에서 반올림 오차는 $|w - \hat w| \le s/2$이다. 오차를 $[-s/2, s/2]$의 균등분포로 근사하면 평균제곱오차는 $s^2/12$. $s$가 그룹 내 **최댓값**에 비례하므로, 큰 값(outlier) 하나가 같은 그룹 모든 원소의 해상도를 떨어뜨린다.

### 2.3 양자화 단위와 오버헤드

| 단위 | scale 개수 (`(out, in)` 행렬) | 특징 |
|---|---|---|
| per-tensor | 1 | outlier에 가장 취약 |
| per-channel (출력 행) | out | 행마다 분포가 다른 경우 대응 |
| per-group (입력 축 $G$개씩) | out × in/$G$ | 해상도 최고, scale 오버헤드 증가 |

가중치당 실효 비트 수 $= b + (\text{scale 비트})/G$. 예: int4, $G = 32$, fp16 scale이면 $4 + 16/32 = 4.5$비트.

출력 채널 단위 scale은 행렬곱에서 분리할 수 있다: $y_k = \sum_i W_{ki}x_i \approx s_k\sum_i q_{ki}x_i$. 정수 누적 후 scale을 한 번 곱하면 되므로 정수 연산 커널과 잘 맞는다. 그룹 단위는 그룹별 부분합마다 scale을 곱해야 한다.

### 2.4 Weight-only vs weight-activation

- **Weight-only**(W8A16, W4A16): 가중치만 저비트로 저장하고 연산 직전에 역양자화한다. decode 단계는 가중치 읽기가 병목(memory-bound, step 14)이므로 읽는 바이트를 줄이는 것만으로 속도가 오른다.
- **Weight-activation**(W8A8): 정수 행렬곱으로 연산량도 줄인다. LLM activation에는 특정 채널에 매우 큰 값이 몰리는 outlier feature가 나타나 per-tensor activation 양자화가 어렵다(Dettmers et al. 2022). SmoothQuant는 이 난이도를 가중치 쪽으로 옮기는 채널별 스케일링을 쓴다.
- 반올림(RTN) 대신 오차를 보정하는 방법: GPTQ(2차 정보로 남은 가중치를 갱신), AWQ(activation 크기 기준으로 중요한 채널을 보호).

### 2.5 이 구현의 한계

`QuantLinear`는 forward마다 float로 역양자화한 뒤 일반 matmul을 한다. 저장 메모리는 줄지만, 역양자화와 matmul을 하나의 커널로 합치지 않았기 때문에 속도 이득은 없다(오히려 느리다). 실제 추론 엔진(llama.cpp, MLX, Marlin 커널 등)은 저비트 가중치를 읽으면서 레지스터에서 역양자화해 대역폭 이득을 얻는다. `bench_inference.py`는 메모리와 perplexity를 비교하며, int4는 저장 단위가 int8이므로 표시되는 메모리는 비트 수로 계산한 이론값이다. 임베딩/lm_head(tied)는 fp32로 남긴다.

## 3. 예제로 따라가기

### 3.1 가중치 한 행을 int8로

$w = [0.12,\ -0.50,\ 0.33,\ 2.00,\ -0.07,\ 0.25,\ -0.31,\ 0.04]$ (행 하나, 8개 원소).

| 단계 | 값 |
|---|---|
| $\max|w|$ | 2.00 |
| scale $s = 2.00/127$ | 0.01575 |
| $q = \mathrm{round}(w/s)$ | [8, −32, 21, 127, −4, 16, −20, 3] |
| $\hat w = s\cdot q$ | [0.1260, −0.5039, 0.3307, 2.0000, −0.0630, 0.2520, −0.3150, 0.0472] |
| 최대 오차 | 0.0072 ($\le s/2 = 0.0079$) |

8비트면 모든 값이 2자리 정밀도 정도로 보존된다.

### 3.2 같은 행을 int4로: outlier의 영향

int4는 $q \in [-7, 7]$, 15단계뿐이다.

**행 전체를 한 그룹으로 (per-channel)**

| | 값 |
|---|---|
| scale $= 2.00/7$ | 0.2857 |
| $q$ | [0, −2, 1, 7, **0**, 1, −1, **0**] |
| $\hat w$ | [0, −0.5714, 0.2857, 2.0, **0**, 0.2857, −0.2857, **0**] |

outlier 2.00 때문에 간격이 0.2857로 커져, 0.12, −0.07, 0.04가 모두 0이 되었다.

**4개씩 그룹으로 (group-wise, $G = 4$)**

| 그룹 | 원소 | scale | $q$ | 그룹 최대 오차 |
|---|---|---|---|---|
| 1 | 0.12, −0.50, 0.33, **2.00** | 0.2857 | 0, −2, 1, 7 | 0.1200 |
| 2 | −0.07, 0.25, −0.31, 0.04 | **0.0443** | −2, 6, −7, 1 | **0.0443** |

outlier가 없는 그룹 2는 자기만의 작은 scale을 가져 오차가 약 1/3로 줄었다. 대가는 scale을 2개 저장하는 것이다.

```
per-channel :  |-------------- scale 1개 --------------|
              0.12 -0.50 0.33 2.00 -0.07 0.25 -0.31 0.04

group (G=4) :  |---- scale A ----|  |---- scale B ----|
              0.12 -0.50 0.33 2.00   -0.07 0.25 -0.31 0.04
```

### 3.3 메모리 계산 (블록 가중치 10.6M개)

| 형식 | 가중치당 비트 | 크기 | fp32 대비 |
|---|---|---|---|
| fp32 | 32 | 40.5 MiB | 1× |
| int8 per-channel (행마다 fp32 scale) | 약 8 | 약 10.1 MiB | 약 4× |
| int4 $G = 32$ (fp32 scale) | 4 + 32/32 = 5 | 6.3 MiB | 6.4× |
| int4 $G = 32$ (fp16 scale) | 4 + 16/32 = 4.5 | 5.7 MiB | 7.1× |

### 3.4 확인 코드

```python
import torch
from minillm.quant import quantize, dequantize

w = torch.tensor([[0.12, -0.50, 0.33, 2.00, -0.07, 0.25, -0.31, 0.04]])
for bits, g in [(8, None), (4, None), (4, 4)]:
    q, s = quantize(w, bits, g)
    print(bits, g, q.tolist(), dequantize(q, s))
```

## 4. 구현 가이드

- `quantize(w, bits, group_size)`: `w.reshape(out, in // G, G)` (G = `group_size or in`), `amax(-1, keepdim=True)`로 그룹 최댓값, `clamp(min=1e-8)`(전부 0인 그룹 대비), `round` → `clamp(-qmax, qmax)` → `int8`. scale shape은 `(out, in // G, 1)`.
- `dequantize(q, scale)`: 같은 shape으로 reshape 후 `q.float() * scale`, 다시 `(out, in)`.

## 5. 검증

int8/int4의 오차가 그룹별 $s/2$ 이내인지, outlier가 다른 그룹 오차에 영향을 주지 않는지, 양자화 모델 출력이 원 모델과 가까운지를 확인한다.

## 6. 확인 문제

> 해설: [answers/19_quantization.md](answers/19_quantization.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. m5 프리셋 모델의 블록 가중치(임베딩 제외)를 int4/$G$=32(fp32 scale)로 저장하면 몇 MiB인가? fp32 대비 몇 배 작은가?
2. 대칭 대신 비대칭($z \ne 0$) 양자화가 유리한 분포는 어떤 경우인가?
3. RTN 양자화가 행렬곱 출력 오차 $\|WX - \hat WX\|$를 최소화하지 못하는 이유는? GPTQ가 최소화하는 목적함수를 쓰라.

## 참고문헌

- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018.
- M. Nagel et al., "A White Paper on Neural Network Quantization," arXiv:2106.08295, 2021.
- T. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS 2022.
- G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," ICML 2023.
- E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023.
- J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024.
