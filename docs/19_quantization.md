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

## 3. 구현 가이드

- `quantize(w, bits, group_size)`: `w.reshape(out, in // G, G)` (G = `group_size or in`), `amax(-1, keepdim=True)`로 그룹 최댓값, `clamp(min=1e-8)`(전부 0인 그룹 대비), `round` → `clamp(-qmax, qmax)` → `int8`. scale shape은 `(out, in // G, 1)`.
- `dequantize(q, scale)`: 같은 shape으로 reshape 후 `q.float() * scale`, 다시 `(out, in)`.

## 4. 검증

int8/int4의 오차가 그룹별 $s/2$ 이내인지, outlier가 다른 그룹 오차에 영향을 주지 않는지, 양자화 모델 출력이 원 모델과 가까운지를 확인한다.

## 5. 확인 문제

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
