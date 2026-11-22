# Step 07. Rotary Position Embedding (RoPE)

- 구현: `minillm/layers.py` — `precompute_rope`, `apply_rope`
- 테스트: `pytest tests/test_07_rope.py`

## 1. 목표

Su et al. (2021)의 RoPE를 구현하고, 회전 후 내적이 상대 위치 $m - n$에만 의존한다는 성질을 수치로 확인한다.

## 2. 이론

### 2.1 위치 정보가 필요한 이유와 기존 방식

step 06에서 보았듯 attention은 순열 등변이다. 위치를 주입하는 방식은 크게 세 가지다.

- **절대 위치(additive)**: 임베딩에 $PE(m)$을 더한다. sinusoidal(Vaswani 2017: $PE_{m,2i} = \sin(m/10000^{2i/d})$, $PE_{m,2i+1} = \cos(\cdot)$) 또는 학습형(GPT-2).
- **상대 위치(attention bias)**: logit에 $b(m - n)$을 더한다(Shaw et al. 2018, T5, ALiBi).
- **RoPE(multiplicative)**: $q, k$를 위치에 따라 회전시켜 내적에 상대 위치가 자연스럽게 나타나게 한다. LLaMA 이후 대부분의 LLM이 사용한다.

### 2.2 2차원에서의 유도

목표: 함수 $f(x, m)$으로 위치 $m$의 query와 위치 $n$의 key를 변환했을 때

$$\langle f(q, m), f(k, n)\rangle = g(q, k, m - n)$$

이 되게 하는 것. 2차원 벡터를 복소수 $q = q_1 + i q_2$로 보고 $f(q, m) = q e^{im\theta}$로 두면

$$\langle f(q,m), f(k,n)\rangle = \mathrm{Re}\left[q e^{im\theta}\,\overline{k e^{in\theta}}\right] = \mathrm{Re}\left[q\bar k\, e^{i(m-n)\theta}\right]$$

로 $m - n$에만 의존한다. 실수 표현으로는 회전행렬

$$R_m = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix},\qquad R_m^\top R_n = R_{n-m}$$

이므로 $(R_m q)^\top (R_n k) = q^\top R_{n-m} k$.

### 2.3 $d$차원 확장

head 차원 $d$를 $d/2$개의 2차원 쌍으로 나누고, 쌍 $i$마다 다른 주파수

$$\theta_i = \mathrm{base}^{-2i/d},\quad i = 0, \dots, d/2 - 1,\quad \mathrm{base} = 10000$$

로 회전한다. 전체 변환은 블록 대각 회전행렬이므로 직교변환이고 노름을 보존한다. 주파수는 $1$부터 $1/\mathrm{base}$까지 기하급수적으로 분포하여, 파장 $2\pi/\theta_i$가 약 $6$토큰부터 약 $6 \times 10^4$토큰까지 덮는다. 빠른 주파수 쌍은 가까운 위치 구분, 느린 주파수 쌍은 먼 위치 구분에 쓰인다.

### 2.4 적용 위치

- $q$와 $k$에만 적용하고 $v$에는 적용하지 않는다. 위치 정보는 attention 가중치 계산(내적)에만 필요하기 때문이다.
- head를 나눈 뒤 head별로 적용한다(step 08). 회전 각도는 head 간에 공유된다.
- 학습 길이를 넘는 위치로 외삽하면 본 적 없는 각도 조합이 생긴다. Position Interpolation, NTK-aware scaling, YaRN 등은 $m$ 또는 base를 조정해 문맥 길이를 확장한다.

### 2.5 쌍을 구성하는 방식

- **interleaved**: $(x_0, x_1), (x_2, x_3), \dots$ — 원 논문과 복소수 표현에 대응. 이 레포의 규약.
- **half-split**: $(x_0, x_{d/2}), (x_1, x_{d/2+1}), \dots$ — GPT-NeoX, HF LLaMA 구현의 `rotate_half`.

두 방식은 차원의 순열만 다르고, 학습과 추론에서 일관되게 쓰면 동등하다. 다만 다른 구현의 체크포인트를 불러올 때는 규약이 일치해야 한다.

## 3. 예제로 따라가기

### 3.1 회전 하나: 2차원 벡터를 위치만큼 돌리기

$\theta = 1$ rad, $q = (1, 0)$을 위치 $m$에 따라 $m\theta$만큼 회전시키면:

| 위치 $m$ | 각도 $m\theta$ (rad) | $R_m q = (\cos m\theta,\ \sin m\theta)$ |
|---|---|---|
| 0 | 0 | (1.0000, 0.0000) |
| 1 | 1 | (0.5403, 0.8415) |
| 2 | 2 | (−0.4161, 0.9093) |
| 3 | 3 | (−0.9900, 0.1411) |

```
        y
        |   m=1 ●
  m=2 ● |
        |
 m=3 ●--+---------● m=0   x
        |
```

길이는 항상 1이다. 회전은 벡터의 크기를 바꾸지 않고, 위치 정보를 각도로만 표현한다.

### 3.2 상대 위치 성질을 숫자로 확인

$\theta = 0.5$, $q = (1, 2)$, $k = (3, -1)$. query를 위치 $m$, key를 위치 $n$에서 회전시킨 뒤 내적:

| $(m, n)$ | $m - n$ | $\langle R_m q, R_n k\rangle$ |
|---|---|---|
| (3, 1) | 2 | −5.3500 |
| (10, 8) | 2 | −5.3500 |
| (2, 0) | 2 | −5.3500 |
| (0, 0) | 0 | 1.0000 (= 회전 없는 $q\cdot k$) |
| (1, 3) | −2 | 6.4306 |

절대 위치가 달라도 거리 $m - n$이 같으면 attention score가 정확히 같다. 부호까지 구분되므로($m - n = 2$와 $-2$가 다름) "앞에 있는 토큰"과 "뒤에 있는 토큰"도 구분된다.

### 3.3 head 차원 8일 때의 주파수

$\theta_i = 10000^{-2i/8}$:

| 쌍 $i$ | 성분 | $\theta_i$ | 한 바퀴 도는 데 걸리는 토큰 수 $2\pi/\theta_i$ |
|---|---|---|---|
| 0 | $(x_0, x_1)$ | 1 | 6.3 |
| 1 | $(x_2, x_3)$ | 0.1 | 62.8 |
| 2 | $(x_4, x_5)$ | 0.01 | 628 |
| 3 | $(x_6, x_7)$ | 0.001 | 6,283 |

쌍 0은 몇 토큰만 지나도 한 바퀴를 돌아 **가까운** 위치 차이를 세밀하게 구분하지만, 6토큰 이상 떨어진 위치는 각도가 겹쳐 구분하지 못한다. 쌍 3은 수천 토큰에 걸쳐 천천히 돌아 **먼** 위치 차이를 구분한다. 주기가 다른 여러 쌍을 함께 쓰기 때문에 짧은 거리와 긴 거리를 모두 구분할 수 있다.

### 3.4 `apply_rope`의 인덱싱

```
x        = [x0, x1, x2, x3, x4, x5, x6, x7]
x1 = x[0::2] = [x0, x2, x4, x6]     ← 각 쌍의 첫 성분
x2 = x[1::2] = [x1, x3, x5, x7]     ← 각 쌍의 둘째 성분
cos, sin    : [θ0·m, θ1·m, θ2·m, θ3·m] 각도의 cos, sin  (shape (T, 4))

new_first  = x1·cos − x2·sin
new_second = x1·sin + x2·cos
stack(..., dim=-1) → [[f0, s0], [f1, s1], [f2, s2], [f3, s3]]
flatten(-2)        → [f0, s0, f1, s1, f2, s2, f3, s3]   ← 원래 인터리브 순서로 복원
```

### 3.5 확인 코드

```python
import torch
from minillm.layers import precompute_rope, apply_rope

cos, sin = precompute_rope(head_dim=2, max_seq_len=4, base=1.0)   # θ = 1
q = torch.tensor([[1.0, 0.0]])
for m in range(4):
    print(m, apply_rope(q, cos[m:m + 1], sin[m:m + 1]))  # 3.1의 표
```

## 4. 구현 가이드

- `precompute_rope`: `inv_freq = base ** (-arange(0, d, 2) / d)`, `angles = outer(arange(T), inv_freq)` → `(cos, sin)` 각각 `(T, d/2)`.
- `apply_rope`: `x1, x2 = x[..., 0::2], x[..., 1::2]`, 회전 결과 `(x1 cos − x2 sin, x1 sin + x2 cos)`를 `stack(dim=-1).flatten(-2)`로 다시 interleave. `cos, sin`의 shape `(T, d/2)`는 `x`의 `(..., T, d/2)`와 broadcast된다.

## 5. 검증

복소수 곱 $x \cdot e^{im\theta}$로 만든 참조값과 비교하고, 노름 보존과 $\langle R_{10}q, R_3k\rangle = \langle R_{40}q, R_{33}k\rangle$를 확인한다.

## 6. 확인 문제

> 해설: [answers/07_rope.md](answers/07_rope.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. $v$에도 RoPE를 적용하면 출력에 어떤 변화가 생기는가? 상대 위치 성질이 유지되는가?
2. $\mathrm{base}$를 키우면 가장 느린 주파수의 파장은 어떻게 변하는가? 긴 문맥 확장 기법들이 base를 조정하는 이유와 연결하라.
3. sinusoidal 절대 위치 인코딩을 더하는 방식에서 $q_m^\top k_n$을 전개하면 어떤 항들이 생기는가? 상대 위치에만 의존하지 않는 항은 무엇인가?

## 참고문헌

- J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.
- P. Shaw, J. Uszkoreit, A. Vaswani, "Self-Attention with Relative Position Representations," NAACL 2018.
- O. Press, N. A. Smith, M. Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation," ICLR 2022.
- S. Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," arXiv:2306.15595, 2023.
- B. Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," ICLR 2024.
