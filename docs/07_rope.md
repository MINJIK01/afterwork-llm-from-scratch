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

## 3. 구현 가이드

- `precompute_rope`: `inv_freq = base ** (-arange(0, d, 2) / d)`, `angles = outer(arange(T), inv_freq)` → `(cos, sin)` 각각 `(T, d/2)`.
- `apply_rope`: `x1, x2 = x[..., 0::2], x[..., 1::2]`, 회전 결과 `(x1 cos − x2 sin, x1 sin + x2 cos)`를 `stack(dim=-1).flatten(-2)`로 다시 interleave. `cos, sin`의 shape `(T, d/2)`는 `x`의 `(..., T, d/2)`와 broadcast된다.

## 4. 검증

복소수 곱 $x \cdot e^{im\theta}$로 만든 참조값과 비교하고, 노름 보존과 $\langle R_{10}q, R_3k\rangle = \langle R_{40}q, R_{33}k\rangle$를 확인한다.

## 5. 확인 문제

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
