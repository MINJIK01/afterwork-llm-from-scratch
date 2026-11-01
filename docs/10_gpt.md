# Step 10. GPT 모델 조립: 임베딩, weight tying, 초기화

- 구현: `minillm/model.py` — `GPT.__init__`(tying·초기화), `GPT.hidden`, `GPT.forward`
- 테스트: `pytest tests/test_10_gpt.py`
- 선행: step 02, 07, 09

## 1. 목표

토큰 임베딩 → $L$개 블록 → final RMSNorm → lm_head로 이어지는 decoder-only LM을 조립하고, GPT-2식 초기화와 weight tying을 적용한다.

## 2. 이론

### 2.1 모델과 목적함수

$$h^{(0)} = E[x],\quad h^{(l)} = \mathrm{Block}_l(h^{(l-1)}),\quad \mathrm{logits} = \mathrm{RMSNorm}(h^{(L)})\,W_{out}^\top$$

$$\mathcal L(\theta) = -\frac{1}{BT}\sum_{b,t}\log p_\theta(x_{b,t+1}\mid x_{b,\le t})$$

토큰당 평균 음의 로그우도이며 step 02의 `cross_entropy`로 계산한다.

### 2.2 Weight tying

출력층 가중치를 입력 임베딩과 공유한다: $W_{out} = E \in \mathbb R^{V \times d}$ (Press & Wolf 2017). $V d$개의 파라미터를 절약하고, 출력 쪽 gradient가 모든 토큰 임베딩을 매 스텝 갱신하므로 드문 토큰의 임베딩 학습에도 도움이 된다. PyTorch에서는 `self.lm_head.weight = self.tok_emb.weight`로 같은 `Parameter` 객체를 가리키게 한다. `model.parameters()`는 중복을 제거해 한 번만 반환한다.

### 2.3 초기화

- 모든 Linear/Embedding: $\mathcal N(0, 0.02^2)$ (GPT-2).
- residual stream에 더해지는 출력 투영(`attn.proj`, `ffn.w_down`): 표준편차 $0.02/\sqrt{2L}$.

근거: residual stream은 $x_L = x_0 + \sum_{k=1}^{2L} f_k$ ($L$개 블록 × 2개 서브레이어). 각 기여가 서로 독립이고 분산 $\sigma_f^2$이면 $\mathrm{Var}(x_L) = \mathrm{Var}(x_0) + 2L\sigma_f^2$로 깊이에 비례해 커진다. 출력 투영의 표준편차를 $1/\sqrt{2L}$배 하면 $\sigma_f^2 \propto 1/(2L)$이 되어 합의 분산이 깊이와 무관하게 유지된다.

RMSNorm gain(1차원 파라미터)은 1로 둔다.

### 2.4 초기 loss 점검

초기 logits는 0 근처이므로 출력 분포가 거의 균등하고 loss $\approx \ln V$이다. $V = 4096$이면 $\ln 4096 \approx 8.32$. 학습 시작 loss가 이보다 크게 높으면 초기화나 구현에 문제가 있다는 신호다. (tied embedding에서는 target이 입력 토큰과 같을 때 그 logit이 높아지므로, 점검 시 target은 입력과 무관하게 뽑아야 한다.)

### 2.5 파라미터 수와 FLOPs

$$N = Vd + L(4d^2 + 3dh + 2d) + d$$

m5 프리셋($V$=4096, $d$=384, $h$=1024, $L$=6): $1{,}572{,}864 + 6 \times 1{,}770{,}240 + 384 = 12{,}194{,}688 \approx 12.2$M.

학습 FLOPs는 토큰당 약 $6N + 6LTd$ (앞항: 행렬곱 forward 2N + backward 4N, step 03; 뒷항: attention score/가중합, Kaplan et al. 2020 Table 1).

### 2.6 `hidden`과 `forward`의 분리

reward model(step 16)은 lm_head 대신 스칼라 head를 붙이므로 final norm까지의 hidden state가 필요하다. `hidden(idx, caches)`는 hidden을, `forward`는 logits와 loss를 반환한다. RoPE 테이블은 `register_buffer(persistent=False)`로 등록해 device 이동은 따라가되 체크포인트에는 저장하지 않는다.

## 3. 구현 가이드

- `__init__`: tying → `named_parameters()`를 돌며 `p.dim() >= 2`인 파라미터를 정규분포로 초기화하되 이름이 `proj.weight`, `w_down.weight`로 끝나면 축소된 표준편차 사용.
- `hidden`: `cos, sin`은 cache 위치(`start`)부터 $T$개를 잘라 둔 상태로 주어진다. 블록 $i$에 `caches[i]`(없으면 None)를 넘긴다.
- `forward`: `cross_entropy(logits.flatten(0, 1), targets.flatten())`.

## 4. 검증

출력 shape, loss가 `F.cross_entropy`와 일치, 초기 loss ≈ $\ln V$, tying 여부, 초기화 표준편차, 파라미터 수 공식, causality를 확인한다.

## 5. 확인 문제

1. tying을 하면 임베딩 $E$의 gradient는 어떤 두 경로의 합이 되는가?
2. $L = 24$일 때 residual 출력 투영의 초기 표준편차는? 이를 적용하지 않으면 초기 residual stream의 분산은 몇 배가 되는가?
3. m5 프리셋에서 $T = 256$일 때 토큰당 학습 FLOPs 중 attention 항의 비율은?

## 참고문헌

- A. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019.
- O. Press, L. Wolf, "Using the Output Embedding to Improve Language Models," EACL 2017.
- J. Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020.
- H. Touvron et al., "LLaMA," arXiv:2302.13971, 2023.
