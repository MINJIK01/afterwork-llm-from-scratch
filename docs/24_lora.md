# Step 24. LoRA: Low-Rank Adaptation

- 구현: `minillm/lora.py` — `LoRALinear.forward`, `LoRALinear.merge` (`apply_lora`, `merge_lora`는 제공됨)
- 테스트: `pytest tests/test_24_lora.py`
- 실행: `python scripts/sft.py --lora_r 8 --lr 1e-3`
- 선행: step 15 (step 15 직후에 진행해도 된다)

## 1. 목표

사전학습 가중치를 고정하고 저랭크 행렬 두 개만 학습하는 LoRA를 구현해 SFT에 적용한다. 학습 파라미터 수, 메모리, 성능을 전체 미세조정과 비교한다.

## 2. 이론

### 2.1 동기

전체 미세조정은 모든 파라미터의 gradient와 optimizer 상태(파라미터당 12B 추가, step 23)를 요구하고, 과제마다 모델 전체를 복사해 저장해야 한다. Aghajanyan et al. (2021)은 사전학습 모델의 미세조정이 낮은 내재 차원(intrinsic dimension)의 부분공간에서도 대부분의 성능을 낸다는 것을 보였다. LoRA는 가중치 **변화량** $\Delta W$가 저랭크라고 가정한다.

### 2.2 정식화

$W_0 \in \mathbb R^{d_{out}\times d_{in}}$을 고정하고

$$h = W_0 x + \frac{\alpha}{r} B A x,\qquad A \in \mathbb R^{r \times d_{in}},\ B \in \mathbb R^{d_{out}\times r},\ r \ll \min(d_{in}, d_{out}).$$

학습 파라미터는 $r(d_{in} + d_{out})$개로, 전체 $d_{in}d_{out}$ 대비 $r(d_{in}+d_{out})/(d_{in}d_{out})$이다. 예: m5 프리셋의 `qkv`($384 \to 1152$), $r = 8$이면 $8 \times 1536 = 12{,}288$ 대 $442{,}368$(2.8%).

### 2.3 초기화

$B = 0$, $A$는 Kaiming uniform. 그러면 학습 시작 시 $\Delta W = BA = 0$이어서 모델이 사전학습 모델과 정확히 같다. gradient는

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r} G (Ax)^\top,\qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r} B^\top G x^\top$$

($G = \partial L/\partial h$)이므로 첫 스텝에서는 $B = 0$ 때문에 $\partial L/\partial A = 0$이고 $B$만 갱신된다. 다음 스텝부터 $A$도 학습된다. 둘 다 0으로 초기화하면 두 gradient가 모두 0이 되어 학습이 시작되지 않는다.

### 2.4 $\alpha / r$ 스케일

Adam은 업데이트 크기를 대략 lr로 정규화하므로, $r$을 바꾸면 $BA$ 변화량의 크기가 $r$에 따라 달라진다. $\alpha/r$을 곱하면 $r$을 바꿔도 lr을 다시 튜닝할 필요가 줄어든다. rsLoRA(Kalajdzievski 2023)는 큰 $r$에서 학습이 느려지는 문제를 분석해 $\alpha/\sqrt r$을 제안했다.

### 2.5 추론 시 병합

학습 후 $W = W_0 + \frac{\alpha}{r}BA$로 합치면 일반 Linear와 구조가 같아져 추론 지연이 0이다. 병합하지 않고 base 모델 하나에 과제별 어댑터($A, B$)만 바꿔 끼우는 서빙도 가능하다. `merge_lora`는 병합 후 `state_dict` 키를 원래 모델과 같게 만들어 이후 스크립트(DPO, GRPO)가 일반 체크포인트처럼 불러올 수 있게 한다.

### 2.6 적용 위치와 메모리

- 원 논문은 attention의 $W_Q$, $W_V$에만 적용했다. QLoRA(Dettmers et al. 2023)는 모든 Linear에 적용해야 전체 미세조정 성능에 가까워진다고 보고했다. 이 레포의 기본값은 블록의 모든 Linear(`qkv`/`q`/`kv`, `proj`, `w_gate`, `w_up`, `w_down`)이며, tied 된 `lm_head`와 임베딩은 고정한다.
- 줄어드는 메모리는 gradient와 optimizer 상태이다. activation은 여전히 저장해야 하므로 큰 배치에서는 step 23의 checkpointing과 함께 쓴다.
- QLoRA는 base 가중치를 4비트(NF4)로 양자화(step 19와 연결)하고 그 위에 LoRA를 학습해 메모리를 더 줄인다.
- Biderman et al. (2024)은 LoRA가 전체 미세조정보다 새 지식을 덜 학습하지만 기존 능력도 덜 잊는다고 보고했다.

### 2.7 이 레포의 실험

```bash
python scripts/sft.py --lora_r 8 --lr 1e-3 --out out/sft_lora.pt
```

학습 파라미터 수(로그의 `trainable params`)와 학습 후 키워드 적중률을 전체 SFT(`out/sft.pt`)와 비교한다. LoRA는 학습 파라미터가 적어 전체 미세조정보다 큰 lr(1e-3 수준)을 쓴다.

## 3. 구현 가이드

- `forward`: `self.base(x) + (x @ A.T @ B.T) * scaling`. $BA$를 먼저 곱해 $d_{out}\times d_{in}$ 행렬을 만들지 말고, $x$에 $A^\top$, $B^\top$ 순으로 곱해 $O(r(d_{in}+d_{out}))$로 계산한다.
- `merge`: 새 `nn.Linear`에 `base.weight + scaling * B @ A`와 bias를 복사.

## 4. 검증

초기 출력 = base 출력, 학습 파라미터가 `A`, `B`뿐인지, forward 식과 병합 결과 일치, 초기 $\partial L/\partial A = 0$이고 $B$만 gradient를 받는지, GPT에 적용 후 학습하면 loss가 줄고 병합 후 `state_dict` 키와 출력이 유지되는지 확인한다.

## 5. 확인 문제

> 해설: [answers/24_lora.md](answers/24_lora.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. m5 프리셋 전체에 $r = 8$로 적용하면 학습 파라미터는 몇 개이고 전체의 몇 %인가?
2. $A$와 $B$를 모두 작은 랜덤 값으로 초기화하면 어떤 문제가 생기는가?
3. 학습된 $\Delta W$의 특이값 분포를 보면 $r$을 어떻게 정할 수 있는가? 실제로 `B @ A`의 특이값을 계산해 보라.

## 참고문헌

- E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
- A. Aghajanyan, S. Gupta, L. Zettlemoyer, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," ACL 2021.
- T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023.
- D. Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA," arXiv:2312.03732, 2023.
- D. Biderman et al., "LoRA Learns Less and Forgets Less," TMLR 2024.
