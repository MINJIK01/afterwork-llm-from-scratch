# Step 12. 학습 루프: gradient accumulation, 평가, 스케일링 법칙

- 구현: `minillm/trainer.py` — `train_step`, `evaluate`
- 테스트: `pytest tests/test_12_training.py`
- 실행: `python scripts/pretrain.py --preset m5` (M5 Mac)
- 선행: step 04, 10, 11

## 1. 목표

마이크로배치 누적, gradient clipping, optimizer step을 하나의 학습 스텝으로 묶고, perplexity 평가 함수를 구현한다. 그 다음 실제로 TinyStories에서 12M 모델을 학습한다.

## 2. 이론

### 2.1 Perplexity와 bits-per-byte

$$\mathrm{PPL} = \exp(\mathcal L_{CE})$$

토큰당 평균 음의 로그우도의 지수이며, 모델이 매 스텝 균등분포로 고르는 후보 수와 같은 불확실성을 의미한다(균등 분포 $V$개면 PPL = $V$). PPL은 토크나이저에 의존하므로 토크나이저가 다른 모델끼리 비교할 때는 bits-per-byte를 쓴다:

$$\mathrm{BPB} = \frac{\mathcal L_{CE}}{\ln 2}\cdot\frac{\#\text{tokens}}{\#\text{bytes}}.$$

### 2.2 Gradient accumulation

메모리 한계로 큰 배치를 한 번에 올릴 수 없을 때, 크기가 같은 $K$개의 마이크로배치로 나눈다. 전체 배치 평균 loss는 $\mathcal L = \frac1K\sum_k \mathcal L_k$이고 gradient는 선형이므로

$$\nabla\mathcal L = \frac1K\sum_k\nabla\mathcal L_k.$$

각 마이크로배치의 loss를 $1/K$배 해서 `backward()`하면 PyTorch가 `.grad`에 합산하므로 전체 배치 gradient와 정확히 같아진다. 마이크로배치마다 유효 토큰 수가 다르면(ignore_index가 섞인 SFT 등) 단순 $1/K$ 가중은 전체 토큰 평균과 달라진다는 점에 주의한다.

스텝 순서: `zero_grad` → 마이크로배치별 forward/backward → clip(누적된 전체 gradient에 한 번) → `optimizer.step()`. lr은 스텝 전에 스케줄(step 04)로 설정한다.

### 2.3 평가

`model.eval()` + `torch.no_grad()`로 고정 시드의 검증 배치에서 평균 loss를 계산한다. 이 레포의 모델에는 dropout이 없어 train/eval 동작이 같지만, 일반적으로 dropout/BatchNorm 동작을 바꾸기 위해 모드를 전환하는 습관을 들인다. 평가 후 `model.train()`으로 복귀해야 한다.

### 2.4 디버깅: 단일 배치 overfit

배치 하나를 반복 학습해 loss가 0 근처로 떨어지는지 확인하는 것은 구현 오류(마스크, shift, 학습률)를 빠르게 찾는 표준 절차다. 테스트에도 포함되어 있다.

### 2.5 계산 예산과 스케일링 법칙

학습 FLOPs $C \approx 6ND$ ($N$: 파라미터, $D$: 토큰). Chinchilla(Hoffmann et al. 2022)는 고정 $C$에서 loss를 최소화하는 배분이 대략 $D \approx 20N$임을 보였다. 12.2M 모델이면 약 240M 토큰이 compute-optimal이다.

m5 프리셋은 4000 스텝 × 16,384 토큰 ≈ 65.5M 토큰, $C \approx 6 \times 12.2\text{M} \times 65.5\text{M} \approx 4.8 \times 10^{15}$ FLOPs로 Chinchilla 기준보다 적게 학습한다(노트북 시간 제약). Eldan & Li (2023)는 TinyStories에서 수천만 파라미터 이하 모델도 문법적으로 일관된 이야기를 생성함을 보였으므로, 이 규모로도 이후 정렬 단계를 진행할 수 있는 모델을 얻을 수 있다. 시간이 허락하면 `--max_steps`를 늘려 비교해 보라.

처리량이 `tok/s`라면 MFU(model FLOPs utilization) $= 6N \cdot \mathrm{tok/s} / \text{하드웨어 피크 FLOPs}$로 하드웨어 활용도를 추정할 수 있다.

### 2.6 모니터링 지표

- train/val loss 차이: 과적합 여부.
- grad norm: 급증(spike)은 불안정 신호. clipping 임계값(1.0)과 비교.
- lr: 스케줄이 의도대로 적용되는지.

`pretrain.py`는 이 지표들을 `--out`과 같은 이름의 CSV(기본 `out/pretrain.csv`)에 기록한다. 설정을 바꾼 실행(예: `--dtype bf16`, `--n_kv_head 2`)을 `--out`만 달리해 돌리면 CSV끼리 학습 곡선을 비교할 수 있다. 체크포인트에는 optimizer 상태와 데이터 샘플링 난수 상태도 저장되므로, `--resume`으로 재개하면 중단 없이 학습한 것과 같은 결과가 나온다.

## 3. 구현 가이드

- `train_step(model, optimizer, batches, grad_clip)`: `batches`는 `[(x, y), ...]`. 반환은 `(평균 loss: float, 클리핑 전 grad norm: float)`.
- `evaluate`: step 11의 `get_batch`를 `iters`번 호출해 평균 loss, `math.exp(loss)` 반환.
- MPS: `loss.item()`은 GPU 동기화를 일으킨다. 마이크로배치당 한 번은 허용 범위이다.

## 4. 검증

2개 마이크로배치 누적 gradient가 전체 배치 gradient와 같은지(`lr=0`으로 파라미터 고정), 단일 배치 overfit, `evaluate`의 반환값과 train 모드 복귀를 확인한다.

## 5. 확인 문제

> 해설: [answers/12_training.md](answers/12_training.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. 마이크로배치 크기가 2, 3으로 다르면 단순 $1/K$ 가중은 어떤 가중 평균이 되는가?
2. 같은 토큰 수에서 배치 크기를 키우고 스텝 수를 줄이면 어떤 trade-off가 있는가? (critical batch size, McCandlish et al. 2018)
3. 학습 로그의 `tok/s`로 이 Mac의 MFU를 추정하라.

## 참고문헌

- J. Kaplan et al., "Scaling Laws for Neural Language Models," 2020.
- J. Hoffmann et al., "Training Compute-Optimal Large Language Models," NeurIPS 2022.
- R. Eldan, Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?," arXiv:2305.07759, 2023.
- S. McCandlish et al., "An Empirical Model of Large-Batch Training," arXiv:1812.06162, 2018.
