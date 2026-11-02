# 30분 세션 학습 계획

[`STUDY_PLAN.md`](STUDY_PLAN.md)(평일 60분 + 주말 3시간)를 더 잘게 나눈 버전이다. 퇴근 후 집중력이 떨어지는 날에도 한 단위를 끝낼 수 있도록, 모든 세션을 **30분 안에 끝나는 하나의 작업**으로 나눴다.

- 세션 106개, 합계 약 53시간. 요일에 묶여 있지 않으므로 번호 순서대로 진행한다.
- 구현 세션은 함수 1–3개만 다루고, **특정 테스트가 통과하면 끝난다**. 시간이 아니라 완료 기준이 세션의 단위이다. 30분 안에 못 끝내면 다음 날 같은 세션을 이어서 한다.
- 각 구현 세션의 완료 기준은 "직전 세션까지 구현한 상태에서는 실패하고, 그 세션의 함수를 구현하면 통과한다"를 함수 단위 시뮬레이션으로 검증했다. 순서를 바꾸면 기준 테스트가 통과하지 않을 수 있다.
- 학습 실행 시간은 M5에서 실측하지 않은 추정치이다.

## 진행 속도

| 진행 방식 | 기간 |
|---|---|
| 평일 1세션 (주 5세션, 주 2.5시간) | 약 21주 |
| 매일 1세션 (주 7세션, 주 3.5시간) | 약 15주 |
| 평일 1세션 + 주말 하루 4세션 (주 9세션, 주 4.5시간) | 약 12주 |
| 평일 2세션 (주 10세션, 주 5시간) | 약 11주 |

## 세션 유형

| 유형 | 30분 구성 |
|---|---|
| 읽기 | 25분 문서의 해당 절을 읽으며 유도를 종이에 따라 쓰기, 5분 핵심을 한 줄로 요약 |
| 구현 | 3분 지난 세션 테스트 확인, 7분 해당 절 읽기, 17분 구현과 테스트, 3분 커밋 |
| 문제 | 20분 확인 문제 풀기, 10분 `docs/answers/` 대조 |
| 실행 | 스크립트를 실행하고 결과를 기록한다. 긴 학습을 걸어 두기만 하는 세션은 5분 안에 끝난다 |
| 버퍼 | 밀린 세션 처리. 밀린 것이 없으면 적힌 과제 |

막혔을 때, 긴 실행, 브랜치 운영 규칙은 [`STUDY_PLAN.md`의 세션 운영 규칙](STUDY_PLAN.md#세션-운영-규칙)과 같다.

## 0. 준비

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 1 | -- | 준비 | README의 환경 설정을 따라 venv와 패키지 설치, `docs/STUDY_PLAN.md`의 세션 운영 규칙 읽기, 본인 구현용 브랜치 생성(`git switch -c my-impl`) | `MINILLM_SOLUTIONS=1 pytest tests -q` → 84 passed |

## 1부. 기초 (step 01–04)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 2 | 01 | 읽기 | `docs/01` §2 전체. §2.2의 국소 미분 표를 종이에 옮겨 적기 | 표를 보지 않고 7개 연산의 국소 미분을 쓸 수 있음 |
| 3 | 01 | 구현 | `Value.__add__`, `Value.__mul__`의 `_backward`, `Value.backward`(위상 정렬) | `pytest tests/test_01_autograd.py -k "reused or deep_chain"` 통과 |
| 4 | 01 | 구현 | `__pow__`, `exp`, `log`, `tanh`, `relu`의 `_backward` | `pytest tests/test_01_autograd.py` 통과 |
| 5 | 01 | 문제 | 확인 문제 01 풀기 → `docs/answers/01` 대조 | 틀린 문항의 이유를 한 줄로 메모 |
| 6 | 02 | 구현 | `docs/02` §2.1–2.2 읽고 `softmax`, `log_softmax` | `pytest tests/test_02_softmax_xent.py -k "softmax_and_log or stability"` 통과 |
| 7 | 02 | 구현 | §2.3 읽고 `cross_entropy` (ignore_index 마스킹) | `pytest tests/test_02_softmax_xent.py -k "ignore_index"` 통과 |
| 8 | 02 | 구현 | §2.4 유도를 종이로 따라간 뒤 `cross_entropy_backward` | `pytest tests/test_02_softmax_xent.py` 통과 |
| 9 | 02 | 문제 | 확인 문제 02 → 해설 대조 | 해설 대조 완료 |
| 10 | 03 | 구현 | `docs/03` §2.1–2.2 읽고 `linear_backward` | `pytest tests/test_03_manual_backward.py -k "linear"` 통과 |
| 11 | 03 | 구현 | §2.3 읽고 `layer_norm_forward` (cache 구성 결정) | `pytest tests/test_03_manual_backward.py -k "forward"` 통과 |
| 12 | 03 | 읽기 | §2.4의 1–3단계 유도를 종이로: $\partial\sigma/\partial x_j = \hat x_j/D$까지 | 보지 않고 3단계까지 재현 |
| 13 | 03 | 구현 | §2.4의 4–5단계 유도 → `layer_norm_backward` | `pytest tests/test_03_manual_backward.py` 통과 |
| 14 | 03 | 문제 | 확인 문제 03 → 해설 대조 | 해설 대조 완료 |
| 15 | 04 | 읽기 | `docs/04` §2.1–2.3. bias correction 유도를 종이로 | $\mathbb E[m_t] = (1-\beta_1^t)\mathbb E[g]$를 보지 않고 유도 |
| 16 | 04 | 구현 | `AdamW.step` | `pytest tests/test_04_optim.py -k "adamw"` 통과 |
| 17 | 04 | 구현 | §2.4–2.5 읽고 `cosine_lr`, `clip_grad_norm_` | `pytest tests/test_04_optim.py` 통과 |
| 18 | 04 | 문제 | 확인 문제 04 → 해설 대조 | 해설 대조 완료 |

## 2부. 토크나이저와 Transformer (step 05–09)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 19 | 05 | 읽기 | `docs/05` §2.1–2.3. `aaabdaaabac` 예제를 손으로 3번 병합 | 병합 결과 `Z=aa, Y=ab, X=ZY`를 직접 도출 |
| 20 | 05 | 구현 | `get_pair_counts`, `merge` | `pytest tests/test_05_bpe.py -k "pair_counts"` 통과 |
| 21 | 05 | 구현 | `BPETokenizer.train` (동률 규칙 포함) | `pytest tests/test_05_bpe.py -k "wikipedia"` 통과 |
| 22 | 05 | 구현 | `_encode_chunk`, `decode` | `pytest tests/test_05_bpe.py` 통과 |
| 23 | 05 | 실행 | `python scripts/prepare_data.py` 시작 (수 분~수십 분, 추정). 도는 동안 §2.4–2.7 읽기 | `data/train.bin` 생성, 출력된 bytes/token 기록 |
| 24 | 05 | 문제 | 확인 문제 05 → 해설 대조 | 해설 대조 완료 |
| 25 | 06 | 구현 | `docs/06` §2.1–2.3 읽고 `scaled_dot_product_attention` | `pytest tests/test_06_attention.py` 통과 |
| 26 | 06 | 문제 | §2.4–2.5 읽기, 확인 문제 06 → 해설 대조 | 해설 대조 완료 |
| 27 | 07 | 읽기 | `docs/07` §2.1–2.3. 2차원 회전으로 상대 위치 성질을 종이로 유도 | $R_m^\top R_n = R_{n-m}$을 보지 않고 보임 |
| 28 | 07 | 구현 | `precompute_rope`, `apply_rope` | `pytest tests/test_07_rope.py` 통과 |
| 29 | 07 | 문제 | §2.4–2.5 읽기, 확인 문제 07 → 해설 대조 | 해설 대조 완료 |
| 30 | 08 | 읽기 | `docs/08` §2.1–2.4. $T=1$, $T_k=5$일 때 오프셋 마스크를 종이에 그리기 | `tril(diagonal=Tk-T)`의 결과를 손으로 씀 |
| 31 | 08 | 구현 | `CausalSelfAttention.forward` 두 구간 (QKV 분리·head 분할·RoPE / 마스크·attention·출력 투영) | `pytest tests/test_08_causal_attention.py` 통과 |
| 32 | 08 | 문제 | 확인 문제 08 → 해설 대조 | 해설 대조 완료 |
| 33 | 09 | 구현 | `docs/09` §2.1–2.3 읽고 `RMSNorm.forward` | `pytest tests/test_09_block.py -k "rmsnorm"` 통과 |
| 34 | 09 | 구현 | §2.4 읽고 `SwiGLU.forward`, `Block.forward` | `pytest tests/test_09_block.py` 통과 |
| 35 | 09 | 문제 | 확인 문제 09 → 해설 대조 | 해설 대조 완료 |

## 3부. 프리트레이닝 (step 10–12)

이 부의 마지막에 처음으로 1–2시간짜리 학습을 돌린다.

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 36 | 10 | 읽기 | `docs/10` §2.1–2.3. residual 분산 $\mathrm{Var}(x_L) = \mathrm{Var}(x_0) + 2L\sigma_f^2$ 유도 | 초기화 스케일 $0.02/\sqrt{2L}$의 근거를 설명할 수 있음 |
| 37 | 10 | 구현 | `GPT.__init__`의 weight tying과 초기화 | `pytest tests/test_10_gpt.py -k "tying"` 통과 |
| 38 | 10 | 구현 | `GPT.hidden`, `GPT.forward` | `pytest tests/test_10_gpt.py` 통과 |
| 39 | 10 | 문제 | §2.4–2.6 읽기, 확인 문제 10 → 해설 대조 | 해설 대조 완료 |
| 40 | 11 | 구현 | `docs/11` 읽고 `pack_documents`, `get_batch` | `pytest tests/test_11_data.py` 통과 |
| 41 | 11 | 문제 | 확인 문제 11 → 해설 대조 | 해설 대조 완료 |
| 42 | 12 | 구현 | `docs/12` §2.1–2.4 읽고 `train_step` | `pytest tests/test_12_training.py -k "accumulation or overfit"` 통과 |
| 43 | 12 | 구현 | `evaluate` 구현 후 `python scripts/pretrain.py --preset tiny`로 학습 흐름 확인 (수 분) | `pytest tests/test_12_training.py` 통과 |
| 44 | 12 | 실행 | 잠들기 전 `caffeinate -i python scripts/pretrain.py --preset m5` 시작 (5분이면 끝나는 세션, 학습은 1–2시간 추정) | 첫 로그 줄 확인 후 종료 |
| 45 | 12 | 실행 | `out/pretrain.csv`로 train/val loss 곡선 확인, 생성 샘플 읽기, 해설 12의 스니펫으로 MFU 측정 | val loss, 샘플 1개, MFU 기록 |
| 46 | 12 | 문제 | §2.5–2.6 읽기, 확인 문제 12 → 해설 대조 | 해설 대조 완료 |
| 47 | -- | 버퍼 | 밀린 세션 처리 | — |
| 48 | -- | 버퍼 | 1–3부 확인 문제 중 틀린 것 다시 풀기 | — |
| 49 | -- | 버퍼 | 정답을 봤던 구간을 보지 않고 다시 작성 | — |
| 50 | -- | 버퍼 | 선택: step 21 (심화 첫 세션부터) | — |

## 4부. 디코딩 (step 13–14)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 51 | 13 | 구현 | `docs/13` §2.1–2.2 읽고 `sample_next` (greedy 분기와 multinomial 분기) | `pytest tests/test_13_sampling.py -k "greedy"` 통과 |
| 52 | 13 | 구현 | §2.3–2.5 읽고 `filter_probs` (top-k, top-p 경계 사례) | `pytest tests/test_13_sampling.py -k "top_k or top_p or frequency"` 통과 |
| 53 | 13 | 구현 | §2.6 읽고 `generate`의 샘플링 구간 | `pytest tests/test_13_sampling.py` 통과 |
| 54 | 13 | 문제 | 확인 문제 13 → 해설 대조 | 해설 대조 완료 |
| 55 | 14 | 구현 | `docs/14` §2.1–2.5 읽고 `KVCache.update` | `pytest tests/test_14_kv_cache.py -k "not bytes"` 통과 |
| 56 | 14 | 구현 | `kv_cache_bytes` 구현 후 `python scripts/bench_inference.py`의 KV cache 부분 확인 | `pytest tests/test_14_kv_cache.py` 통과 |
| 57 | 14 | 실행 | 확인 문제 14 → 해설 대조, 이어서 `python scripts/pretrain.py --preset draft` 시작 (15–30분 추정, step 20용) | `out/draft.pt` 생성 |

## 5부. 정렬 (step 15–18)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 58 | 15 | 구현 | `docs/15` 읽고 `build_sft_example` | `pytest tests/test_15_sft.py` 통과 |
| 59 | 15 | 실행 | `python scripts/sft.py` (10–20분 추정), 학습 전후 키워드 적중률과 샘플 확인 | 전후 적중률 기록 |
| 60 | 15 | 문제 | 확인 문제 15 → 해설 대조 | 해설 대조 완료 |
| 61 | 16 | 구현 | `docs/16` §2.1–2.3 읽고 `bradley_terry_loss` ($z=0$ gradient 주의) | `pytest tests/test_16_reward_model.py -k "bradley"` 통과 |
| 62 | 16 | 구현 | §2.4–2.5 읽고 `RewardModel.forward` | `pytest tests/test_16_reward_model.py` 통과 |
| 63 | 16 | 문제 | 확인 문제 16 (3번은 직접 실행) → 해설 대조 | 해설 대조 완료 |
| 64 | 17 | 읽기 | `docs/17` §2.1–2.2. 최적 정책의 닫힌 형태를 보지 않고 유도 | $\pi^* \propto \pi_{ref}e^{r/\beta}$ 유도 |
| 65 | 17 | 읽기 | §2.3–2.5. BT에 대입해 $Z(x)$ 소거, gradient 식 해석 | DPO loss를 보지 않고 씀 |
| 66 | 17 | 구현 | `token_logprobs`, `sequence_logprobs` | `pytest tests/test_17_dpo.py -k "logprobs"` 통과 |
| 67 | 17 | 구현 | `dpo_loss` | `pytest tests/test_17_dpo.py` 통과 |
| 68 | 17 | 실행 | `python scripts/dpo.py` (20–40분 추정) | reward acc와 학습 후 적중률 기록 |
| 69 | 17 | 실행 | `python scripts/compare.py`로 SFT와 DPO 비교, §2.6–2.8 읽기 | 적중률·길이·KL·val loss 변화 기록 |
| 70 | 17 | 문제 | 확인 문제 17 → 해설 대조 | 해설 대조 완료 |
| 71 | 18 | 읽기 | `docs/18` §2.1–2.3. REINFORCE와 baseline의 비편향성 유도 | $\mathbb E[b\nabla\log\pi]=0$을 보임 |
| 72 | 18 | 구현 | §2.4 읽고 `group_advantages` | `pytest tests/test_18_grpo.py -k "group"` 통과 |
| 73 | 18 | 구현 | §2.5 읽고 `ppo_clip_loss` (표의 네 경우 확인) | `pytest tests/test_18_grpo.py -k "ppo"` 통과 |
| 74 | 18 | 구현 | §2.6 읽고 $k_3$ 비편향성 유도 후 `kl_k3` | `pytest tests/test_18_grpo.py` 통과 |
| 75 | 18 | 실행 | 잠들기 전 `caffeinate -i python scripts/grpo.py` 시작 (30–60분 추정) | 첫 iteration 로그 확인 |
| 76 | 18 | 실행 | `compare.py`로 SFT/DPO/GRPO 비교, §2.7–2.8 읽기 | 세 방식의 차이 기록 |
| 77 | 18 | 문제 | 확인 문제 18 (3번 $k_3$ gradient 유도 포함) → 해설 대조 | 해설 대조 완료 |

## 6부. 서빙 최적화 (step 19–20)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 78 | 19 | 구현 | `docs/19` §2.1–2.3 읽고 `quantize`, `dequantize` | `pytest tests/test_19_quantization.py` 통과 |
| 79 | 19 | 실행 | `python scripts/bench_inference.py`의 양자화 부분, §2.4–2.5 읽기 | 정밀도별 perplexity·메모리 기록 |
| 80 | 19 | 문제 | 확인 문제 19 → 해설 대조 | 해설 대조 완료 |
| 81 | 20 | 읽기 | `docs/20` §2.1–2.4. 출력 분포 = $p$ 증명을 보지 않고 재현 | 증명 재현 |
| 82 | 20 | 구현 | `speculative_accept` | `pytest tests/test_20_speculative.py -k "accept"` 통과 |
| 83 | 20 | 구현 | §2.6 읽고 `speculative_generate` (cache rollback 불변식) | `pytest tests/test_20_speculative.py` 통과 |
| 84 | 20 | 실행 | `python scripts/speculative.py`, 확인 문제 20 → 해설 대조 | 라운드당 확정 토큰 수를 이론값과 비교 |
| 85 | -- | 버퍼 | 밀린 세션 처리 | — |
| 86 | -- | 버퍼 | 밀린 세션 처리 / 4–6부 확인 문제 복습 | — |
| 87 | -- | 버퍼 | 종합 1: 본인 구현으로 `prepare_data` → 잠들기 전 `pretrain --preset m5` 시작 | — |
| 88 | -- | 버퍼 | 종합 2: `sft` → `dpo` → `grpo` (각각 별도 세션으로 나눠도 됨) | — |
| 89 | -- | 버퍼 | 종합 3: `compare` → `bench_inference` → `speculative` | — |
| 90 | -- | 버퍼 | 회고: 가장 어려웠던 단계 3개와 이유 기록 | — |

## 7부. 심화 (step 21–24)

| # | Step | 유형 | 할 일 | 완료 기준 |
|---|---|---|---|---|
| 91 | 21 | 읽기 | `docs/21` §2.1–2.3. online softmax 갱신식의 불변식 확인 | $a/l$이 정확한 이유를 설명 |
| 92 | 21 | 구현 | `online_logsumexp` | `pytest tests/test_21_flash.py -k "logsumexp"` 통과 |
| 93 | 21 | 구현 | `flash_attention` | `pytest tests/test_21_flash.py` 통과 |
| 94 | 21 | 문제 | §2.4–2.6 읽기, 확인 문제 21 → 해설 대조 | 해설 대조 완료 |
| 95 | 22 | 구현 | `docs/22` 읽고 `repeat_kv` | `pytest tests/test_22_gqa.py -k "repeat"` 통과 |
| 96 | 22 | 구현 | `GroupedQueryAttention.forward` 두 구간. 잠들기 전 선택: `pretrain.py --preset m5 --n_kv_head 2 --out out/pretrain_gqa.pt` | `pytest tests/test_22_gqa.py` 통과 |
| 97 | 22 | 문제 | CSV 비교, `bench_inference.py --ckpt out/pretrain_gqa.pt`, 확인 문제 22 → 해설 대조 | KV cache 크기·val loss 비교 기록 |
| 98 | 23 | 구현 | `docs/23` §2.1–2.2 읽고 `bf16_round` (비트 연산) | `pytest tests/test_23_precision.py -k "bf16_round"` 통과 |
| 99 | 23 | 구현 | §2.5 읽고 `LossScaler.unscale_`, `LossScaler.update` | `pytest tests/test_23_precision.py -k "scaler"` 통과 |
| 100 | 23 | 구현 | §2.3–2.4 읽고 `train_step_amp` | `pytest tests/test_23_precision.py -k "amp"` 통과 |
| 101 | 23 | 읽기 | §3.1–3.3 (학습 메모리 구성, 재계산, custom autograd Function) | checkpoint가 저장하는 것과 버리는 것을 설명 |
| 102 | 23 | 구현 | `CheckpointFunction.forward`, `CheckpointFunction.backward` | `pytest tests/test_23_precision.py` 통과 |
| 103 | 23 | 실행 | `pretrain.py --preset tiny`를 fp32와 `--dtype bf16 --grad_ckpt`로 각각 돌려 tok/s 비교, 확인 문제 23 → 해설 대조 | tok/s 비교 기록 |
| 104 | 24 | 구현 | `docs/24` §2.1–2.4 읽고 `LoRALinear.forward` | `pytest tests/test_24_lora.py -k "init or adapters"` 통과 |
| 105 | 24 | 구현 | §2.5 읽고 `LoRALinear.merge` | `pytest tests/test_24_lora.py` 통과 |
| 106 | 24 | 실행 | `sft.py --lora_r 8 --lr 1e-3 --out out/sft_lora.pt`, `compare.py --ref out/pretrain.pt`, 확인 문제 24 → 해설 대조 | 전체 SFT와 비교 기록 |

## 진도 확인

현재 세션까지의 테스트를 한 번에 돌려 회귀가 없는지 확인한다. 예: 51번 세션(step 13 시작) 직전

```bash
pytest tests/test_0*.py tests/test_1[0-2]*.py -q
```

