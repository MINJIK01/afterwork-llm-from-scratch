# 퇴근 후 학습 계획 (14주)

직장인이 퇴근 후 시간으로 24단계를 끝까지 진행할 수 있도록 세션 단위로 나눈 계획이다.

## 기본 가정

| 항목 | 기본 트랙 | 느린 트랙 |
|---|---|---|
| 평일 | 3회 × 60분 (예: 월·수·목) | 2회 × 60분 |
| 주말 | 1회 × 3시간 | 1회 × 2시간 |
| 주당 시간 | 6시간 | 4시간 |
| 기간 | 14주 (본 과정 12주 + 심화 2주) | 약 22주 (아래 "느린 트랙" 참고) |

- 본 과정(01–20)의 순수 학습량은 약 40시간, 심화(21–24)는 약 8시간으로 추정했다. 나머지 시간은 복습, 실험 분석, 따라잡기용이다.
- 6주차와 12주차는 진도를 넣지 않은 **버퍼 주**이다. 밀린 단계를 마무리하는 데 쓰고, 밀린 것이 없으면 표에 적힌 선택 과제를 한다.
- 학습 실행 시간(프리트레이닝 1–2시간 등)은 연산량으로 계산한 **추정치이며 M5에서 실측하지 않았다**. 첫 실행 후 실제 시간으로 이 문서를 고쳐 쓰는 것을 권장한다.

## 세션 운영 규칙

**평일 60분 세션의 구성**

| 시간 | 할 일 |
|---|---|
| 5분 | 지난 세션 테스트 상태 확인: `pytest tests/test_NN_*.py -q` |
| 15분 | 문서의 해당 절 읽기. 수식은 종이에 직접 따라 쓴다 |
| 35분 | 구현 → 테스트 |
| 5분 | 커밋, 다음 세션에서 할 일을 한 줄로 메모 |

**막혔을 때**
- 같은 문제로 30분 이상 막히면 `solutions/minillm/`의 해당 구간을 확인한다. 확인한 코드는 덮어 두고, 다음 세션에 보지 않고 처음부터 다시 작성한다.
- 한 단계가 며칠씩 막혀 진도가 멈추면 그 단계는 미뤄 두고 `MINILLM_ONLY`로 뒤 단계를 먼저 진행한다(README "앞 단계 없이 한 단계만 채점하기"). 미룬 단계는 버퍼 주에 처리한다.

**피곤한 날**
- 구현을 건너뛰더라도 최소 세션(15–20분)은 유지한다: 지난 단계의 확인 문제 하나를 풀고 `docs/answers/`와 대조한다. 학습 흐름이 끊기지 않게 하는 것이 목적이다.

**긴 실행**
- 1시간 이상 걸리는 학습은 세션을 마칠 때나 잠들기 전에 시작하고, 결과 분석은 다음 세션에서 한다.
- macOS가 잠자기에 들어가면 학습이 멈추므로 `caffeinate -i`로 실행한다.

```bash
caffeinate -i python scripts/pretrain.py --preset m5
```

- 중간에 끊겼으면 같은 명령에 `--resume`을 붙여 이어서 학습한다.

**브랜치**
- 본인 구현은 별도 브랜치에 커밋한다. 레포가 업데이트되어도 본인 코드와 섞이지 않는다.

```bash
git switch -c my-impl
```

## 주차별 계획

표기: S1–S3 = 평일 세션, W = 주말 세션. 각 주의 마지막 줄은 그 주가 끝났을 때 동작해야 하는 것이다.

### 1부. 기초 (1–2주)

**1주차 — autograd, cross-entropy** (step 01–02)
- [ ] S1: 환경 설정(README "환경 설정"), `MINILLM_SOLUTIONS=1 pytest tests -q`로 84개 통과 확인, README 훑기, `docs/01` 읽기
- [ ] S2: step 01 구현 → `test_01` 통과
- [ ] S3: `docs/02` 읽기, `softmax`, `log_softmax`, `cross_entropy` 구현
- [ ] W: `cross_entropy_backward` 유도(종이) → 구현, 확인 문제 01·02 풀고 해설 대조
- 완료 기준: `pytest tests/test_01_autograd.py tests/test_02_softmax_xent.py` 통과

**2주차 — 수동 역전파, optimizer** (step 03–04)
- [ ] S1: `docs/03` 읽기, `linear_backward` 구현
- [ ] S2: `layer_norm_forward` 구현, LayerNorm backward 유도 시작(2.4의 1–3단계)
- [ ] S3: `docs/04` 읽기, `cosine_lr`, `clip_grad_norm_` 구현
- [ ] W: LayerNorm backward 유도 완료 → 구현, `AdamW.step` 구현, 확인 문제 03·04
- 완료 기준: Linear/LayerNorm 역전파와 AdamW가 PyTorch와 일치 (`test_03`, `test_04`)

### 2부. 토크나이저와 Transformer (3–4주)

**3주차 — BPE, 데이터 준비** (step 05)
- [ ] S1: `docs/05` 읽기, `get_pair_counts`, `merge` 구현
- [ ] S2: `BPETokenizer.train` 구현 (동률 규칙 포함)
- [ ] S3: `_encode_chunk`, `decode` 구현 → `test_05` 통과
- [ ] W: 본인 BPE로 `python scripts/prepare_data.py` 실행(다운로드·학습·토큰화, 수 분~수십 분 추정), 출력된 bytes/token 확인, 확인 문제 05, 남는 시간에 `docs/06` 미리 읽기
- 완료 기준: `data/train.bin`, `data/tokenizer.json` 생성

**4주차 — attention, RoPE, 블록** (step 06–09)
- [ ] S1: step 06 구현(짧음), `docs/07` 읽기
- [ ] S2: step 07 RoPE 구현
- [ ] S3: `docs/08` 읽기, `CausalSelfAttention.forward` 첫 구간(QKV 분리, head 분할, RoPE)
- [ ] W: step 08 두 번째 구간(오프셋 마스크) 완성, step 09 구현, 확인 문제 06–09
- 완료 기준: `test_06`–`test_09` 통과

### 3부. 프리트레이닝 (5주)

**5주차 — GPT 조립, 학습 루프, 첫 학습** (step 10–12)
- [ ] S1: `docs/10` 읽기, `GPT.__init__`(tying, 초기화) 구현
- [ ] S2: `GPT.hidden`, `GPT.forward` → `test_10`, `docs/11` 읽고 step 11 구현
- [ ] S3: step 12 `train_step`, `evaluate` → `test_12`, `python scripts/pretrain.py --preset tiny`로 동작 확인(수 분)
- [ ] S3 종료 후 또는 잠들기 전: `caffeinate -i python scripts/pretrain.py --preset m5` 시작 (1–2시간 추정)
- [ ] W: `out/pretrain.csv`로 train/val loss 곡선 확인, 마지막에 출력된 생성 샘플 읽기, 확인 문제 10–12, 해설 12의 스니펫으로 MFU 측정
- 완료 기준: 직접 구현한 GPT가 TinyStories 문체의 이야기를 생성
- 학습 결과가 이상하면 같은 설정을 `MINILLM_SOLUTIONS=1`로 돌려 비교한다. 정답 쪽이 정상이면 본인 구현 문제이므로, `MINILLM_ONLY`로 단계를 하나씩 교체하며 원인 단계를 찾는다.

**6주차 — 버퍼**
- [ ] 밀린 단계 마무리
- [ ] 1–5주차 확인 문제 중 틀린 것 다시 풀기
- [ ] 정답을 봤던 구간을 보지 않고 다시 작성
- [ ] 선택 과제(밀린 것이 없을 때): step 21 FlashAttention 방식 attention (step 06의 확장, 약 2시간)

### 4부. 디코딩 (7주)

**7주차 — 샘플링, KV cache** (step 13–14)
- [ ] S1: `docs/13` 읽기, `filter_probs` (top-k, top-p 경계 사례)
- [ ] S2: `sample_next`, `generate` 샘플링 구간 → `test_13`
- [ ] S3: `docs/14` 읽기, `KVCache.update`, `kv_cache_bytes` → `test_14`
- [ ] W: `python scripts/bench_inference.py`의 KV cache 부분 확인, 확인 문제 13·14, 11주차를 위해 `python scripts/pretrain.py --preset draft` 실행 (15–30분 추정)
- 완료 기준: cache 유무와 무관하게 같은 출력, cache 사용 시 처리량 증가 확인

### 5부. 정렬 (8–10주)

**8주차 — SFT, reward model** (step 15–16)
- [ ] S1: `docs/15` 읽기, `build_sft_example` → `test_15`
- [ ] S2: `python scripts/sft.py` 실행 (10–20분 추정), 학습 전후 키워드 적중률과 샘플 확인
- [ ] S3: `docs/16` 읽기, `bradley_terry_loss`(z=0 gradient 주의), `RewardModel` → `test_16`
- [ ] W: `docs/17` 2.1–2.4의 DPO 유도를 보지 않고 종이에 재현, 확인 문제 15·16
- 완료 기준: SFT 모델이 키워드 지시를 따르는 비율이 사전학습 모델보다 높음

**9주차 — DPO** (step 17)
- [ ] S1: `token_logprobs`, `sequence_logprobs`
- [ ] S2: `dpo_loss` → `test_17`
- [ ] S3: `python scripts/dpo.py` 시작 (rollout 포함 20–40분 추정)
- [ ] W: `python scripts/compare.py`로 SFT와 DPO 비교(적중률, 길이, KL, val loss), `--beta`를 바꿔 한 번 더 실행, 확인 문제 17
- 완료 기준: DPO 후 적중률 변화와 그 대가(길이, KL)를 표로 설명할 수 있음

**10주차 — GRPO** (step 18)
- [ ] S1: `docs/18` 2.1–2.4 읽기, `group_advantages`
- [ ] S2: 2.5–2.6 읽기, `ppo_clip_loss`, `kl_k3`
- [ ] S3: `test_18` 통과, 잠들기 전 `caffeinate -i python scripts/grpo.py` 시작 (30–60분 추정)
- [ ] W: `compare.py`로 SFT/DPO/GRPO 비교, `--kl` 계수를 바꿔 재실행, 확인 문제 18(k3 gradient 유도 포함)
- 완료 기준: DPO(오프라인)와 GRPO(온라인)의 결과 차이를 KL·val loss와 함께 설명할 수 있음

### 6부. 서빙 최적화 (11주)

**11주차 — 양자화, speculative decoding** (step 19–20)
- [ ] S1: `docs/19` 읽기, `quantize`, `dequantize` → `test_19`
- [ ] S2: `bench_inference.py`의 양자화 부분 실행, 정밀도별 perplexity·메모리 비교, 확인 문제 19
- [ ] S3: `docs/20` 읽기(2.3 증명 포함), `speculative_accept`
- [ ] W: `speculative_generate`(cache rollback 불변식) → `test_20`, `python scripts/speculative.py`, 확인 문제 20
- 완료 기준: greedy speculative 출력이 타깃 단독 greedy와 일치, 라운드당 확정 토큰 수를 이론값과 비교

**12주차 — 버퍼, 본 과정 종합**
- [ ] 밀린 단계 마무리
- [ ] 본인 구현으로 전체 파이프라인을 처음부터 다시 실행: `prepare_data` → `pretrain` → `sft` → `dpo`/`grpo` → `compare` → `bench_inference` → `speculative`
- [ ] 7–11주차 확인 문제 복습
- [ ] 회고: 구현이 가장 어려웠던 단계 3개와 그 이유를 기록

### 7부. 심화 (13–14주)

**13주차 — FlashAttention, GQA** (step 21–22)
- [ ] S1: `docs/21` 읽기, `online_logsumexp` (6주차에 했다면 S1–S2를 step 22에 쓴다)
- [ ] S2: `flash_attention` → `test_21`
- [ ] S3: `docs/22` 읽기, `repeat_kv`, `GroupedQueryAttention.forward` → `test_22`
- [ ] 잠들기 전: `caffeinate -i python scripts/pretrain.py --preset m5 --n_kv_head 2 --out out/pretrain_gqa.pt`
- [ ] W: `out/pretrain.csv`와 `out/pretrain_gqa.csv` 비교, `bench_inference.py --ckpt out/pretrain_gqa.pt`로 KV cache 크기 비교, 확인 문제 21·22

**14주차 — mixed precision, checkpointing, LoRA** (step 23–24)
- [ ] S1: `docs/23` 2절 읽기, `bf16_round`(비트 연산 유도), `LossScaler`
- [ ] S2: `train_step_amp`
- [ ] S3: `docs/23` 3절 읽기, `CheckpointFunction` → `test_23`
- [ ] W: step 24 LoRA → `test_24`, `python scripts/sft.py --lora_r 8 --lr 1e-3 --out out/sft_lora.pt`, `compare.py --ref out/pretrain.pt`로 전체 SFT와 비교, `--dtype bf16 --grad_ckpt`로 짧은 학습을 돌려 tok/s 비교, 확인 문제 23·24

## 느린 트랙 (주 4시간)

같은 순서로 진행하되 한 주 분량을 1.5주에 나눠 한다. 평일 세션이 2회이므로 S1–S3을 2주에 걸쳐 진행하고, 주말 세션 W는 다음 주 주말로 넘긴다.

| 기본 트랙 | 느린 트랙 |
|---|---|
| 1–2주 (기초) | 1–3주 |
| 3–4주 (토크나이저, Transformer) | 4–6주 |
| 5주 (프리트레이닝) | 7–8주 |
| 6주 (버퍼) | 9주 |
| 7주 (디코딩) | 10–11주 |
| 8–10주 (정렬) | 12–16주 |
| 11주 (서빙) | 17–18주 |
| 12주 (버퍼, 종합) | 19주 |
| 13–14주 (심화) | 20–22주 |

## 단계별 예상 소요 시간 (추정)

| Step | 학습·구현 | 실행 (M5, 추정) | Step | 학습·구현 | 실행 (M5, 추정) |
|---|---|---|---|---|---|
| 01 | 2h | | 13 | 2h | |
| 02 | 1.5h | | 14 | 1.5h | 수 분 |
| 03 | 2.5h | | 15 | 1.5h | 10–20분 |
| 04 | 2h | | 16 | 1.5h | |
| 05 | 3h | 수 분–수십 분 | 17 | 2.5h | 20–40분 |
| 06 | 1h | | 18 | 3h | 30–60분 |
| 07 | 2h | | 19 | 1.5h | 수 분 |
| 08 | 2h | | 20 | 2h | 15–30분 (draft) |
| 09 | 1.5h | | 21 | 2h | |
| 10 | 2h | | 22 | 1.5h | 1–2시간 (선택) |
| 11 | 1h | | 23 | 3h | |
| 12 | 1.5h | 1–2시간 | 24 | 1.5h | 10–20분 |

학습·구현 시간은 문서 읽기, 구현, 테스트 통과, 확인 문제 풀이를 합친 값이다. 이론을 이미 아는 전공자 기준으로 잡았으므로, 처음 보는 주제(예: speculative decoding의 증명, GRPO의 KL 추정량)는 1.5배 정도 여유를 둔다.

## 진도 확인

완료한 단계까지 한 번에 채점한다. 예: 5주차 종료 시점

```bash
pytest tests/test_0*.py tests/test_1[0-2]*.py -q
```
