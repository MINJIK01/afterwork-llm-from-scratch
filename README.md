# afterwork-llm-from-scratch — 퇴근 후 LLM을 처음부터 구현하는 실습 프로젝트

논문의 수식을 코드로 옮기는 연습을 목표로, 작은 GPT를 **autograd부터 프리트레이닝, 정렬(SFT·DPO·GRPO), 서빙 최적화(KV cache·양자화·speculative decoding)까지** 직접 구현하는 20단계 커리큘럼과, FlashAttention·GQA·mixed precision·LoRA를 다루는 심화 4단계로 구성된다. 모든 단계는 Apple Silicon M5 Mac(PyTorch MPS)에서 돌릴 수 있는 크기(약 12M 파라미터)로 설계했다.

> 퇴근 후 공부하는 일정으로 진행하려면 학습 계획을 따른다.
> - [docs/STUDY_PLAN.md](docs/STUDY_PLAN.md): 평일 3회 × 60분 + 주말 3시간, 14주 (주 4시간 느린 트랙 22주)
> - [docs/STUDY_PLAN_30MIN.md](docs/STUDY_PLAN_30MIN.md): 30분 세션 106개, 세션마다 통과할 테스트 지정 (평일 1세션이면 약 21주, 매일 1세션이면 약 15주)

## 구조

```
docs/NN_*.md          단계별 이론 (유도, 구현 가이드, 확인 문제, 참고문헌)
docs/answers/NN_*.md  확인 문제 해설 (먼저 직접 풀어 본 뒤 확인)
minillm/              실습 대상 패키지. # BEGIN TODO ~ # END TODO 구간을 구현
tests/test_NN_*.py    단계별 자동 채점 (PyTorch 레퍼런스/autograd와 비교)
scripts/              데이터 준비, 학습, 정렬, 벤치마크 실행 스크립트
solutions/minillm/    정답 구현 (막혔을 때만 참고)
tools/hybrid.py       채점 대상 패키지 선택 (기본 / 정답 / 단독 채점)
tools/make_stubs.py   solutions에서 minillm 스텁을 다시 생성 (minillm 덮어씀)
```

## 진행 방법

1. `docs/NN_*.md`를 읽는다.
2. `minillm/`에서 해당 step의 TODO 구간을 찾아, `raise NotImplementedError(...)` 줄을 지우고 그 자리에 구현한다.
3. `pytest tests/test_NN_*.py -v`로 채점한다.
4. 확인 문제를 풀고 `docs/answers/NN_*.md`로 확인한다.
5. 막히면 `solutions/minillm/`의 같은 위치(`# BEGIN SOLUTION (step NN)`)를 참고한다.

```bash
grep -rn "BEGIN TODO (step 05)" minillm/
```

구현은 반드시 마커 사이에 작성하고 마커 줄은 지우지 않는다. 단독 채점이 이 마커로 구현 위치를 찾는다.

```python
        # BEGIN TODO (step 01)
        self.grad += out.grad      # ← 여기에 구현
        other.grad += out.grad
        # END TODO
```

### 앞 단계 없이 한 단계만 채점하기

뒤 단계는 앞 단계 구현을 사용한다(예: step 06 attention은 step 02의 `softmax`를 쓴다). 순서대로 진행하는 것이 기본이지만, `MINILLM_ONLY`를 지정하면 그 step의 TODO 구간만 본인 코드로, 나머지는 정답으로 채운 패키지(`.grade/`에 자동 생성)로 채점한다.

```bash
MINILLM_ONLY=08 pytest tests/test_08_causal_attention.py -v
```

여러 step을 함께 지정할 수도 있다.

```bash
MINILLM_ONLY=6,7,8 pytest tests/test_06_attention.py tests/test_07_rope.py tests/test_08_causal_attention.py -v
```

스크립트에도 같은 방식이 적용된다. 예를 들어 step 12만 구현한 상태로 실제 학습을 돌려 볼 수 있다.

```bash
MINILLM_ONLY=12 python scripts/pretrain.py --preset tiny
```

| 환경 변수 | 사용되는 코드 |
|---|---|
| (없음) | `minillm/` 전체 (본인 구현) |
| `MINILLM_ONLY=NN[,MM...]` | 지정 step만 `minillm/`, 나머지는 `solutions/` |
| `MINILLM_SOLUTIONS=1` | `solutions/` 전체 (정답) |

## 커리큘럼

| Step | 주제 | 구현 위치 | 실행 |
|---|---|---|---|
| **기초** | | | |
| 01 | Reverse-mode autograd | `autograd.py` | |
| 02 | Softmax / log-softmax / cross-entropy와 역전파 | `functional.py` | |
| 03 | Linear·LayerNorm 해석적 역전파 | `functional.py` | |
| 04 | AdamW, warmup+cosine, gradient clipping | `optim.py` | |
| **토크나이저** | | | |
| 05 | Byte-level BPE | `tokenizer.py` | `prepare_data.py` |
| **Transformer** | | | |
| 06 | Scaled dot-product attention | `layers.py` | |
| 07 | RoPE | `layers.py` | |
| 08 | Multi-head causal self-attention | `layers.py` | |
| 09 | RMSNorm, SwiGLU, pre-norm block | `layers.py` | |
| 10 | GPT 조립, weight tying, 초기화 | `model.py` | |
| **프리트레이닝** | | | |
| 11 | 문서 packing, 배치 샘플링 | `data.py` | `prepare_data.py` |
| 12 | 학습 루프, gradient accumulation, 평가 | `trainer.py` | `pretrain.py` |
| **디코딩** | | | |
| 13 | Temperature, top-k, nucleus sampling | `generate.py` | |
| 14 | KV cache | `layers.py` | `bench_inference.py` |
| **정렬** | | | |
| 15 | SFT (프롬프트 loss 마스킹) | `align.py` | `sft.py` |
| 16 | Bradley–Terry, reward model | `align.py` | |
| 17 | DPO | `align.py` | `dpo.py` |
| 18 | Policy gradient, PPO-clip, GRPO | `align.py` | `grpo.py` |
| **서빙 최적화** | | | |
| 19 | Weight-only 양자화 (int8/int4) | `quant.py` | `bench_inference.py` |
| 20 | Speculative decoding | `generate.py` | `speculative.py` |
| **심화** (괄호 안 단계 직후에 진행해도 된다) | | | |
| 21 | Online softmax, FlashAttention 방식 블록 attention (06) | `flash.py` | |
| 22 | Grouped-Query / Multi-Query Attention (14) | `layers.py` | `pretrain.py --n_kv_head` |
| 23 | Mixed precision(bf16, fp16 loss scaling), gradient checkpointing (12) | `precision.py` | `pretrain.py --dtype --grad_ckpt` |
| 24 | LoRA (15) | `lora.py` | `sft.py --lora_r` |

## 환경 설정 (Mac)

Python 3.11 이상.

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

macOS arm64용 PyTorch에는 MPS 백엔드가 포함되어 있다. 스크립트는 MPS → CUDA → CPU 순으로 장치를 자동 선택한다.

## 전체 파이프라인

각 스크립트는 기본적으로 `minillm/`(본인 구현)을 사용한다. 정답 구현으로 먼저 결과를 보고 싶으면 명령 앞에 `MINILLM_SOLUTIONS=1`을 붙인다(테스트도 동일).

```bash
python scripts/prepare_data.py --dataset tinystories
```

TinyStories 앞부분 200MB를 내려받아 BPE(vocab 4096)를 학습하고 `data/train.bin`, `data/val.bin`을 만든다. 순수 Python BPE라 수 분이 걸린다.

```bash
python scripts/pretrain.py --preset m5
```

12.2M 파라미터 모델을 약 65M 토큰으로 학습해 `out/pretrain.pt`에 저장한다. 학습 곡선은 `out/pretrain.csv`(step, train loss, lr, grad norm, tok/s, val loss)에 기록된다. 중간에 끊기면 같은 명령에 `--resume`을 붙여 마지막 평가 시점부터 이어서 학습한다(모델·optimizer·데이터 순서까지 복원되어, 끊김 없이 학습한 것과 같은 결과가 나온다).

```bash
python scripts/sft.py
```

```bash
python scripts/dpo.py
```

```bash
python scripts/grpo.py
```

SFT 후 DPO와 GRPO는 각각 `out/sft.pt`에서 시작하는 독립 실험이며, 평가 키워드 적중률을 학습 전후로 출력한다. 결과는 한 표로 비교한다.

```bash
python scripts/compare.py
```

`out/`에 있는 pretrain/sft/sft_lora/dpo/grpo 체크포인트마다 키워드 적중률, EOS 종료율, 평균 응답 길이, SFT 대비 KL(π‖ref), 사전학습 val loss를 출력한다. 적중률 상승이 길이 증가나 언어 모델링 능력 손실과 함께 왔는지 확인하는 용도이다.

```bash
python scripts/bench_inference.py
```

```bash
python scripts/pretrain.py --preset draft
```

```bash
python scripts/speculative.py
```

### 심화 단계 옵션 (step 22–24)

`pretrain.py`에 `--n_kv_head N`(GQA), `--dtype bf16|fp16`(mixed precision), `--grad_ckpt`(gradient checkpointing)를 붙일 수 있다. MPS에서 bf16이 지원되지 않으면 `--dtype fp16`을 쓴다.

```bash
python scripts/pretrain.py --preset m5 --n_kv_head 2 --dtype bf16 --grad_ckpt --out out/pretrain_gqa.pt
```

```bash
python scripts/sft.py --lora_r 8 --lr 1e-3 --out out/sft_lora.pt
```

### 검증 상태와 소요 시간

- 이 레포는 Linux CPU 환경에서 TinyShakespeare와 `--preset tiny`로 전체 파이프라인(데이터 준비 → 프리트레이닝 → SFT → DPO → GRPO → 벤치마크 → speculative decoding)과 심화 옵션(GQA, bf16/fp16, checkpointing, LoRA SFT)이 끝까지 동작함을 확인했고, 정답 구현으로 84개 테스트가 모두 통과한다.
- mixed precision은 CPU autocast로만 검증했다. MPS의 bf16/fp16 autocast 동작은 실측하지 않았다.
- M5 Mac에서의 소요 시간은 실측하지 않았다. FLOPs 기준 추정으로 m5 프리트레이닝은 1~2시간 수준이며, 실제 처리량은 로그의 `tok/s`로 확인하고 `--max_steps`로 조정한다.
- CPU에서 빠르게 흐름만 확인하려면:

```bash
python scripts/prepare_data.py --dataset shakespeare --vocab_size 1024 --bpe_train_mb 1
```

```bash
python scripts/pretrain.py --preset tiny
```

## 정렬 실험 과제

TinyStories에는 지시 데이터와 선호 데이터가 없으므로, 말뭉치에서 자동으로 만든 과제를 쓴다.

- 지시: `<|user|>Write a story that uses the word "ball".<|assistant|>`
- SFT 데이터: 이야기 속 단어 하나를 키워드로 뽑아 (지시, 원래 이야기) 쌍을 만든다.
- 보상: 응답이 EOS로 끝나고 키워드를 단어 단위로 포함하면 1, 아니면 0 (`scripts/common.py`의 `reward`).
- 평가: 학습 키워드 풀에서 제외한 20개 키워드의 적중률.

보상을 규칙으로 정확히 계산할 수 있어, 학습된 reward model의 오차 없이 DPO와 GRPO의 효과를 비교할 수 있다.
