# ClaudePlayground — LLM을 처음부터 구현하는 실습 프로젝트

논문의 수식을 코드로 옮기는 연습을 목표로, 작은 GPT를 **autograd부터 프리트레이닝, 정렬(SFT·DPO·GRPO), 서빙 최적화(KV cache·양자화·speculative decoding)까지** 직접 구현하는 20단계 커리큘럼이다. 모든 단계는 Apple Silicon M5 Mac(PyTorch MPS)에서 돌릴 수 있는 크기(약 12M 파라미터)로 설계했다.

## 구조

```
docs/NN_*.md          단계별 이론 (유도, 구현 가이드, 확인 문제, 참고문헌)
minillm/              실습 대상 패키지. TODO 위치에 raise NotImplementedError
tests/test_NN_*.py    단계별 자동 채점 (PyTorch 레퍼런스/autograd와 비교)
scripts/              데이터 준비, 학습, 정렬, 벤치마크 실행 스크립트
solutions/minillm/    정답 구현 (막혔을 때만 참고)
tools/make_stubs.py   solutions에서 minillm 스텁을 다시 생성 (minillm 덮어씀)
```

## 진행 방법

1. `docs/NN_*.md`를 읽는다.
2. `minillm/`에서 `NotImplementedError("TODO: step NN ...")`를 찾아 구현한다.
3. `pytest tests/test_NN_*.py -v`로 채점한다.
4. 막히면 `solutions/minillm/`의 같은 위치(`# BEGIN SOLUTION (step NN)`)를 참고한다.

뒤 단계는 앞 단계 구현을 사용하므로 순서대로 진행한다(예: step 06 attention은 step 02의 `softmax`를 쓴다).

```bash
grep -rn "TODO: step 05" minillm/
```

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

12.2M 파라미터 모델을 약 65M 토큰으로 학습해 `out/pretrain.pt`에 저장한다.

```bash
python scripts/sft.py
```

```bash
python scripts/dpo.py
```

```bash
python scripts/grpo.py
```

SFT 후 DPO와 GRPO는 각각 `out/sft.pt`에서 시작하는 독립 실험이며, 평가 키워드 적중률을 학습 전후로 출력한다.

```bash
python scripts/bench_inference.py
```

```bash
python scripts/pretrain.py --preset draft
```

```bash
python scripts/speculative.py
```

### 검증 상태와 소요 시간

- 이 레포는 Linux CPU 환경에서 TinyShakespeare와 `--preset tiny`로 전체 파이프라인(데이터 준비 → 프리트레이닝 → SFT → DPO → GRPO → 벤치마크 → speculative decoding)이 끝까지 동작함을 확인했고, 정답 구현으로 64개 테스트가 모두 통과한다.
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
