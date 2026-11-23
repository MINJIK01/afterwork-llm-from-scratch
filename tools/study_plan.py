"""30분 세션 학습 계획의 단일 명세.

이 파일의 세션 목록에서 두 가지를 생성한다.
- docs/STUDY_PLAN_30MIN.md      (python tools/study_plan.py md)
- docs/tracker/index.html       (python tools/study_plan.py tracker, tools/tracker_template.html 사용)
인자 없이 실행하면 둘 다 생성한다. 계획을 고칠 때는 이 파일만 고치고 다시 생성한다.
구현 세션의 완료 기준(테스트)은 '직전 세션까지 구현한 상태에서 실패, 해당 함수 구현 후 통과'를 만족하도록 순서가 정해져 있다.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# s(step, 유형, 할 일, 구현 대상["file:Qual"], 테스트 파일, -k 식(None이면 파일 전체), 테스트가 없을 때의 완료 기준)
P = []
def s(step, kind, task, impl=(), test=None, k=None, done=None):
    P.append(dict(step=step, kind=kind, task=task, impl=list(impl), test=test, k=k, done=done))

PARTS = []
def part(title, note=""):
    PARTS.append((len(P), title, note))

part("0. 준비")
s("--", "준비", "README의 환경 설정을 따라 venv와 패키지 설치, `docs/STUDY_PLAN.md`의 세션 운영 규칙 읽기, 본인 구현용 브랜치 생성(`git switch -c my-impl`)",
  done="`MINILLM_SOLUTIONS=1 pytest tests -q` → 84 passed")

part("1부. 기초 (step 01–04)")
s("01", "읽기", "`docs/01` §2 전체. §2.2의 국소 미분 표를 종이에 옮겨 적기", done="표를 보지 않고 7개 연산의 국소 미분을 쓸 수 있음")
s("01", "구현", "`Value.__add__`, `Value.__mul__`의 `_backward`, `Value.backward`(위상 정렬)", ["autograd:Value.__add__", "autograd:Value.__mul__", "autograd:Value.backward"],
  "test_01_autograd.py", "reused or deep_chain")
s("01", "구현", "`__pow__`, `exp`, `log`, `tanh`, `relu`의 `_backward`", ["autograd:Value.__pow__", "autograd:Value.exp", "autograd:Value.log", "autograd:Value.tanh", "autograd:Value.relu"],
  "test_01_autograd.py", None)
s("01", "문제", "확인 문제 01 풀기 → `docs/answers/01` 대조", done="틀린 문항의 이유를 한 줄로 메모")
s("02", "구현", "`docs/02` §2.1–2.2 읽고 `softmax`, `log_softmax`", ["functional:softmax", "functional:log_softmax"],
  "test_02_softmax_xent.py", "softmax_and_log or stability")
s("02", "구현", "§2.3 읽고 `cross_entropy` (ignore_index 마스킹)", ["functional:cross_entropy"], "test_02_softmax_xent.py", "ignore_index")
s("02", "구현", "§2.4 유도를 종이로 따라간 뒤 `cross_entropy_backward`", ["functional:cross_entropy_backward"], "test_02_softmax_xent.py", None)
s("02", "문제", "확인 문제 02 → 해설 대조", done="해설 대조 완료")
s("03", "구현", "`docs/03` §2.1–2.2 읽고 `linear_backward`", ["functional:linear_backward"], "test_03_manual_backward.py", "linear")
s("03", "구현", "§2.3 읽고 `layer_norm_forward` (cache 구성 결정)", ["functional:layer_norm_forward"], "test_03_manual_backward.py", "forward")
s("03", "읽기", "§2.4의 1–3단계 유도를 종이로: $\\partial\\sigma/\\partial x_j = \\hat x_j/D$까지", done="보지 않고 3단계까지 재현")
s("03", "구현", "§2.4의 4–5단계 유도 → `layer_norm_backward`", ["functional:layer_norm_backward"], "test_03_manual_backward.py", None)
s("03", "문제", "확인 문제 03 → 해설 대조", done="해설 대조 완료")
s("04", "읽기", "`docs/04` §2.1–2.3. bias correction 유도를 종이로", done="$\\mathbb E[m_t] = (1-\\beta_1^t)\\mathbb E[g]$를 보지 않고 유도")
s("04", "구현", "`AdamW.step`", ["optim:AdamW.step"], "test_04_optim.py", "adamw")
s("04", "구현", "§2.4–2.5 읽고 `cosine_lr`, `clip_grad_norm_`", ["optim:cosine_lr", "optim:clip_grad_norm_"], "test_04_optim.py", None)
s("04", "문제", "확인 문제 04 → 해설 대조", done="해설 대조 완료")

part("2부. 토크나이저와 Transformer (step 05–09)")
s("05", "읽기", "`docs/05` §2.1–2.3. `aaabdaaabac` 예제를 손으로 3번 병합", done="병합 결과 `Z=aa, Y=ab, X=ZY`를 직접 도출")
s("05", "구현", "`get_pair_counts`, `merge`", ["tokenizer:get_pair_counts", "tokenizer:merge"], "test_05_bpe.py", "pair_counts")
s("05", "구현", "`BPETokenizer.train` (동률 규칙 포함)", ["tokenizer:BPETokenizer.train"], "test_05_bpe.py", "wikipedia")
s("05", "구현", "`_encode_chunk`, `decode`", ["tokenizer:BPETokenizer._encode_chunk", "tokenizer:BPETokenizer.decode"], "test_05_bpe.py", None)
s("05", "실행", "`python scripts/prepare_data.py` 시작 (수 분~수십 분, 추정). 도는 동안 §2.4–2.7 읽기", done="`data/train.bin` 생성, 출력된 bytes/token 기록")
s("05", "문제", "확인 문제 05 → 해설 대조", done="해설 대조 완료")
s("06", "구현", "`docs/06` §2.1–2.3 읽고 `scaled_dot_product_attention`", ["layers:scaled_dot_product_attention"], "test_06_attention.py", None)
s("06", "문제", "§2.4–2.5 읽기, 확인 문제 06 → 해설 대조", done="해설 대조 완료")
s("07", "읽기", "`docs/07` §2.1–2.3. 2차원 회전으로 상대 위치 성질을 종이로 유도", done="$R_m^\\top R_n = R_{n-m}$을 보지 않고 보임")
s("07", "구현", "`precompute_rope`, `apply_rope`", ["layers:precompute_rope", "layers:apply_rope"], "test_07_rope.py", None)
s("07", "문제", "§2.4–2.5 읽기, 확인 문제 07 → 해설 대조", done="해설 대조 완료")
s("08", "읽기", "`docs/08` §2.1–2.4. $T=1$, $T_k=5$일 때 오프셋 마스크를 종이에 그리기", done="`tril(diagonal=Tk-T)`의 결과를 손으로 씀")
s("08", "구현", "`CausalSelfAttention.forward` 두 구간 (QKV 분리·head 분할·RoPE / 마스크·attention·출력 투영)", ["layers:CausalSelfAttention.forward"],
  "test_08_causal_attention.py", None)
s("08", "문제", "확인 문제 08 → 해설 대조", done="해설 대조 완료")
s("09", "구현", "`docs/09` §2.1–2.3 읽고 `RMSNorm.forward`", ["layers:RMSNorm.forward"], "test_09_block.py", "rmsnorm")
s("09", "구현", "§2.4 읽고 `SwiGLU.forward`, `Block.forward`", ["layers:SwiGLU.forward", "layers:Block.forward"], "test_09_block.py", None)
s("09", "문제", "확인 문제 09 → 해설 대조", done="해설 대조 완료")

part("3부. 프리트레이닝 (step 10–12)", "이 부의 마지막에 처음으로 1–2시간짜리 학습을 돌린다.")
s("10", "읽기", "`docs/10` §2.1–2.3. residual 분산 $\\mathrm{Var}(x_L) = \\mathrm{Var}(x_0) + 2L\\sigma_f^2$ 유도", done="초기화 스케일 $0.02/\\sqrt{2L}$의 근거를 설명할 수 있음")
s("10", "구현", "`GPT.__init__`의 weight tying과 초기화", ["model:GPT.__init__"], "test_10_gpt.py", "tying")
s("10", "구현", "`GPT.hidden`, `GPT.forward`", ["model:GPT.hidden", "model:GPT.forward"], "test_10_gpt.py", None)
s("10", "문제", "§2.4–2.6 읽기, 확인 문제 10 → 해설 대조", done="해설 대조 완료")
s("11", "구현", "`docs/11` 읽고 `pack_documents`, `get_batch`", ["data:pack_documents", "data:get_batch"], "test_11_data.py", None)
s("11", "문제", "확인 문제 11 → 해설 대조", done="해설 대조 완료")
s("12", "구현", "`docs/12` §2.1–2.4 읽고 `train_step`", ["trainer:train_step"], "test_12_training.py", "accumulation or overfit")
s("12", "구현", "`evaluate` 구현 후 `python scripts/pretrain.py --preset tiny`로 학습 흐름 확인 (수 분)", ["trainer:evaluate"], "test_12_training.py", None)
s("12", "실행", "잠들기 전 `caffeinate -i python scripts/pretrain.py --preset m5` 시작 (5분이면 끝나는 세션, 학습은 1–2시간 추정)", done="첫 로그 줄 확인 후 종료")
s("12", "실행", "`out/pretrain.csv`로 train/val loss 곡선 확인, 생성 샘플 읽기, 해설 12의 스니펫으로 MFU 측정", done="val loss, 샘플 1개, MFU 기록")
s("12", "문제", "§2.5–2.6 읽기, 확인 문제 12 → 해설 대조", done="해설 대조 완료")
for i in range(4):
    s("--", "버퍼", ["밀린 세션 처리", "1–3부 확인 문제 중 틀린 것 다시 풀기", "정답을 봤던 구간을 보지 않고 다시 작성", "선택: step 21 (심화 첫 세션부터)"][i], done="—")

part("4부. 디코딩 (step 13–14)")
s("13", "구현", "`docs/13` §2.1–2.2 읽고 `sample_next` (greedy 분기와 multinomial 분기)", ["generate:sample_next"], "test_13_sampling.py", "greedy")
s("13", "구현", "§2.3–2.5 읽고 `filter_probs` (top-k, top-p 경계 사례)", ["generate:filter_probs"], "test_13_sampling.py", "top_k or top_p or frequency")
s("13", "구현", "§2.6 읽고 `generate`의 샘플링 구간", ["generate:generate"], "test_13_sampling.py", None)
s("13", "문제", "확인 문제 13 → 해설 대조", done="해설 대조 완료")
s("14", "구현", "`docs/14` §2.1–2.5 읽고 `KVCache.update`", ["layers:KVCache.update"], "test_14_kv_cache.py", "not bytes")
s("14", "구현", "`kv_cache_bytes` 구현 후 `python scripts/bench_inference.py`의 KV cache 부분 확인", ["layers:kv_cache_bytes"], "test_14_kv_cache.py", None)
s("14", "실행", "확인 문제 14 → 해설 대조, 이어서 `python scripts/pretrain.py --preset draft` 시작 (15–30분 추정, step 20용)", done="`out/draft.pt` 생성")

part("5부. 정렬 (step 15–18)")
s("15", "구현", "`docs/15` 읽고 `build_sft_example`", ["align:build_sft_example"], "test_15_sft.py", None)
s("15", "실행", "`python scripts/sft.py` (10–20분 추정), 학습 전후 키워드 적중률과 샘플 확인", done="전후 적중률 기록")
s("15", "문제", "확인 문제 15 → 해설 대조", done="해설 대조 완료")
s("16", "구현", "`docs/16` §2.1–2.3 읽고 `bradley_terry_loss` ($z=0$ gradient 주의)", ["align:bradley_terry_loss"], "test_16_reward_model.py", "bradley")
s("16", "구현", "§2.4–2.5 읽고 `RewardModel.forward`", ["align:RewardModel.forward"], "test_16_reward_model.py", None)
s("16", "문제", "확인 문제 16 (3번은 직접 실행) → 해설 대조", done="해설 대조 완료")
s("17", "읽기", "`docs/17` §2.1–2.2. 최적 정책의 닫힌 형태를 보지 않고 유도", done="$\\pi^* \\propto \\pi_{ref}e^{r/\\beta}$ 유도")
s("17", "읽기", "§2.3–2.5. BT에 대입해 $Z(x)$ 소거, gradient 식 해석", done="DPO loss를 보지 않고 씀")
s("17", "구현", "`token_logprobs`, `sequence_logprobs`", ["align:token_logprobs", "align:sequence_logprobs"], "test_17_dpo.py", "logprobs")
s("17", "구현", "`dpo_loss`", ["align:dpo_loss"], "test_17_dpo.py", None)
s("17", "실행", "`python scripts/dpo.py` (20–40분 추정)", done="reward acc와 학습 후 적중률 기록")
s("17", "실행", "`python scripts/compare.py`로 SFT와 DPO 비교, §2.6–2.8 읽기", done="적중률·길이·KL·val loss 변화 기록")
s("17", "문제", "확인 문제 17 → 해설 대조", done="해설 대조 완료")
s("18", "읽기", "`docs/18` §2.1–2.3. REINFORCE와 baseline의 비편향성 유도", done="$\\mathbb E[b\\nabla\\log\\pi]=0$을 보임")
s("18", "구현", "§2.4 읽고 `group_advantages`", ["align:group_advantages"], "test_18_grpo.py", "group")
s("18", "구현", "§2.5 읽고 `ppo_clip_loss` (표의 네 경우 확인)", ["align:ppo_clip_loss"], "test_18_grpo.py", "ppo")
s("18", "구현", "§2.6 읽고 $k_3$ 비편향성 유도 후 `kl_k3`", ["align:kl_k3"], "test_18_grpo.py", None)
s("18", "실행", "잠들기 전 `caffeinate -i python scripts/grpo.py` 시작 (30–60분 추정)", done="첫 iteration 로그 확인")
s("18", "실행", "`compare.py`로 SFT/DPO/GRPO 비교, §2.7–2.8 읽기", done="세 방식의 차이 기록")
s("18", "문제", "확인 문제 18 (3번 $k_3$ gradient 유도 포함) → 해설 대조", done="해설 대조 완료")

part("6부. 서빙 최적화 (step 19–20)")
s("19", "구현", "`docs/19` §2.1–2.3 읽고 `quantize`, `dequantize`", ["quant:quantize", "quant:dequantize"], "test_19_quantization.py", None)
s("19", "실행", "`python scripts/bench_inference.py`의 양자화 부분, §2.4–2.5 읽기", done="정밀도별 perplexity·메모리 기록")
s("19", "문제", "확인 문제 19 → 해설 대조", done="해설 대조 완료")
s("20", "읽기", "`docs/20` §2.1–2.4. 출력 분포 = $p$ 증명을 보지 않고 재현", done="증명 재현")
s("20", "구현", "`speculative_accept`", ["generate:speculative_accept"], "test_20_speculative.py", "accept")
s("20", "구현", "§2.6 읽고 `speculative_generate` (cache rollback 불변식)", ["generate:speculative_generate"], "test_20_speculative.py", None)
s("20", "실행", "`python scripts/speculative.py`, 확인 문제 20 → 해설 대조", done="라운드당 확정 토큰 수를 이론값과 비교")
for t in ["밀린 세션 처리", "밀린 세션 처리 / 4–6부 확인 문제 복습",
          "종합 1: 본인 구현으로 `prepare_data` → 잠들기 전 `pretrain --preset m5` 시작",
          "종합 2: `sft` → `dpo` → `grpo` (각각 별도 세션으로 나눠도 됨)",
          "종합 3: `compare` → `bench_inference` → `speculative`", "회고: 가장 어려웠던 단계 3개와 이유 기록"]:
    s("--", "버퍼", t, done="—")

part("7부. 심화 (step 21–24)")
s("21", "읽기", "`docs/21` §2.1–2.3. online softmax 갱신식의 불변식 확인", done="$a/l$이 정확한 이유를 설명")
s("21", "구현", "`online_logsumexp`", ["flash:online_logsumexp"], "test_21_flash.py", "logsumexp")
s("21", "구현", "`flash_attention`", ["flash:flash_attention"], "test_21_flash.py", None)
s("21", "문제", "§2.4–2.6 읽기, 확인 문제 21 → 해설 대조", done="해설 대조 완료")
s("22", "구현", "`docs/22` 읽고 `repeat_kv`", ["layers:repeat_kv"], "test_22_gqa.py", "repeat")
s("22", "구현", "`GroupedQueryAttention.forward` 두 구간. 잠들기 전 선택: `pretrain.py --preset m5 --n_kv_head 2 --out out/pretrain_gqa.pt`", ["layers:GroupedQueryAttention.forward"],
  "test_22_gqa.py", None)
s("22", "문제", "CSV 비교, `bench_inference.py --ckpt out/pretrain_gqa.pt`, 확인 문제 22 → 해설 대조", done="KV cache 크기·val loss 비교 기록")
s("23", "구현", "`docs/23` §2.1–2.2 읽고 `bf16_round` (비트 연산)", ["precision:bf16_round"], "test_23_precision.py", "bf16_round")
s("23", "구현", "§2.5 읽고 `LossScaler.unscale_`, `LossScaler.update`", ["precision:LossScaler.unscale_", "precision:LossScaler.update"], "test_23_precision.py", "scaler")
s("23", "구현", "§2.3–2.4 읽고 `train_step_amp`", ["precision:train_step_amp"], "test_23_precision.py", "amp")
s("23", "읽기", "§3.1–3.3 (학습 메모리 구성, 재계산, custom autograd Function)", done="checkpoint가 저장하는 것과 버리는 것을 설명")
s("23", "구현", "`CheckpointFunction.forward`, `CheckpointFunction.backward`", ["precision:CheckpointFunction.forward", "precision:CheckpointFunction.backward"],
  "test_23_precision.py", None)
s("23", "실행", "`pretrain.py --preset tiny`를 fp32와 `--dtype bf16 --grad_ckpt`로 각각 돌려 tok/s 비교, 확인 문제 23 → 해설 대조", done="tok/s 비교 기록")
s("24", "구현", "`docs/24` §2.1–2.4 읽고 `LoRALinear.forward`", ["lora:LoRALinear.forward"], "test_24_lora.py", "init or adapters")
s("24", "구현", "§2.5 읽고 `LoRALinear.merge`", ["lora:LoRALinear.merge"], "test_24_lora.py", None)
s("24", "실행", "`sft.py --lora_r 8 --lr 1e-3 --out out/sft_lora.pt`, `compare.py --ref out/pretrain.pt`, 확인 문제 24 → 해설 대조", done="전체 SFT와 비교 기록")



def cmd(p):
    return f"`pytest tests/{p['test']}" + (f' -k "{p["k"]}"' if p["k"] else "") + "`"


def emit():
    n_total = len(P)
    header = HEADER
    for w in (5, 7, 9, 10):
        header = header.replace("{w%d:.0f}" % w, f"{n_total / w:.0f}")
    print(header.format(n=n_total, h=n_total / 2))
    starts = [a for a, _, _ in PARTS] + [len(P)]
    for (a, title, note), b in zip(PARTS, starts[1:]):
        print(f"## {title}\n")
        if note:
            print(note + "\n")
        print("| # | Step | 유형 | 할 일 | 완료 기준 |\n|---|---|---|---|---|")
        for i in range(a, b):
            p = P[i]
            done = cmd(p) + " 통과" if p["test"] else p["done"]
            print(f"| {i + 1} | {p['step']} | {p['kind']} | {p['task']} | {done} |")
        print()
    print(FOOTER.replace("{s13}", str(next(i for i, p in enumerate(P, 1) if p["step"] == "13"))))


HEADER = """# 30분 세션 학습 계획

[`STUDY_PLAN.md`](STUDY_PLAN.md)(평일 60분 + 주말 3시간)를 더 잘게 나눈 버전이다. 퇴근 후 집중력이 떨어지는 날에도 한 단위를 끝낼 수 있도록, 모든 세션을 **30분 안에 끝나는 하나의 작업**으로 나눴다.

- 세션 {n}개, 합계 약 {h:.0f}시간. 요일에 묶여 있지 않으므로 번호 순서대로 진행한다.
- 구현 세션은 함수 1–3개만 다루고, **특정 테스트가 통과하면 끝난다**. 시간이 아니라 완료 기준이 세션의 단위이다. 30분 안에 못 끝내면 다음 날 같은 세션을 이어서 한다.
- 각 구현 세션의 완료 기준은 "직전 세션까지 구현한 상태에서는 실패하고, 그 세션의 함수를 구현하면 통과한다"를 함수 단위 시뮬레이션으로 검증했다. 순서를 바꾸면 기준 테스트가 통과하지 않을 수 있다.
- 학습 실행 시간은 M5에서 실측하지 않은 추정치이다.
- 모든 문서에는 '예제로 따라가기' 절이 있다. 읽기·구현 세션에서 해당 절의 숫자 예제를 종이로 직접 따라 계산한 뒤 구현하고, 구현 후에는 절 끝의 확인 코드로 같은 숫자가 나오는지 본다.

## 진행 속도

| 진행 방식 | 기간 |
|---|---|
| 평일 1세션 (주 5세션, 주 2.5시간) | 약 {w5:.0f}주 |
| 매일 1세션 (주 7세션, 주 3.5시간) | 약 {w7:.0f}주 |
| 평일 1세션 + 주말 하루 4세션 (주 9세션, 주 4.5시간) | 약 {w9:.0f}주 |
| 평일 2세션 (주 10세션, 주 5시간) | 약 {w10:.0f}주 |

## 세션 유형

| 유형 | 30분 구성 |
|---|---|
| 읽기 | 25분 문서의 해당 절을 읽으며 유도를 종이에 따라 쓰기, 5분 핵심을 한 줄로 요약 |
| 구현 | 3분 지난 세션 테스트 확인, 7분 해당 절 읽기, 17분 구현과 테스트, 3분 커밋 |
| 문제 | 20분 확인 문제 풀기, 10분 `docs/answers/` 대조 |
| 실행 | 스크립트를 실행하고 결과를 기록한다. 긴 학습을 걸어 두기만 하는 세션은 5분 안에 끝난다 |
| 버퍼 | 밀린 세션 처리. 밀린 것이 없으면 적힌 과제 |

막혔을 때, 긴 실행, 브랜치 운영 규칙은 [`STUDY_PLAN.md`의 세션 운영 규칙](STUDY_PLAN.md#세션-운영-규칙)과 같다.
"""

FOOTER = """## 진도 확인

현재 세션까지의 테스트를 한 번에 돌려 회귀가 없는지 확인한다. 예: {s13}번 세션(step 13 시작) 직전

```bash
pytest tests/test_0*.py tests/test_1[0-2]*.py -q
```
"""


LATEX = {
    "$\\partial\\sigma/\\partial x_j = \\hat x_j/D$": "∂σ/∂x_j = x̂_j / D",
    "$\\mathbb E[m_t] = (1-\\beta_1^t)\\mathbb E[g]$": "E[m_t] = (1 − β₁ᵗ)·E[g]",
    "$R_m^\\top R_n = R_{n-m}$": "R_mᵀ·R_n = R_(n−m)",
    "$T=1$": "T=1", "$T_k=5$": "T_k=5",
    "$\\mathrm{Var}(x_L) = \\mathrm{Var}(x_0) + 2L\\sigma_f^2$": "Var(x_L) = Var(x_0) + 2L·σ_f²",
    "$0.02/\\sqrt{2L}$": "0.02/√(2L)",
    "$z=0$": "z=0",
    "$\\pi^* \\propto \\pi_{ref}e^{r/\\beta}$": "π* ∝ π_ref·exp(r/β)",
    "$Z(x)$": "Z(x)",
    "$\\mathbb E[b\\nabla\\log\\pi]=0$": "E[b·∇log π] = 0",
    "$k_3$": "k3",
    "$a/l$": "a/l",
    "$p$": "p",
}


def clean(s):
    for k, v in LATEX.items():
        s = s.replace(k, v)
    assert "$" not in s, s
    return s

DOCS = {f"{i:02d}": n for i, n in enumerate([
    None, "autograd", "softmax_xent", "manual_backward", "optimizer", "bpe", "attention", "rope", "causal_attention",
    "block", "gpt", "data", "training", "sampling", "kv_cache", "sft", "reward_model", "dpo", "grpo", "quantization",
    "speculative_decoding", "flash_attention", "gqa", "mixed_precision_checkpointing", "lora"]) if n}



def sessions_js():
    starts = [a for a, _, _ in PARTS] + [len(P)]
    part_of = {i: pi for pi, (a, b) in enumerate(zip(starts, starts[1:])) for i in range(a, b)}
    sessions = [{
        "n": i + 1, "part": part_of[i], "step": p["step"], "kind": p["kind"], "task": clean(p["task"]),
        "cmd": cmd(p).strip("`") if p["test"] else None, "done": None if p["test"] else clean(p["done"]),
        "doc": f"{p['step']}_{DOCS[p['step']]}.md" if p["step"] in DOCS else None,
    } for i, p in enumerate(P)]
    return ("const PARTS = " + json.dumps([t for _, t, _ in PARTS], ensure_ascii=False) + ";\n"
            + "const SESSIONS = " + json.dumps(sessions, ensure_ascii=False, separators=(",", ":")) + ";")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("md", "all"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit()
        (ROOT / "docs/STUDY_PLAN_30MIN.md").write_text(buf.getvalue())
        print("wrote docs/STUDY_PLAN_30MIN.md")
    if what in ("tracker", "all"):
        template = (ROOT / "tools/tracker_template.html").read_text()
        out = ROOT / "docs/tracker/index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(template.replace("/*__DATA__*/", sessions_js()))
        print("wrote docs/tracker/index.html")
