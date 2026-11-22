# Step 15. Supervised Fine-Tuning (SFT)

- 구현: `minillm/align.py` — `build_sft_example`
- 테스트: `pytest tests/test_15_sft.py`
- 실행: `python scripts/sft.py`
- 선행: step 02(ignore_index), 12

## 1. 목표

사전학습 모델을 (지시, 응답) 쌍으로 미세조정해 지시를 따르게 한다. 핵심 구현은 프롬프트 토큰을 loss에서 제외하는 label 마스킹이다.

## 2. 이론

### 2.1 목적함수

사전학습 모델은 텍스트 분포 $p(x)$를 모델링할 뿐 특정 형식의 지시에 응답하도록 학습되지 않았다. SFT는 시연 데이터 $(x, y)$에 대해 조건부 MLE를 수행한다:

$$\mathcal L_{SFT}(\theta) = -\mathbb E_{(x,y)}\sum_{t=1}^{|y|}\log p_\theta(y_t \mid x, y_{<t}).$$

합은 **응답 토큰에 대해서만** 취한다. 프롬프트는 조건으로만 쓰이고 예측 대상이 아니다. 프롬프트(사용자 입력)의 분포를 모델링할 필요가 없고, 학습 신호를 응답 생성에 집중시키기 위해서다. (Shi et al. 2024는 데이터 조건에 따라 프롬프트 loss를 일부 포함하는 것이 도움이 되는 경우도 보고했다.)

### 2.2 템플릿과 종료 토큰

```
<|user|>Write a story that uses the word "ball".<|assistant|>{이야기}<|endoftext|>
```

special token으로 역할 경계를 표시하고, 응답 끝의 EOS도 학습 대상에 포함해 **언제 멈출지**를 학습시킨다. 추론 시에는 `<|assistant|>`까지를 프롬프트로 주고 EOS가 나올 때까지 생성한다.

### 2.3 이 레포의 과제: 말뭉치에서 자동 생성한 지시 데이터

TinyStories에는 지시 데이터가 없으므로 다음과 같이 만든다(`scripts/sft.py`):

1. 이야기 하나를 고른다.
2. 이야기 속 단어(4글자 이상, 불용어 제외) 하나를 키워드로 뽑는다.
3. 지시 = "이 키워드를 써서 이야기를 써라", 응답 = 원래 이야기.

응답이 키워드를 포함한다는 것이 구성상 보장되므로 사람의 라벨 없이 정답 시연을 얻는다. 평가는 학습 풀에서 제외한 키워드 20개(`EVAL_KEYWORDS`)로 생성해 **키워드 적중률**(EOS로 끝맺고 키워드를 단어 단위로 포함한 비율)을 측정한다. 이 같은 규칙 기반 보상이 이후 DPO(step 17)와 GRPO(step 18)의 보상으로 그대로 쓰인다.

### 2.4 인덱싱

$s = \text{prompt} + \text{response}$, 입력 $x = s_{0:-1}$, 라벨 $y = s_{1:}$. 위치 $j$는 $s_{j+1}$을 예측하므로, $j + 1 < |\text{prompt}|$이면 라벨을 `ignore_index`로 둔다. $j = |\text{prompt}| - 1$(마지막 프롬프트 토큰 `<|assistant|>`)의 라벨은 응답 첫 토큰이므로 **포함**된다.

예: prompt `[10, 11, 12]`, response `[20, 21, 2]`, `block_size=8`
- $x$ = `[10, 11, 12, 20, 21, 0, 0, 0]`
- $y$ = `[-100, -100, 20, 21, 2, -100, -100, -100]`

### 2.5 Padding과 truncation

길이를 `block_size`로 맞추기 위해 오른쪽에 padding한다(입력 0, 라벨 `ignore_index`). causal attention에서 오른쪽 padding은 앞쪽 유효 위치에 영향을 주지 않는다. 길면 `block_size + 1`에서 자른다(`sft.py`는 응답이 잘리지 않는 예시만 사용).

### 2.6 학습 설정

사전학습보다 작은 lr(3e-4 → cosine), 짧은 학습. 너무 오래 학습하면 사전학습 능력이 손상되거나(catastrophic forgetting) 작은 데이터에 과적합한다. LIMA(Zhou et al. 2023)는 적은 수의 고품질 시연으로도 형식 정렬이 가능함을 보였다.

## 3. 예제로 따라가기

### 3.1 SFT 예시 한 개가 텐서가 되기까지

실제 데이터 한 건:

```
<|user|>Write a story that uses the word "garden".<|assistant|>Tom liked to play in the garden. ... <|endoftext|>
└──────────────── prompt (loss 제외) ────────────────┘└──────────── response (loss 대상) ─────────────┘
```

토큰을 짧게 줄여 prompt = `[1, 2, 3, 4]` (마지막 4가 `<|assistant|>`), response = `[10, 11, 12, 0]` (마지막 0이 EOS), `block_size = 8`이라 하자.

| 위치 $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `s` (prompt+response) | 1 | 2 | 3 | 4 | 10 | 11 | 12 | 0 |
| 입력 `x = s[:-1]` + padding | 1 | 2 | 3 | 4 | 10 | 11 | 12 | **0** (pad) |
| 예측 대상 `s[j+1]` | 2 | 3 | 4 | 10 | 11 | 12 | 0 | — |
| 라벨 `y` | −100 | −100 | −100 | **10** | 11 | 12 | **0** | −100 (pad) |
| loss에 포함? | ✗ | ✗ | ✗ | O | O | O | O | ✗ |

- 위치 0–2는 다음 토큰이 프롬프트 토큰이라 제외한다.
- 위치 3(`<|assistant|>`)은 응답 첫 토큰 10을 예측하므로 **포함**한다. 이 위치를 빼면 모델은 응답을 어떻게 시작할지 배우지 못한다.
- 위치 6은 EOS(0)를 예측한다. 이 라벨 덕분에 모델이 이야기를 언제 끝낼지 배운다.
- 위치 7의 입력 0은 padding이다. causal attention이라 앞 위치들의 계산에 영향을 주지 않고, 라벨이 −100이라 loss에도 들어가지 않는다.

### 3.2 같은 배치 안의 길이가 다른 예시들

| 예시 | 프롬프트 길이 | 응답 길이 | loss에 들어가는 위치 수 |
|---|---|---|---|
| ① | 15 | 180 | 180 |
| ② | 15 | 60 | 60 |

`cross_entropy`는 유효 위치 전체(240개)의 평균이므로 ①이 ②보다 3배 큰 비중을 가진다(토큰 평균). 응답 단위로 같은 비중을 주려면 예시별 평균을 먼저 낸 뒤 평균해야 한다(해설 15의 2번).

### 3.3 학습 전후로 무엇이 바뀌나

같은 프롬프트에 대해 기대하는 변화:

| | 사전학습 모델 | SFT 후 |
|---|---|---|
| `<|assistant|>` 다음 첫 토큰 | 학습 때 본 적 없는 special token 뒤라 무작위에 가까움 | 이야기 첫 문장 ("Once", "Tom", …) |
| 키워드 포함 | 우연히 포함될 때만 | 지시를 따라 포함하는 비율이 오름 |
| 끝맺음 | block_size까지 계속 생성하거나 다른 이야기로 넘어감 | 이야기 하나 뒤 EOS |

`scripts/sft.py`는 학습 전후의 키워드 적중률(EOS로 끝나고 키워드를 포함한 비율)을 출력한다.

### 3.4 확인 코드

```python
from minillm.align import build_sft_example

x, y = build_sft_example([1, 2, 3, 4], [10, 11, 12, 0], block_size=8)
print(x.tolist())   # [1, 2, 3, 4, 10, 11, 12, 0]
print(y.tolist())   # [-100, -100, -100, 10, 11, 12, 0, -100]
```

## 4. 구현 가이드

- `build_sft_example(prompt_ids, response_ids, block_size, ignore_index)` → `(input_ids, labels)` 각각 길이 `block_size`의 int64 텐서.
- loss는 별도 구현이 필요 없다. `model(x, y)`가 step 02의 `cross_entropy(ignore_index=-100)`를 쓴다.

## 5. 검증

마스킹·padding·truncation 경계를 테스트한다. `sft.py`는 학습 전후 키워드 적중률과 샘플을 출력한다.

## 6. 확인 문제

> 해설: [answers/15_sft.md](answers/15_sft.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. 프롬프트 토큰까지 loss에 포함하면 모델은 무엇을 추가로 학습하는가? 이 과제에서 결과가 어떻게 달라질지 예측하고 실험해 보라.
2. 마이크로배치마다 응답 토큰 수가 다를 때, 토큰 평균 loss와 시퀀스 평균 loss는 어떻게 다른가? 긴 응답에 주는 가중치 관점에서 설명하라.
3. 키워드를 이야기 속 단어에서 뽑는 방식이 평가 키워드(일반 명사)와 분포가 다르다면 어떤 문제가 생길 수 있는가?

## 참고문헌

- L. Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022.
- J. Wei et al., "Finetuned Language Models Are Zero-Shot Learners," ICLR 2022.
- Y. Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions," ACL 2023.
- C. Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023.
- Z. Shi et al., "Instruction Tuning With Loss Over Instructions," NeurIPS 2024.
