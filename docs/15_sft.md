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

## 3. 구현 가이드

- `build_sft_example(prompt_ids, response_ids, block_size, ignore_index)` → `(input_ids, labels)` 각각 길이 `block_size`의 int64 텐서.
- loss는 별도 구현이 필요 없다. `model(x, y)`가 step 02의 `cross_entropy(ignore_index=-100)`를 쓴다.

## 4. 검증

마스킹·padding·truncation 경계를 테스트한다. `sft.py`는 학습 전후 키워드 적중률과 샘플을 출력한다.

## 5. 확인 문제

1. 프롬프트 토큰까지 loss에 포함하면 모델은 무엇을 추가로 학습하는가? 이 과제에서 결과가 어떻게 달라질지 예측하고 실험해 보라.
2. 마이크로배치마다 응답 토큰 수가 다를 때, 토큰 평균 loss와 시퀀스 평균 loss는 어떻게 다른가? 긴 응답에 주는 가중치 관점에서 설명하라.
3. 키워드를 이야기 속 단어에서 뽑는 방식이 평가 키워드(일반 명사)와 분포가 다르다면 어떤 문제가 생길 수 있는가?

## 참고문헌

- L. Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022.
- J. Wei et al., "Finetuned Language Models Are Zero-Shot Learners," ICLR 2022.
- Y. Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions," ACL 2023.
- C. Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023.
- Z. Shi et al., "Instruction Tuning With Loss Over Instructions," NeurIPS 2024.
