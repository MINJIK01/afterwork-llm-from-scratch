# Step 05. Byte-level BPE 토크나이저

- 구현: `minillm/tokenizer.py` — `get_pair_counts`, `merge`, `BPETokenizer.train`, `_encode_chunk`, `decode`
- 테스트: `pytest tests/test_05_bpe.py`
- 사용처: `scripts/prepare_data.py`가 이 구현으로 말뭉치 토크나이저를 학습한다.

## 1. 목표

GPT-2 계열의 byte-level BPE를 처음부터 구현한다: 학습(merge 규칙 생성), 인코딩(merge 재적용), 디코딩, special token 처리.

## 2. 이론

### 2.1 왜 subword인가

- 단어 단위: 어휘 밖 단어(OOV)를 표현할 수 없고 어휘가 매우 커진다.
- 문자 단위: OOV는 없지만 시퀀스가 길어진다. self-attention 비용이 $O(T^2)$이므로 같은 텍스트를 더 많은 토큰으로 표현하는 것은 계산·문맥 길이 측면에서 손해다.
- **Byte-level BPE**: 기본 기호를 256개 바이트로 두면 모든 UTF-8 텍스트(한글, 이모지 포함)를 OOV 없이 표현하고, 자주 나오는 바이트 열은 병합해 시퀀스를 줄인다.

### 2.2 학습 알고리즘

1. **Pre-tokenization**: 정규식으로 텍스트를 "단어" 조각으로 나눈다(공백은 뒤 단어의 앞에 붙는다: `" dog"`). merge가 단어·구두점 경계를 넘지 않게 하여, `"dog."`, `"dog!"` 같은 조합이 각각 별도 토큰을 차지하는 것을 막는다.
2. 각 조각을 바이트 id 튜플로 바꾸고, 같은 조각은 `{튜플: 빈도}`로 묶는다.
3. 반복 $k = 0, 1, \dots, M-1$:
   - 모든 인접 pair의 빈도(조각 빈도로 가중)를 센다.
   - 최빈 pair $(a, b)$를 고른다. 동률이면 이 레포는 **사전순으로 작은 pair**를 고른다(재현성).
   - 새 id $256 + k$를 할당하고, 모든 조각에서 $(a, b)$를 왼쪽부터 겹치지 않게 치환한다.

예: `"aaabdaaabac"` → (a,a)=4회 → `Z=aa`: `ZabdZabac` → (Z,a)와 (a,b)가 2회로 동률, 사전순으로 (a,b)<(Z,a) (`a`=97 < `Z`=256) → `Y=ab`: `ZYdZYac` → (Z,Y)=2회 → `X=ZY`.

### 2.3 인코딩: merge 순서의 재현

새 텍스트를 인코딩할 때는 학습 순서(rank)를 그대로 따라야 한다. 각 조각에서 "현재 시퀀스에 존재하는 pair 중 rank가 가장 낮은(가장 먼저 학습된) 것"을 찾아 병합하고, 더 이상 병합할 pair가 없을 때까지 반복한다. rank를 무시하고 임의 순서로 병합하면 학습 때와 다른 분할이 나와 모델이 본 적 없는 토큰 열이 된다.

### 2.4 Special token

`<|endoftext|>`(문서 경계·응답 종료), `<|user|>`, `<|assistant|>`(SFT 템플릿, step 15)는 merge로 만들어지지 않는 예약 id를 받는다. 인코딩 시 정규식 분할 **전에** 먼저 잘라내야 `<`, `|` 등으로 쪼개지지 않는다(`encode`에 제공됨).

### 2.5 계산 비용

- 학습: 순진한 구현은 매 merge마다 모든 조각의 pair를 다시 센다. 비용은 $O(M \cdot \sum_{\text{distinct } w} |w|)$. 조각 빈도로 묶는 것만으로도 원문 길이 대신 고유 조각 수에 비례하게 된다. 실제 라이브러리(HF tokenizers, SentencePiece)는 merge된 pair가 포함된 조각만 갱신하는 증분 방식과 우선순위 큐를 쓴다. 이 레포의 기본 설정(2MB, vocab 4096)은 순수 Python으로 수 분 내에 끝난다.
- 인코딩: 조각 단위 결과를 캐시(`self._cache`)하면 자연어의 Zipf 분포 덕분에 대부분 캐시에 적중한다.

### 2.6 Vocabulary 크기의 trade-off

$V$가 크면 시퀀스가 짧아지지만, 임베딩과 출력층 파라미터 $V \cdot d$가 늘고 각 토큰의 학습 빈도가 줄어든다. 12M 모델(d=384)에서 $V = 4096$이면 임베딩이 약 1.6M(전체의 13%)이다. 압축률은 bytes/token으로 측정한다(`prepare_data.py`가 출력).

### 2.7 디코딩

id → 바이트열을 이어 붙이고 UTF-8로 디코드한다. 토큰 경계가 멀티바이트 문자 중간에 있을 수 있으므로 **전체를 이어 붙인 뒤** 디코드해야 하며, 생성 도중의 불완전한 바이트열은 `errors="replace"`로 처리한다.

## 3. 구현 가이드

- 조각은 `tuple[int, ...]`로 다룬다(dict key로 쓰기 위해 hashable).
- `merge((1,1,1), (1,1), 9)` → `(9, 1)`: 왼쪽부터, 겹치지 않게.
- `train`: `max(counts, key=lambda p: (counts[p], -p[0], -p[1]))`로 동률 규칙을 구현할 수 있다. 더 병합할 pair가 없으면 조기 종료한다(이 경우 vocab이 목표보다 작아진다).
- `_encode_chunk`: `min(pairs, key=lambda p: self.merges.get(p, inf))`로 최저 rank pair를 찾는다.

## 4. 검증

Wikipedia 예제의 정확한 merge 순서, 한글·이모지·탭·special token이 섞인 문자열의 왕복(encode→decode) 일치, 저장/로드 후 동일 인코딩, 압축률을 확인한다.

## 5. 확인 문제

> 해설: [answers/05_bpe.md](answers/05_bpe.md) — 먼저 직접 풀어 본 뒤 확인할 것.

1. pre-tokenization 없이 학습하면 어떤 토큰들이 생기는가? 공백을 단어 **앞**에 붙이는 설계의 장점은?
2. 숫자를 BPE로 병합하면 `"1234"`가 `"12"`+`"34"` 또는 `"123"`+`"4"` 등 일관되지 않게 분할될 수 있다. 산술 능력에 어떤 영향이 있는가? (LLaMA는 숫자를 한 자리씩 분할한다.)
3. 같은 말뭉치로 학습한 BPE와 Unigram LM 토크나이저(SentencePiece)의 학습 원리 차이를 설명하라.

## 참고문헌

- R. Sennrich, B. Haddow, A. Birch, "Neural Machine Translation of Rare Words with Subword Units," ACL 2016.
- P. Gage, "A New Algorithm for Data Compression," C Users Journal, 1994.
- A. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019 (GPT-2, byte-level BPE).
- T. Kudo, J. Richardson, "SentencePiece," EMNLP 2018 (demo).
- K. Bostrom, G. Durrett, "Byte Pair Encoding is Suboptimal for Language Model Pretraining," Findings of EMNLP 2020.
