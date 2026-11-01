# Step 11. 데이터 파이프라인: 문서 packing과 배치 샘플링

- 구현: `minillm/data.py` — `pack_documents`, `get_batch`
- 테스트: `pytest tests/test_11_data.py`
- 사용처: `scripts/prepare_data.py`(packing), `scripts/pretrain.py`(배치)

## 1. 목표

토큰화된 문서들을 하나의 토큰 스트림으로 이어 붙이고(packing), 무작위 위치의 고정 길이 창으로 `(x, y)` 학습 배치를 만든다.

## 2. 이론

### 2.1 Packing

문서 길이는 제각각이므로 문서마다 padding하면 계산이 낭비된다. 사전학습에서는 문서를 EOS 토큰으로 구분해 하나의 긴 스트림으로 연결하고, 고정 길이 $T$의 창을 잘라 쓴다(GPT-2/3, T5 등). 모든 위치가 유효 토큰이므로 padding 낭비가 0이다.

결과적으로 한 창 안에 여러 문서가 섞이고, attention은 EOS 너머 이전 문서도 참조한다. EOS가 문서 경계를 알려 주므로 모델은 경계 이후 이전 문맥을 무시하도록 학습된다. 이 간섭을 완전히 없애려면 문서 단위 attention mask를 쓴다(Krell et al. 2021; LLaMA 3). 이 레포는 단순한 기본형을 쓴다.

### 2.2 다음 토큰 예측 배치

스트림 $s$와 시작점 $i$에 대해

$$x = s[i : i+T],\qquad y = s[i+1 : i+1+T]$$

위치 $t$에서 모델은 $x_{\le t} = s[i..i+t]$를 보고 $y_t = s_{i+t+1}$을 예측한다. 한 창이 $T$개의 예측 문제를 동시에 제공한다.

시작점 $i$는 $[0, |s| - T)$에서 균등하게 뽑는다. $y$의 마지막 인덱스 $i + T \le |s| - 1$이 되어야 하므로 상한은 $|s| - T$(배타)이다.

### 2.3 Epoch 대신 무작위 창

창을 복원추출하면 "epoch" 개념 없이 스텝 수로 학습량을 정한다. 한 스텝의 토큰 수는 $B \cdot T \cdot (\text{accumulation})$이고, 총 학습 토큰은 이 값 × 스텝 수다(`pretrain.py`가 출력). 같은 위치가 중복 샘플링될 수 있지만, 전체 스트림을 여러 번 덮는 규모에서는 통계적으로 epoch 방식과 큰 차이가 없다.

### 2.4 저장 형식

$V < 65536$이면 토큰을 `uint16`(2바이트)로 저장할 수 있다. 학습 시에는 `np.memmap`으로 파일을 메모리에 다 올리지 않고 필요한 부분만 읽는다. train/val은 **문서 단위**로 분리해 같은 문서가 양쪽에 섞이지 않게 한다(`prepare_data.py`).

## 3. 구현 가이드

- `pack_documents`: 각 문서 뒤에 `eos_id`를 붙여 `uint16` 배열로 연결. 빈 문서도 EOS 하나를 남긴다(테스트 규약).
- `get_batch`: `torch.randint(len(data) - block_size, (batch_size,), generator=generator)`로 시작점을 뽑아야 테스트의 재현성 검사를 통과한다. memmap 슬라이스를 `astype(np.int64)` 후 `torch.from_numpy`, 마지막에 `.to(device)`.

## 4. 검증

packing 결과와 dtype, `y`가 `x`를 한 칸 민 것인지, 창이 연속 구간인지, 같은 시드에서 같은 배치가 나오는지 확인한다.

## 5. 확인 문제

1. 문서 단위 padding과 packing의 계산 효율을 문서 길이 분포의 평균과 분산으로 비교하라.
2. 상한을 `len(data) - block_size + 1`로 두면 어떤 경우에 오류가 나는가?
3. packing 시 문서 경계를 넘는 attention을 막는 마스크를 `scaled_dot_product_attention`의 `mask`로 만들려면 어떤 정보가 필요한가?

## 참고문헌

- T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020.
- C. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," JMLR 2020.
- M. M. Krell et al., "Efficient Sequence Packing without Cross-contamination," arXiv:2107.02027, 2021.
- A. Grattafiori et al., "The Llama 3 Herd of Models," arXiv:2407.21783, 2024.
