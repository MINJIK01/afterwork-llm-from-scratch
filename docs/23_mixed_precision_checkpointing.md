# Step 23. Mixed Precision 학습과 Gradient Checkpointing

- 구현: `minillm/precision.py` — `bf16_round`, `LossScaler.unscale_`, `LossScaler.update`, `train_step_amp`, `CheckpointFunction.forward/backward`
- 테스트: `pytest tests/test_23_precision.py`
- 실행: `python scripts/pretrain.py --preset m5 --dtype bf16 --grad_ckpt`
- 선행: step 01, 04, 12 (step 12 직후에 진행해도 된다)

## 1. 목표

학습 메모리와 속도를 좌우하는 두 기법을 구현한다. (1) 부동소수점 형식을 비트 수준에서 다루고, fp32 master weight + 저정밀 연산 + loss scaling으로 이루어진 mixed precision 학습 스텝을 만든다. (2) activation을 저장하지 않고 backward 때 다시 계산하는 gradient checkpointing을 custom autograd Function으로 구현한다.

## 2. 이론: Mixed precision

### 2.1 부동소수점 형식

| 형식 | 부호/지수/가수 비트 | 최대값 | 최소 정규수 | 단위 반올림 $u$ |
|---|---|---|---|---|
| fp32 | 1/8/23 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $2^{-24}\approx 6\times10^{-8}$ |
| fp16 | 1/5/10 | 65504 | $6.1\times10^{-5}$ | $2^{-11}\approx 4.9\times10^{-4}$ |
| bf16 | 1/8/7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | $2^{-8}\approx 3.9\times10^{-3}$ |

bf16은 fp32와 지수 비트가 같아 **표현 범위가 같고**, 가수만 7비트로 줄였다. fp16은 정밀도가 bf16보다 높지만 범위가 좁아 overflow/underflow가 쉽게 일어난다.

### 2.2 fp32 → bf16 반올림을 비트로 구현하기

bf16 값은 fp32 비트열의 상위 16비트와 같다. 따라서 변환은 하위 16비트를 반올림해 버리는 연산이다. round-to-nearest-even(RNE): 하위 16비트가 `0x8000`보다 크면 올리고, 작으면 버리고, 정확히 `0x8000`(정확히 중간)이면 결과의 최하위 비트(원래 비트 16)가 짝수가 되도록 한다. 이는

$$\text{bits}' = \big(\text{bits} + \texttt{0x7FFF} + ((\text{bits} \gg 16)\ \&\ 1)\big)\ \&\ \texttt{0xFFFF0000}$$

한 줄로 구현된다. 비트 16이 1(홀수)이면 `0x8000`을 더한 셈이 되어 중간값이 올라가고, 0이면 `0x7FFF`만 더해져 중간값이 버려진다. 올림이 가수를 넘치면 지수로 자연스럽게 carry되고, 최대 유한값 근처에서는 $\infty$(`0x7F800000`)가 된다.

### 2.3 Master weight가 필요한 이유

업데이트 $\Delta = \eta \hat m/(\sqrt{\hat v}+\epsilon)$은 대개 가중치보다 몇 자릿수 작다. 가중치를 bf16으로 저장하면 $|\Delta| < u|w|$일 때 $w + \Delta$가 $w$로 반올림되어 업데이트가 사라진다. 예: $w = 1$, $\Delta = 10^{-3}$이면 bf16의 $u \approx 3.9\times10^{-3}$보다 작으므로 업데이트가 소실된다. 그래서 파라미터·gradient·optimizer 상태는 fp32로 유지하고, forward/backward의 행렬곱만 저정밀로 계산한다(Micikevicius et al. 2018).

### 2.4 Autocast

`torch.autocast(device, dtype)`는 연산 단위로 입력을 캐스팅한다. 행렬곱·Linear 등은 저정밀로, softmax·norm·loss처럼 합산 오차나 범위가 민감한 연산은 fp32로 실행한다. 다만 직접 구현한 연산(`exp`, `sum` 조합)은 장치별 autocast 목록에 따라 저정밀로 남을 수 있으므로, `train_step_amp`는 logits를 `float()`으로 올린 뒤 loss를 계산한다. forward만 autocast 안에서 실행하고 backward는 밖에서 호출한다. 각 연산의 backward는 forward와 같은 dtype으로 자동 실행된다.

### 2.5 fp16의 loss scaling

fp16에서는 작은 gradient(예: $10^{-8}$)가 표현 범위 아래로 떨어져 0이 된다(underflow). loss에 $S$(예: $2^{16}$)를 곱해 backward하면 chain rule에 의해 모든 gradient가 $S$배로 커져 표현 범위 안으로 들어온다. optimizer step 전에 $S$로 나누어 원래 크기로 되돌린다.

**동적 scaling**: $S$가 너무 크면 gradient가 overflow(inf/NaN)한다. 이때는 그 스텝을 건너뛰고 $S$를 절반으로 줄인다. overflow 없이 $N$스텝 연속 성공하면 $S$를 2배로 늘린다. 이렇게 $S$는 overflow 직전 수준을 추적한다. bf16은 범위가 fp32와 같아 loss scaling이 필요 없다.

### 2.6 Apple Silicon(MPS)에서의 주의

MPS의 autocast·bf16 지원은 PyTorch와 macOS 버전에 따라 다르다. `--dtype bf16`이 오류를 내면 `--dtype fp16`(loss scaling 사용)을 쓴다. 통합 메모리 구조와 작은 모델에서는 속도 향상이 GPU 서버보다 작을 수 있으므로 로그의 `tok/s`로 확인한다. (이 레포는 CPU autocast로만 검증했다.)

## 3. 이론: Gradient checkpointing

### 3.1 학습 메모리 구성

AdamW + fp32 학습에서 파라미터당 메모리는 파라미터 4B + gradient 4B + $m$, $v$ 8B = 16B이다(ZeRO 논문의 계산). 여기에 backward를 위해 저장하는 activation이 더해지는데, 레이어당 대략 $B \cdot T \cdot d$의 상수배 + attention 확률 $B \cdot H \cdot T^2$에 비례하고 레이어 수 $L$만큼 쌓인다. 배치와 문맥이 커지면 activation이 지배적이 된다.

### 3.2 재계산으로 메모리 줄이기

블록 입력만 저장하고 블록 내부 activation은 버린 뒤, backward 때 해당 블록의 forward를 다시 실행해 필요한 값을 복원한다. activation 메모리는 "블록 경계 $L$개 + 한 번에 한 블록의 내부"로 줄고, 비용은 forward 1회 추가(학습 FLOPs $6N$에 $2N$ 추가, 약 33%)이다. Chen et al. (2016)은 $\sqrt L$ 간격으로 경계를 저장하면 $O(\sqrt L)$ 메모리를 얻을 수 있음을 보였다. Korthikanti et al. (2022)은 재계산 비용이 싼 연산(attention softmax 등)만 골라 재계산하는 selective recomputation을 제안했다.

### 3.3 Custom autograd Function으로 구현

- `forward(ctx, fn, *args)`: `torch.no_grad()`로 `fn`을 실행해 내부 그래프와 activation을 만들지 않고, 입력만 `ctx.save_for_backward`로 저장한다.
- `backward(ctx, grad_out)`: 저장된 입력을 `detach().requires_grad_()`로 새 leaf로 만들고, `torch.enable_grad()` 아래에서 `fn`을 다시 실행한 뒤 `torch.autograd.backward(out, grad_out)`를 호출한다. 이때 블록 파라미터의 `.grad`에는 gradient가 직접 누적되고, 입력 leaf의 `.grad`를 바깥 그래프로 반환한다.
- 이 방식(재진입형)은 입력 중 최소 하나가 `requires_grad`여야 하고(아니면 Function 출력이 grad를 요구하지 않아 backward가 호출되지 않는다), dropout이 있으면 forward와 재계산의 난수 상태를 맞춰야 한다. 이 모델은 dropout이 없다.

`enable_checkpointing(model)`은 각 `Block` 인스턴스의 클래스를 `CheckpointedBlock`(Block 하위 클래스)으로 바꿔, 학습 중(grad 활성, cache 없음) forward만 checkpoint로 감싼다. 모듈 구조가 그대로이므로 `state_dict` 키가 바뀌지 않아 체크포인트 호환성이 유지된다.

## 4. 구현 가이드

- `bf16_round`: `x.float().view(torch.int32)`로 비트를 얻고 2.2의 식 적용 후 `view(torch.float32)`. Python의 `~0xFFFF`는 int32 마스크 `0xFFFF0000`과 같다.
- `LossScaler.unscale_`: grad를 나누고 `torch.isfinite`로 검사, `update`: 2.5의 규칙.
- `train_step_amp`: step 12 `train_step`과 같은 구조에 autocast, fp32 loss, scaler 분기를 더한다. overflow면 optimizer step 없이 `(loss, inf)` 반환.
- `CheckpointFunction`: 3.3 그대로.

## 5. 검증

랜덤 비트 패턴 20만 개 + 정확히 중간인 값 + 최대값 근처 + subnormal에서 `x.to(torch.bfloat16)`와 비트 단위 일치, loss scaler 규칙, bf16 학습 스텝의 loss가 fp32와 0.05 이내이고 파라미터·grad가 fp32로 유지되는지, overflow 시 step 건너뛰기, checkpointing에서 gradient 일치·저장 activation 바이트 절반 미만·`state_dict` 키 불변을 확인한다.

## 6. 확인 문제

1. bf16에서 $1 + 2^{-8}$과 $1 + 3\cdot2^{-8}$은 각각 무엇으로 반올림되는가? 2.2의 식으로 직접 계산하라.
2. m5 프리셋($N$=12.2M)의 파라미터·gradient·Adam 상태 메모리를 계산하고, $B$=32, $T$=256일 때 attention 확률 행렬($B \cdot H \cdot T^2$, fp32)의 레이어당 메모리와 비교하라.
3. 모든 블록을 checkpoint하면 FLOPs가 몇 % 늘어나는가? 같은 메모리로 배치를 2배 키울 수 있다면 시간당 처리 토큰은 어떻게 되는가?

## 참고문헌

- P. Micikevicius et al., "Mixed Precision Training," ICLR 2018.
- D. Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," arXiv:1905.12322, 2019.
- T. Chen et al., "Training Deep Nets with Sublinear Memory Cost," arXiv:1604.06174, 2016.
- S. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC 2020.
- V. Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023.
