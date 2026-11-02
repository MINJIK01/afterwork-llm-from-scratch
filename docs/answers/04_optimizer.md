# Step 04 확인 문제 해설

[← 본문](../04_optimizer.md)

## 1. $\epsilon$을 $\sqrt{\hat v + \epsilon}$ 안에 넣으면

업데이트 $\Delta = \eta\,\hat m/\sqrt{\hat v + \epsilon}$. gradient가 매우 작은 좌표($\hat v \ll \epsilon$)에서

- 바깥($\sqrt{\hat v} + \epsilon$, $\epsilon = 10^{-8}$): $\Delta \approx \eta\,\hat m/|\hat m|$ 수준. gradient 크기와 무관하게 거의 $\pm\eta$만큼 움직인다.
- 안쪽: $\Delta \approx \eta\,\hat m/\sqrt\epsilon = \eta\,\hat m\cdot 10^{4}$. gradient에 비례하는 SGD처럼 동작하고 업데이트가 훨씬 작다.

즉 안쪽 $\epsilon$은 바깥 $\sqrt\epsilon$과 같은 효과를 낸다. 같은 숫자를 쓰면 실질적인 damping이 크게 달라지므로, 다른 구현이나 논문의 하이퍼파라미터를 가져올 때는 $\epsilon$의 위치와 bias correction 적용 여부를 반드시 확인해야 한다.

## 2. loss를 $k$배 해도 Adam 업데이트가 거의 같은 이유

$g \to kg$이면 $m \to km$, $v \to k^2 v$, 따라서 $\hat m/\sqrt{\hat v} \to k\hat m/(k\sqrt{\hat v})$로 불변이다($\epsilon$ 무시). Adam은 좌표별로 gradient의 스케일을 정규화한다.

SGD는 $\Delta = \eta k g$로 업데이트가 $k$배가 된다. loss 스케일을 바꾸면 lr을 다시 맞춰야 한다. fp16 loss scaling(step 23)에서 unscale을 반드시 해야 하는 이유도 같다. Adam은 업데이트 방향 자체는 거의 불변이지만 $\epsilon$의 상대적 크기와 gradient clipping 임계값이 달라진다.

## 3. Gradient accumulation과 clipping의 순서

**누적이 끝난 뒤 한 번** clip해야 한다. 누적된 gradient가 전체 배치 gradient이고, clipping은 비선형 연산($g \cdot \min(1, c/\|g\|)$)이다. 마이크로배치마다 clip하면 그 합은 전체 배치 gradient를 clip한 것과 달라져 "accumulation = 큰 배치"라는 동치가 깨진다. 예: 마이크로배치 gradient의 norm이 각각 3, 3이고 방향이 반대면 전체 gradient는 0인데, 각각 1로 clip해도 합은 0이다. 반면 방향이 같으면 전체 norm 6을 1로 줄여야 할 것을 각각 1로 줄여 합이 2가 된다. 이 레포의 `train_step`은 누적 후 한 번 clip한다.
