# Step 12 확인 문제 해설

[← 본문](../12_training.md)

## 1. 마이크로배치 크기가 2, 3일 때 단순 $1/K$ 가중

$L = \frac12(\bar L_1 + \bar L_2)$, $\bar L_1 = \frac12(l_1 + l_2)$, $\bar L_2 = \frac13(l_3 + l_4 + l_5)$이므로 샘플 가중치는 마이크로배치 1의 샘플이 각 $1/4$, 마이크로배치 2의 샘플이 각 $1/6$이다. 전체 평균이라면 모두 $1/5$이어야 한다. 크기가 작은 마이크로배치의 샘플이 과대 반영된다. 정확히 맞추려면 마이크로배치 loss를 합(sum)으로 계산하고 전체 샘플(토큰) 수로 나누거나, $n_k/N$으로 가중한다. SFT처럼 마이크로배치마다 유효 토큰 수가 다를 때 같은 문제가 생긴다.

## 2. 배치를 키우고 스텝을 줄이는 trade-off

- 장점: gradient 노이즈가 줄고, 하드웨어 병렬성을 더 잘 활용해 시간당 처리 토큰이 늘어난다.
- 단점: 같은 토큰 수에서 업데이트 횟수가 줄어든다.

McCandlish et al.의 critical batch size $B_{crit}$보다 작은 배치에서는 배치를 2배로 하면 필요한 스텝이 거의 절반이 된다(토큰 효율 유지). $B_{crit}$보다 크면 스텝 감소가 둔화되어 같은 loss에 더 많은 토큰이 필요하다. $B_{crit}$는 gradient noise scale로 추정하며, loss가 낮아질수록 커진다. 그래서 대형 학습은 학습 중 배치를 점차 키우기도 한다.

## 3. 이 Mac의 MFU 추정

$$\mathrm{MFU} = \frac{6N\cdot\mathrm{tok/s}}{\text{하드웨어 피크 FLOPs}}.$$

예: 로그가 20,000 tok/s이면 $6\times12.2\text{M}\times2\times10^4 \approx 1.46$ TFLOPS. 분모는 칩 사양값 대신 실제로 달성 가능한 행렬곱 처리량을 측정해 쓰는 것이 현실적이다.

```python
import time, torch
x = torch.randn(4096, 4096, device="mps"); y = torch.randn(4096, 4096, device="mps")
for _ in range(3): x @ y
torch.mps.synchronize(); t = time.perf_counter()
for _ in range(20): x @ y
torch.mps.synchronize()
print(2 * 4096**3 * 20 / (time.perf_counter() - t) / 1e12, "TFLOPS (fp32 matmul)")
```

작은 모델은 커널 실행 오버헤드와 element-wise 연산(softmax, norm, 직접 구현한 cross-entropy) 비중이 커서 MFU가 낮게 나오는 것이 정상이다.
