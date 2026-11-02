# Step 13 확인 문제 해설

[← 본문](../13_sampling.md)

## 1. top-p = 1, top-k = None, τ = 1이면?

모델 분포 $p_\theta(\cdot\mid x_{<t})$에서 그대로 샘플링한다(ancestral sampling). 생성된 시퀀스의 분포가 모델이 학습한 분포와 정확히 같다. 수치적으로는 누적합이 1을 약간 넘거나 모자라는 float 오차 때문에 `cum - p >= 1.0`인 꼬리 토큰이 드물게 잘릴 수 있다. 정확한 동작이 필요하면 `top_p is None`으로 필터를 끄는 편이 안전하다.

## 2. τ < 1과 top-p의 순서를 바꾸면?

결과가 달라진다. top-p 집합은 분포 모양에 따라 정해지고, temperature가 모양을 바꾸기 때문이다.

- τ 먼저(이 레포): τ < 1로 분포가 날카로워진 뒤 누적합이 $p$에 빨리 도달하므로 nucleus가 **작아진다**.
- top-p 먼저: τ = 1 분포로 집합을 정한 뒤 그 안에서 τ를 적용하므로 nucleus가 더 크다. 집합 내부 분포만 날카로워진다.

HF transformers 등 대부분의 구현은 temperature를 먼저 적용한다. 순서가 구현마다 다르면 같은 설정값이라도 다른 결과가 나오므로, 실험 재현 시 확인해야 한다.

## 3. Repetition penalty

`filter_probs` 앞에서 logit을 조정한다. 두 가지 대표 형태:

- CTRL(Keskar et al. 2019): 이미 등장한 토큰 $i$에 대해 $z_i \leftarrow z_i/\theta$ ($z_i > 0$), $z_i \leftarrow z_i\theta$ ($z_i < 0$), $\theta > 1$. 부호에 따라 나누거나 곱해야 항상 확률이 낮아진다.
- frequency/presence penalty(OpenAI API): $z_i \leftarrow z_i - \alpha\cdot\mathrm{count}_i - \beta\cdot\mathbb 1[\mathrm{count}_i > 0]$.

```python
def repetition_penalty(logits, generated, theta=1.2):
    z = logits.gather(-1, generated)
    return logits.scatter(-1, generated, torch.where(z > 0, z / theta, z * theta))
```
