# Step 24 확인 문제 해설

[← 본문](../24_lora.md)

## 1. m5 프리셋 전체에 $r = 8$을 적용할 때의 학습 파라미터

레이어당 $r(d_{in} + d_{out})$:

| 모듈 | $d_{in}\to d_{out}$ | 파라미터 |
|---|---|---|
| `qkv` | 384 → 1152 | $8\times1536 = 12{,}288$ |
| `proj` | 384 → 384 | $8\times768 = 6{,}144$ |
| `w_gate`, `w_up` | 384 → 1024 | 각 $8\times1408 = 11{,}264$ |
| `w_down` | 1024 → 384 | $11{,}264$ |
| 합 | | $52{,}224$ |

6레이어 합 **313,344개**, 전체 12,194,688개의 **2.57%** (`apply_lora` 후 `trainable_params`로 확인). optimizer 상태도 이 파라미터에 대해서만 필요하다.

## 2. $A$와 $B$를 모두 작은 랜덤 값으로 초기화하면?

학습 시작 시 $\Delta W = \frac{\alpha}{r}BA \ne 0$이므로 모델이 사전학습 모델과 다른 상태에서 출발한다. 랜덤한 섭동이 모든 대상 Linear에 더해져, 학습 초반에 사전학습 능력을 일부 잃은 상태에서 시작하고 그 손상을 복구하는 데 스텝을 쓴다. $B = 0$이면 $\Delta W = 0$에서 출발해 정확히 사전학습 모델과 같은 상태에서 시작한다. 반대로 둘 다 0이면 $\partial L/\partial A \propto B^\top = 0$, $\partial L/\partial B \propto Ax = 0$이 되어 영원히 학습되지 않는다. 그래서 하나만 0으로 둔다.

## 3. $\Delta W$의 특이값으로 $r$ 정하기

$\Delta W = \frac\alpha rBA$는 rank가 $r$ 이하이다. 특이값 $\sigma_1 \ge \dots \ge \sigma_r$의 에너지 분포를 보고, 누적 에너지 $\sum_{i\le k}\sigma_i^2/\sum\sigma_i^2$가 예컨대 90%에 도달하는 $k$가 $r$보다 훨씬 작으면 $r$을 줄여도 된다. 반대로 에너지가 고르게 퍼져 있으면 $r$이 부족할 수 있다.

```python
import torch
from minillm.lora import LoRALinear
for name, m in model.named_modules():
    if isinstance(m, LoRALinear):
        s = torch.linalg.svdvals(m.scaling * m.B @ m.A)
        energy = (s**2).cumsum(0) / (s**2).sum()
        print(name, "90% 에너지 rank:", int((energy < 0.9).sum()) + 1, "/ r =", m.A.shape[0])
```

`merge_lora` 전에 실행해야 한다. Hu et al.은 서로 다른 $r$로 학습한 $\Delta W$의 상위 특이벡터 부분공간이 크게 겹친다는 것을 보이며, 매우 작은 $r$(1–4)로도 많은 정보가 담긴다고 분석했다.
