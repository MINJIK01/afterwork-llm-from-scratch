# Step 09 확인 문제 해설

[← 본문](../09_block.md)

## 1. SiLU의 도함수

$\mathrm{SiLU}(z) = z\sigma(z)$, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\mathrm{SiLU}'(z) = \sigma(z) + z\sigma(z)(1-\sigma(z)) = \sigma(z)\big(1 + z(1 - \sigma(z))\big).$$

예: $z = -2$이면 $\sigma = 0.119$, 도함수 $= 0.119 \times (1 - 2\times0.881) \approx -0.091$. 음수 입력에서도 0이 아니다(약 $z < -1.28$에서는 음수). ReLU는 $z < 0$에서 도함수가 정확히 0이어서, 입력이 계속 음수인 뉴런은 gradient를 받지 못해 회복되지 않는다(dead ReLU). SiLU는 매끄럽고 비단조이며 음수 영역에서도 작은 gradient가 흐른다. 최솟값은 $z \approx -1.278$에서 $\approx -0.278$.

## 2. Pre-norm에서 final norm을 빼면?

residual stream $x_L = x_0 + \sum_{k=1}^{2L} f_k$에는 정규화가 없으므로, 크기(RMS)가 깊이와 학습 진행에 따라 커진다. final norm 없이 lm_head에 넣으면 logits 크기가 그대로 커져 softmax가 사실상 낮은 temperature로 날카로워지고, 크기가 학습 중 변해 출력 분포가 불안정해진다. 블록 내부는 각 서브레이어 앞에서 정규화하므로 문제가 없지만, 마지막 출력에는 별도의 정규화가 필요하다. post-norm은 각 블록 출력이 이미 정규화되어 있어 final norm이 필요 없다.

## 3. RMSNorm이 불변인 변환

- 양의 스칼라배 $x \to ax$ ($a > 0$): $ax/\mathrm{rms}(ax) = x/\mathrm{rms}(x)$ → **불변** ($\epsilon$ 무시).
- 음의 스칼라배: 부호가 뒤집힌다(불변 아님).
- 상수 더하기 $x \to x + c\mathbf 1$: **불변 아님**.

LayerNorm은 평균을 빼므로 상수 더하기와 양의 스칼라배 모두에 불변이다. 이 차이가 step 03 해설 2번에서 본 backward 식의 $-\overline h$ 항 유무로 나타난다.
