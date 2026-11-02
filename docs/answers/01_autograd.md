# Step 01 확인 문제 해설

[← 본문](../01_autograd.md)

## 1. `+=` 대신 `=`이면 `b = a * a + a`의 `a.grad`는?

$a = 3$일 때 정답은 $\partial b/\partial a = 2a + 1 = 7$이다.

그래프: `m = a * a` (mul, `self`와 `other`가 모두 `a`), `b = m + a` (add). 역위상 순서로 `b._backward` → `m._backward` 순으로 실행된다.

1. `b._backward` (add): `m.grad = 1`, `a.grad = 1`
2. `m._backward` (mul): `a.grad = other.data * out.grad = 3` (덮어씀), 이어서 `a.grad = self.data * out.grad = 3` (다시 덮어씀)

결과는 **3**이다. add에서 받은 기여 1과 mul의 두 경로 기여 3+3이 모두 합쳐져야 7이 되는데, 대입은 마지막 값만 남긴다. 한 노드가 여러 소비자를 가지는 경우(2.1의 합)를 처리하려면 누적이 필수이다.

## 2. `backward()`를 두 번 연속 호출하면?

루트의 grad만 1로 다시 설정되고, 다른 노드의 grad는 첫 호출 값에서 계속 누적된다. 단순히 2배가 되는 것이 아니라 **중간 노드의 누적값이 다시 전파되어 더 커진다**. 예: $y = 2x$, $z = 3y$.

| | `z.grad` | `y.grad` | `x.grad` |
|---|---|---|---|
| 1회 후 | 1 | 3 | 6 |
| 2회 후 | 1 | 3 + 3 = 6 | 6 + 2·6 = **18** |

(실제 실행 결과 6 → 18로 확인.) 깊이가 깊을수록 오차가 더 커진다.

PyTorch는 두 가지로 이 문제를 피한다. (1) backward가 끝나면 그래프의 중간 버퍼를 해제하므로(`retain_graph=False`) 같은 그래프로 두 번 backward하면 오류가 난다. (2) `.grad`는 leaf 텐서(파라미터)에만 누적된다. 대신 매 iteration마다 새 그래프를 만들어도 파라미터의 `.grad`는 계속 더해지므로, step 시작 시 `optimizer.zero_grad()`로 비워야 한다. 이 누적 동작을 의도적으로 쓰는 것이 gradient accumulation(step 12)이다.

## 3. 출력 $m$개, 입력 $n$개 함수의 Jacobian

- Forward mode: 한 pass가 Jacobian의 한 **열**($J e_i$)을 준다 → $n$번.
- Reverse mode: 한 pass가 한 **행**($e_j^\top J$)을 준다 → $m$번.

$m \ll n$(신경망 loss: $m = 1$)이면 reverse mode, $n \ll m$이면 forward mode가 유리하다. 둘을 섞어 Hessian-vector product를 계산하는 방식(forward-over-reverse)도 있다.
