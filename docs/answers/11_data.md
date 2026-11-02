# Step 11 확인 문제 해설

[← 본문](../11_data.md)

## 1. 문서 단위 padding vs packing 효율

배치를 배치 내 최대 길이로 padding하면 유효 토큰 비율은

$$\eta_{pad} = \frac{\mathbb E[\ell]}{\mathbb E[\max_{b} \ell_b]}.$$

길이 분산이 크거나 배치가 클수록 최대 길이가 평균보다 커져 효율이 떨어진다. 예: 평균 200, 표준편차 100인 분포에서 배치 32의 최대 길이는 대략 평균 + 2표준편차 ≈ 400 정도이므로 효율이 약 50%이다. 길이별 bucketing으로 개선할 수 있지만 완전하지 않다. packing은 모든 위치가 유효 토큰이므로 $\eta \approx 100\%$이다. 대가는 문서 경계를 넘는 attention이다.

## 2. 상한을 `len(data) - block_size + 1`로 두면?

`torch.randint(high)`는 high를 포함하지 않으므로 시작점 최댓값이 `len - block_size`가 된다. 그 시작점에서 `y = data[i+1 : i+1+block_size]`의 끝 인덱스가 `len + 1`이 되어, numpy 슬라이스가 조용히 잘려 길이 `block_size - 1`이 된다. 그러면 `torch.stack`이 shape 불일치로 실패한다. 확률 $1/(\text{len} - T + 1)$로만 일어나므로 테스트에서 놓치기 쉽다.

## 3. 문서 경계 마스크에 필요한 정보

위치별 문서 id $\mathrm{doc}[t]$가 필요하다. EOS 위치로부터 `doc = cumsum(x == eos)`(EOS를 앞 문서에 포함시키려면 한 칸 shift)로 만들 수 있다. 마스크는

$$\mathrm{mask}[i, j] = (j \le i) \wedge (\mathrm{doc}[i] = \mathrm{doc}[j]).$$

배치마다 마스크가 다르므로 `(B, 1, T, T)` 형태로 브로드캐스트한다. 문서마다 위치를 0부터 다시 세려면 RoPE 위치 인덱스도 문서별로 다시 시작해야 한다. FlashAttention의 varlen 인터페이스는 마스크 대신 문서 경계 offset으로 같은 일을 효율적으로 처리한다.
