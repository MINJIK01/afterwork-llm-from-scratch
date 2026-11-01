# ClaudePlayground

논문의 수식을 코드로 옮기는 연습을 위한 레포.

## 방법

각 `exercises/NN_<name>/` 디렉터리는 하나의 연습 문제다.

- `SPEC.md`: 대상 논문의 수식과 표기법(기호 → 텐서 shape 매핑)
- `impl.py`: 함수 시그니처만 있고 본문은 `NotImplementedError`. 이 파일의 TODO를 채우는 게 연습이다.
- `test_impl.py`: 신뢰할 수 있는 레퍼런스(torch 내장 함수 또는 autograd)와 비교해 정답 여부를 자동 채점한다.

## 실행

```bash
source .venv/bin/activate
pytest exercises/01_softmax_xent_backward -v
pytest exercises/02_attention -v
```

## 난이도 순서

1. `01_softmax_xent_backward` — softmax + cross-entropy의 해석적 역전파 유도
2. `02_attention` — Vaswani et al. (2017) scaled dot-product attention

다음 후보: layernorm backward, VAE ELBO (Kingma & Welling 2013), PPO clipped objective (Schulman et al. 2017), DDPM forward/reverse process (Ho et al. 2020). 필요하면 요청해서 추가.
