# Step 14 확인 문제 해설

[← 본문](../14_kv_cache.md)

## 1. m5 프리셋의 KV cache 크기

$$2\cdot L\cdot B\cdot H\cdot d_h\cdot T\cdot 4 = 2\times6\times1\times6\times64\times256\times4 = 4{,}718{,}592\text{ B} = 4.5\text{ MiB}.$$

가중치는 $12{,}194{,}688\times4 \approx 46.5$ MiB이므로 배치 1에서 cache는 가중치의 약 10%이다. 배치 10이면 cache가 가중치와 같아진다. 큰 모델·긴 문맥·큰 배치에서 cache가 메모리를 지배하는 이유이다.

## 2. 프롬프트 길이가 다른 배치 생성

이 구현은 배치 전체가 하나의 `pos`를 공유하므로, 프롬프트 길이가 다르면 시퀀스마다 위치가 어긋나고 짧은 프롬프트 뒤의 padding까지 참조한다. 해결 방법:

1. **Left padding**: 모든 프롬프트가 같은 위치에서 끝나도록 왼쪽을 padding한다. 그러면 새 토큰들의 `pos`가 배치 전체에서 같다.
2. **Attention mask**: padding key를 모든 query에서 가리는 `(B, 1, 1, Tk)` 마스크를 causal mask와 AND한다.
3. **위치 보정**: RoPE 위치를 시퀀스별로 "padding 수만큼" 빼서, 실제 첫 토큰이 위치 0이 되게 한다.

vLLM 같은 서빙 엔진은 padding 없이 시퀀스별로 길이가 다른 cache를 블록 단위로 관리한다(PagedAttention).

## 3. Cache의 속도 향상이 이론값보다 작은 이유

FLOPs로는 cache가 생성 길이 $n$에 대해 $O(n)$배를 줄이지만, 이 모델은 매우 작아서 한 스텝의 시간이 연산량이 아니라 **고정 비용**에 좌우된다. 레이어당 수십 개 커널의 실행 오버헤드, Python 루프, 동기화가 그것이다. cache 없이 $t$개 토큰을 처리하는 forward도 GPU에서는 병렬로 처리되어 1개 토큰 forward와 시간이 크게 다르지 않다. 또 cache 버퍼 슬라이싱과 쓰기 비용이 추가된다. 모델이 크고 문맥이 길수록 실제 속도비가 이론값에 가까워진다.
