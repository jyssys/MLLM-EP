# Online Encoder 및 Prefill 동시 활동

동일한 QPS1 Main100 도착에서 Qwen2.5-VL의 Graph 기반 1E/1P/1D PoC 다섯 조건이 각각 100건을 완료했다. E 보조와 P 보조는 위치 기반 25%, Visual은 같은 비율에서 Visual row를 우선 배치했다. Admission 시점의 Idle 조건을 사용하는 Adaptive는 24건에 E KV를 배치했다. 측정 범위는 이 PoC의 한 server start씩이며, Stock EPD 기준선은 별도 온라인 성능표에 있다.

## 같은 Decode 작업의 관측 시간

아래 비교는 D GPU, 실제 B1, 요청과 generation step, 전체 KV row 수, 그 step에 입력된 토큰 prefix, Graph backend가 일치하는 step만 사용한다. Decode 시작 시 양쪽의 E/P 동시 활동 분류도 일치한다. 시간은 graph metadata 반영부터 Full28과 샘플링이 호스트에 반환되기까지의 경계로 집계했다. 같은 요청의 여러 step이 중복 포함되므로 표는 표본 분포다.

| 후보 | 양쪽 step 시작 시 활동 | Step 쌍 | Local p50 | 후보 p50 | 쌍별 차이 p50 | 쌍별 차이 p95 |
|---|---|---:|---:|---:|---:|---:|
| E 보조 | E/P idle | 455 | 10.78 ms | 11.72 ms | +0.90 ms | +2.43 ms |
| E 보조 | E active, P idle | 127 | 13.72 ms | 65.62 ms | +51.79 ms | +114.49 ms |
| P 보조 | E/P idle | 608 | 10.78 ms | 11.75 ms | +0.98 ms | +4.11 ms |
| P 보조 | E active, P idle | 466 | 13.76 ms | 14.79 ms | +0.94 ms | +9.11 ms |
| Visual-first E 보조 | E/P idle | 431 | 10.77 ms | 11.80 ms | +1.01 ms | +3.85 ms |
| Visual-first E 보조 | E active, P idle | 91 | 14.36 ms | 67.79 ms | +53.31 ms | +102.73 ms |
| Adaptive E 보조의 assisted step | E/P idle | 468 | 10.78 ms | 11.80 ms | +1.00 ms | +2.01 ms |
| Adaptive E 보조의 assisted step | E active, P idle | 21 | 12.87 ms | 44.74 ms | +30.73 ms | +71.88 ms |

E/P가 step 전체에서 비어 있던 쌍에서도 보조의 p50 비용은 위치 기반 E **+0.90 ms**(449쌍), P **+0.97 ms**(605쌍), Visual-first E **+1.01 ms**(427쌍)다. Encoder가 동작 중일 때 E 보조에서 큰 추가 지연이 관측됐다. Prefill 보조의 P-only 매치가 3쌍이라 P가 동작 중일 때의 효과를 추정하기에는 표본이 부족하다. 후보별 전체 Decode step의 E host enclosure 중첩 시간은 Local **69.13%**, E 보조 **79.38%**, P 보조 **67.31%**, Visual-first **80.24%**, Adaptive **73.80%**였다. Step이 길어질수록 같은 Encode 구간과 겹칠 시간이 늘어날 수 있으므로 중첩 비율의 변화는 원인으로 해석하기 어렵다.

## Encoder와 큐의 동반 변화

같은 요청의 Encoder CUDA-event enclosure p95는 Local **1,865 ms**, E 보조 **2,204 ms**, P 보조 **1,870 ms**, Visual-first **2,235 ms**, Adaptive **1,963 ms**였다. 쌍별 Encoder enclosure 중앙값 차이는 각각 **+67.13**, **−3.14**, **+62.69**, **+6.33 ms**다. 위치 기반 E 보조에서 D queue p95는 **21.37→108.01 ms**, 초기 KV 배치 p95는 **38.17→172.40 ms**였다. P Prefill 경과 시간 p95는 **1,016.67→1,018.87 ms**였다. 상이한 Pair 분포와 대기 구간이 TTFT에 함께 작용한다.

Adaptive에서 실제 보조 step은 **1,517/3,341**이었다. 그 중 **392/1,517**은 시작 시 E가 이미 다른 요청을 인코딩 중이었다. KV 배치는 Decode admission 시점에 한 번 결정되므로, 이미 원격에 존재하는 KV에 대한 나중의 attention은 새 Encode와 겹칠 수 있다. Adaptive 보조 step의 **434/1,517**은 혼합 로컬/원격 batch에서 Eager 경로를 사용했다. 위 Graph 비교는 경로가 일치하는 step을 사용한다.

E/P 상태는 각각 Encoder와 Prefill의 호스트 경과 구간에서 계산했다. Encoder 호스트 구간으로 계산한 D step 시작의 상태와 D가 읽은 Encoder flag는 이 다섯 회차에서 모두 일치했다. 이 분석은 독립적으로 진행된 실제 큐에서 관측한 상관관계이며 SM, HBM, NVLink, host enqueue 및 kernel preemption의 지연을 개별적으로 식별할 수 없다. 신뢰구간을 위해서는 독립 서버 시작에서 동일 trace를 반복해야 한다. [실제 GPU 간섭의 고립 분석](../../reports/mllm_encoder_assisted_decode_attention_microbenchmark.md)

## 재현

```bash
raw=../h200_eada_runs
python3 -m poc_flashvep.encoder_assisted_decode_attention.online_interference \
  --runs "$raw/img_local_q1/cohort_1.json" "$raw/img_encoder_q1/cohort_1.json" \
  "$raw/img_prefill_q1/cohort_1.json" "$raw/img_visual_q1/cohort_1.json" \
  "$raw/img_adaptive_q1/cohort_1.json" --out /tmp/eada-online-interference-repro.json
```

원본 SHA-256, 실제 batch 및 context band별 분포, matched step과 요청별 단계 분포를 JSON에 기록했다. [분석 데이터](online_interference_diagnostic.json)
