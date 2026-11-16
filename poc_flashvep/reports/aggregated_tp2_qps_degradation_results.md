# Aggregated TP2의 정합성 및 부하 Diagnostic

## 적격성 판정

TP2의 strict TP1 Output Parity는 `FAILED`다.
주 성능 비교의 적격 TP2 Curve는 `NOT_MEASURED`다.
실행 자체의 실패와 출력 정합성 실패를 구분한다.
TP2는 표준 Continuous vLLM의 단일 TP2 Instance로 실행했다.
두 TP2 Replica를 사용하지 않았다.
두 GPU가 전체 36개 Layer의 Tensor Shard를 보유한다.
Original BF16과 모델 Revision을 유지했다.

| 진단 실행 | TP1 불일치 요청/8 | 첫 TP2와 달라진 요청 |
| --- | --- | --- |
| qual_TP2_B8192_S1024_FULL_N8_O200_q4_s2026111601_v1_repeat2 | 7 | 0 |
| diagnostic_TP2_sequential_B8192_N8_O200_v1 | 7 | 0 |
| diagnostic_TP2_B512_S16_FULL_N8_O200_q4_v1 | 7 | 0 |

반복, 순차 Eager와 Budget512에서 같은 TP2 출력이 나왔다.
그러나 TP1 Reference와 다른 Token Sequence를 생성했다.
이 진단은 Chunk Boundary나 Mixed Batch만으로 차이가 생겼다는 설명을 지지하지 않는다.
TP 분할에 따른 BF16 Reduction 순서는 가능한 원인이다.
동일 Prefix의 Logit과 Reduction별 비교가 없어 배타적 원인은 `NOT_IDENTIFIED`다.
차이를 무조건 허용 오차로 처리해 PASS로 바꾸지 않았다.

## Within-TP2 부하 진단

아래 수치는 `MEASURED`이지만 Primary Comparison에는 부적격이다.
별도의 TP2 N64 Reference를 사용했다.
독립 순차 Eager N8 출력과 그 Prefix를 비교했다.
이는 TP1 출력 또는 전체 KV/Hidden 정합성을 해결한 것이 아니다.
같은 TP2 안에서 QPS 변화의 양상을 설명하는 Diagnostic이다.
각 점은 N128의 단일 Server Start다.
TP2 Client는 Engine의 직접 수신 경계다.
DP2 Coordinator의 IPC 전달 경계와 다르다.
따라서 절대 Latency를 동일 Client 경계로 해석하지 않는다.

| Method | Offered QPS | Start 수 | Mean TTFT (s) | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TP2_B512_S16_FULL | 2 | 1 | 7.518 | 34.848 | 39.720 | 37.344 | 320.909 |
| TP2_B512_S16_FULL | 3 | 1 | 25.125 | 43.238 | 52.067 | 44.719 | 268.378 |
| TP2_B512_S16_FULL | 4 | 1 | 21.331 | 35.121 | 40.872 | 36.220 | 330.140 |
| TP2_B512_S16_FULL | 5 | 1 | 33.267 | 44.116 | 73.678 | 64.243 | 266.962 |
| TP2_B512_S16_FULL | 6 | 1 | 26.423 | 35.014 | 39.577 | 35.961 | 331.552 |
| TP2_B512_S16_FULL | 7 | 1 | 27.881 | 34.967 | 39.729 | 35.907 | 331.994 |
| TP2_B512_S16_FULL | 8 | 1 | 29.230 | 35.153 | 40.029 | 36.102 | 330.209 |
| TP2_B512_S16_FULL | 9 | 1 | 30.041 | 35.104 | 39.733 | 36.099 | 330.732 |
| TP2_B512_S16_FULL | 10 | 1 | 30.610 | 34.993 | 39.509 | 36.023 | 331.788 |
| TP2_B8192_S1024_FULL | 2 | 1 | 1.916 | 110.755 | 430.057 | 170.926 | 375.277 |
| TP2_B8192_S1024_FULL | 3 | 1 | 8.706 | 130.434 | 430.471 | 173.630 | 399.348 |
| TP2_B8192_S1024_FULL | 4 | 1 | 13.255 | 131.609 | 428.013 | 175.123 | 402.812 |
| TP2_B8192_S1024_FULL | 5 | 1 | 16.247 | 132.641 | 427.317 | 176.783 | 402.821 |
| TP2_B8192_S1024_FULL | 6 | 1 | 18.040 | 131.549 | 427.130 | 174.849 | 406.348 |
| TP2_B8192_S1024_FULL | 7 | 1 | 19.604 | 131.636 | 426.889 | 174.937 | 405.911 |
| TP2_B8192_S1024_FULL | 8 | 1 | 20.721 | 131.622 | 427.207 | 174.950 | 406.039 |
| TP2_B8192_S1024_FULL | 9 | 1 | 21.598 | 131.603 | 426.785 | 174.912 | 406.084 |
| TP2_B8192_S1024_FULL | 10 | 1 | 22.264 | 131.585 | 427.009 | 174.969 | 406.357 |

TP Collective의 실제 GPU 실행시간과 Host 비용 분해는 `NOT_MEASURED`다.
TP2가 본질적으로 DP2보다 느리거나 빠르다고 일반화하지 않는다.
논문 Figure 7은 Qwen3-14B TP2다.
현재 Qwen3-8B 결과로 Figure 7의 정확한 재현을 주장하지 않는다.
