# Chunked Prefill Token Budget 민감도

## 통제 조건

Input8000/Output200을 유지했다.
QPS4와 QPS8에서 Budget256/512/1024/2048/4096/8192를 비교했다.
각 비교는 S128, 동일 Prompt, Arrival Seed와 Graph Capture 집합을 사용했다.
Per-request Long Prefill Threshold도 동일하게 유지했다.
이는 Budget의 고정 QPS 비교다.
QPS 증가 실험과 합치지 않는다.
각 점은 단일 Start다.
이 Budget Sweep은 Primary 범위 변경 전에 완료한 보충 실험이다.
이후 B512/S16의 추가 실행은 하지 않았다.

| QPS | 설정 | Mean TTFT (s) | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | DP2_B1024_S128_FULL | 0.930 | 38.910 | 67.470 | 45.845 | 720.941 |
| 8 | DP2_B1024_S128_FULL | 4.643 | 43.616 | 52.847 | 48.807 | 855.205 |
| 4 | DP2_B2048_S128_FULL | 0.685 | 50.871 | 96.142 | 68.500 | 717.775 |
| 8 | DP2_B2048_S128_FULL | 4.311 | 59.504 | 96.503 | 72.651 | 851.467 |
| 4 | DP2_B256_S128_FULL | 4.581 | 18.332 | 24.740 | 20.967 | 589.099 |
| 8 | DP2_B256_S128_FULL | 10.783 | 18.180 | 24.399 | 20.924 | 612.754 |
| 4 | DP2_B4096_S128_FULL | 0.705 | 57.583 | 175.592 | 76.611 | 712.795 |
| 8 | DP2_B4096_S128_FULL | 4.772 | 65.058 | 179.331 | 80.311 | 815.126 |
| 4 | DP2_B512_S128_FULL | 1.459 | 26.755 | 35.641 | 27.783 | 703.370 |
| 8 | DP2_B512_S128_FULL | 7.062 | 27.774 | 36.023 | 30.238 | 745.860 |
| 4 | DP2_B8192_S128_FULL | 0.785 | 60.153 | 301.437 | 79.338 | 704.891 |
| 8 | DP2_B8192_S128_FULL | 4.866 | 66.466 | 304.542 | 81.660 | 812.795 |

고정 Capture 최대값은 256이다.
Budget에 따라 실제 Shape와 Graph/Eager 사용 비중은 달라질 수 있다.
이 변화는 실제 Scheduler 경로의 매개 효과다.
Budget512에 Capture512를 추가한 별도 Control도 보존했다.
그 값으로 Graph Coverage 영향을 별도로 확인한다.

| QPS | Capture 최대 Token 수 | Mean TBT (ms) | ITL p99 (ms) | TPOT p95 (ms) | Output Token/s | Mixed Full-graph Replay Counter |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 256 | 26.755 | 35.641 | 27.783 | 703.370 | 5 |
| 4 | 512 | 26.225 | 30.178 | 27.329 | 706.797 | 2017 |
| 8 | 256 | 27.774 | 36.023 | 30.238 | 745.860 | 1 |
| 8 | 512 | 26.824 | 30.269 | 27.421 | 780.833 | 2015 |

이 Control의 다른 설정과 예정 Arrival은 동일하다.
단일 Start의 차이를 확정적인 인과 기여도나 CI로 표현하지 않는다.
큰 Budget이 모든 Workload와 SLO에서 더 효율적이라고 가정하지 않는다.
작은 Budget도 Mixed CUDA Graph를 사용할 수 있는 현재 Runtime의 기존 결과를 보존했다.
Raw Token/s와 Joint SLO-goodput을 함께 평가한다.
모든 수치는 `MEASURED`다.
각 Prefill Chunk 및 실제 Batch 분포는 Raw Metrics에 있다.
