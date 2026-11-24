# Low-load CUDA Event 계측 검증 — C20

QPS0.5의 Clean/Profile/Clean은 고정한 5% 계측 Gate를 통과했다. 독립 초기화는 1회다.
이 판정은 실행 구간 계측에만 적용한다. Kernel-exclusive Busy 또는 Safe Headroom의 PASS가 아니다.
Main100 Equal-output Replay를 사용했다. 각 실행은 원본 요청 100개와 Output5269 Tokens다.
GPU4/5/6은 E/P/D다. Original BF16, BI0, 원본 Compiled 모델과 동일한 Arrival을 유지했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.

## 관측 구간

Central Arrival Window는 예정 Arrival p10–p90 구간이다. 길이는 157.172988초다.
이 구간은 Low Load다. Prefill-heavy 또는 Sustainable Steady State라고 부르지 않는다.

| Stage | Execute 구간 비율 | 구간 밖 비율 | 관측 Waiting 평균 | 관측 Waiting 최대 |
|---|---:|---:|---:|---:|
| E | 13.7370% | 86.2630% | 0.0000 | 0 |
| P | 22.2888% | 77.7112% | 0.0520 | 3 |
| D | 15.2372% | 84.7628% | 0.0000 | 0 |

Waiting은 Scheduler Sample의 값이다. 정확한 시간 가중 Queue는 아니다.
E/D가 동시에 Execute 구간 밖인 시간은 Central Window의 72.3765%다.
그 시간과 P Execute 구간의 겹침은 Central Window의 10.6240%다.
같은 겹침은 P Execute 구간 시간의 47.6652%다. 두 분모를 합치지 않는다.
이 비율은 Physical Idle Overlap이나 Actionable Slack이 아니다.

E의 구간 밖 Window는 Count 기준 p50 813.062ms, p95 4698.545ms다.
D는 p50 0.023438ms, p95 1.625787ms다. D의 p99는 1201.151ms다.
D의 짧은 Gap 수가 많다는 사실과 긴 Gap이 차지하는 시간 비율을 구분한다.
0.1/1/5/10/20/50/100ms의 Count 및 Duration 가중 Coverage는 대응 CSV에 있다.

## 한계와 판정

Event는 원래 model-runner Execute를 둘러싼다. 그 안에는 Host Gap과 Stream Dependency가 있을 수 있다.
구간 밖에서 별도 Stream의 Copy나 Export가 실행될 수 있다.
Endpoint Clock Bracket의 최대 폭은 E252.344µs/P191.359µs/D163.526µs다.
매우 짧은 Cross-GPU Gap의 정밀한 선후관계를 이 수치로 확정하지 않는다.
추가 Prefill은 삽입하지 않았다. E/D-safe Quantum과 Safe Headroom은 미측정이다.
QPS2의 C19 실패와 QPS4의 CUPTI 실패도 그대로 유지한다.
Low-load 통과를 High-load 계측이나 Native 전체 Task Quality의 통과로 확대하지 않는다.

원시 Event, Clock Anchor와 Scheduler Sample은 `raw/gpu_enclosures/ec_GPU_ENCLOSURES_C20_CPC_q0.5_v0/`에 있다.
재계산 JSON과 CSV는 `analysis/ec_GPU_ENCLOSURES_C20_CPC_q0.5_v0_enclosure_analysis_v0.json` 및 대응 CSV다.
Threshold Figure와 CSV는 `figures/low_load_C20_outside_bracket_threshold_coverage_v0.{png,pdf,csv}`다.
Timeline은 `figures/low_load_C20_aligned_execute_timeline_v0.{png,pdf,csv}`다.
경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
