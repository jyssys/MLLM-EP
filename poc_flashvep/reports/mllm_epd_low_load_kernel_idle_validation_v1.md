# Low-load 실제 Kernel Idle 검증 — C25

QPS0.5의 Full CUPTI Clean/Profile/Clean이 고정한 5% 계측 Gate를 통과했다.
E/D의 물리적 Kernel Idle과 동시 Idle은 관측됐다. 이 조건은 Prefill-heavy가 아니다.
추가 Prefill은 실행하지 않았다. Encode-safe와 Decode-safe Headroom은 `NOT_MEASURED`다.

## 계약과 증거 수준

증거 수준은 `TRACE_ANALYSIS`다. 실제 Main100 Online EPD 실행의 Kernel/Copy Activity를 사용했다.
Task Quality는 미확정이다. 이 보고서는 품질 PASS 또는 Online Token Split 성과가 아니다.
Original Qwen2.5-VL-7B BF16, BI0, Full28, 원본 Processor와 동일한 Arrival을 유지했다.
모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
GPU4/5/6은 E/P/D다. B8192/S1024, KV42GiB, Block128과 원래 Compiled 경로를 사용했다.
각 실행은 Main100과 고정한 Output5269 Tokens다. 독립 초기화는 1회다.

## 계측 Gate

| Metric | Profile Difference % | Clean Spread % | Gate |
|---|---:|---:|---|
| duration_seconds | 0.0136 | 0.0149 | PASS |
| TTFT_p50_ms | -0.7676 | 0.3865 | PASS |
| TTFT_p95_ms | -1.5221 | 0.4269 | PASS |
| ITL_p99_ms | 1.1577 | 1.7144 | PASS |
| TPOT_p95_ms | 1.5535 | 2.4453 | PASS |

차이는 비계측 두 실행의 평균 대비 값이다. Clean Spread도 같은 평균을 분모로 사용한다.
모든 Metric과 Collector 검사가 통과했다. 이전 QPS2/4의 실패 판정은 변경하지 않았다.

## 같은 시간 구간의 E/P/D 상태

Central Arrival Window는 예정 Arrival p10–p90 구간이다. 길이는 157.172988초다.
마지막 Drain과 긴 초기 구간은 아래 분모에서 제외했다. Sustainable Steady State를 뜻하지 않는다.

| Stage | Kernel Busy % | Kernel Idle % | Kernel+Copy Quiet % | Observed Waiting Mean | Waiting Max | Actual Scheduled Max |
|---|---:|---:|---:|---:|---:|---:|
| encoder | 12.7326 | 87.2674 | 87.1527 | 0.509677 | 1 | 1 |
| prefill | 14.0682 | 85.9318 | 85.8897 | 0.041166 | 2 | 3 |
| decode | 14.3716 | 85.6284 | 85.1970 | 0.019316 | 1 | 4 |

Waiting은 Scheduler 관측별 평균이다. 정확한 시간 가중 Queue 평균이 아니다.
Actual Scheduled Max는 관측한 요청 수다. B/S 상한 또는 Prefill/Decode Batch가 모두 같다는 뜻이 아니다.

E/D 동시 Kernel Idle은 Central Window의 74.2376%다.
그 시간과 P Busy의 겹침은 전체 Central Window의 9.9946%다.
같은 겹침은 P Busy 시간의 71.0440%다.
두 분모를 혼합하지 않는다. 동시 Idle이 동일 요청의 3GPU 병렬화를 증명하지는 않는다.

## Window 해석과 한계

Kernel Gap 수에는 작업 내부의 매우 짧은 Gap이 많다. Count CDF와 Duration 가중 Coverage를 구분했다.
100ms 이상 Gap은 E Idle Time의 99.3391%, D Idle Time의 98.9227%를 차지했다.
같은 기준의 Window Count 비율은 E0.091111%, D0.007100%다.
이 값은 미래 Idle Duration을 미리 아는 Scheduler가 실행 가능하다는 뜻이 아니다.

Pending HTTP가 없는 Kernel Idle과 Pending HTTP가 있는 Kernel Idle을 따로 기록했다.
Host Scheduler, CPU Media Preparation, Transfer Wait와 Downstream Backpressure의 정확한 분해는 미완료다.
따라서 모든 Gap에 TRUE_ENCODE_IDLE 또는 Actionable Slack Label을 부여하지 않았다.
Kernel Busy는 SM Active, Tensor Core Activity 또는 HBM Bandwidth Utilization이 아니다. 그 Counter는 미측정이다.
GPU 간 짧은 Gap의 선후관계에는 Clock Anchor 오차가 있다. 원시 Anchor 폭은 JSON에 보존했다.
추가 Prefill과 Encoder/Decode SLO 비열등성은 별도 Online Intervention이 필요하다.

원시 Trace와 Collector Counter는 `raw/profiles/ec_LOW_CUPTI_C25_CPC_q0.5_v0/`에 있다.
대응 Window CSV는 `analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_windows.csv`다.
Stage 및 Threshold Figure는 `figures/low_load_C25_*_v0.{png,pdf,csv}`다.
모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
