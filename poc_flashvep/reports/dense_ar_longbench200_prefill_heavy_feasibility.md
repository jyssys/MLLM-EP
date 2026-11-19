# LongBench v1 Mixed200: Prefill-heavy 검증

판정: `P_BOTTLENECK_CONFIRMED / DECODE_SAFE_HEADROOM_NOT_ESTABLISHED / OFFLOADING_HOLD`.

높은 부하에서 P Compute 병목은 확인했다. 그러나 D가 Decode SLO를 보호하며 추가 Prefill을 실행할 수 있다는 근거는 확보하지 못했다. Layer/Token 실험은 재개하지 않는다.

## 공통 조건과 관측 범위

Qwen3-8B Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다. Original BF16, Full36 Layers, Greedy Natural EOS를 사용했다. Thinking과 Prefix Cache는 비활성화했다. Output Cap은2048이다.

GPU0/P는 `GPU-f217c8a0-1142-20f4-d84b-af29f3a47a0d`다. GPU1/D는 `GPU-a77f3471-67d4-20b0-9fab-e502d4de5adb`다. vLLM0.26.0+cu129, PyTorch2.11.0+cu129, NIXL1.4.1을 사용했다. 모든 Engine은 B8192/S1024와 KV42GiB를 명시했다. FLASH_ATTN/HND와 공식 Batch-invariant 경로를 사용했다.

Mixed200은 LongBench v1의 HotpotQA, MuSiQue, NarrativeQA, GovReport, QMSum을 각각40개 포함한다. [Dataset 계약](dense_ar_longbench200_dataset_contract.md)에 ID, Revision, Sampling Seed, Truncation Manifest와 SHA256을 보존했다.

## 같은 실행 구간의 P/D 관측

각 QPS는 Profile1회와 Clean Control2회로 구성한다. Profiled Start의 처음90초를 수집했다. 두 GPU의 공통 구간에서 양끝2초를 제외했다. Seed는2026112041이다. Kernel Busy는 모든 Stream의 Kernel Interval Union을 공통 구간으로 나눈 값이다.

| 같은 구간의 Metric | QPS1.0 | QPS2.35 | 증거 수준 |
|---|---:|---:|---|
| 공통 관측 구간 | 85.9991s | 86.0165s | `TRACE_ANALYSIS` |
| P Kernel Busy | 69.1322% | 99.3415% | `MEASURED` |
| D Kernel Busy | 89.1763% | 85.2237% | `MEASURED` |
| P Waiting 평균 / p95 / 최대 | 1.373 / 6 / 7 | 36.772 / 54 / 57 | `TRACE_ANALYSIS` |
| D Remote-KV Waiting 평균 | 0.185 | 0.264 | `TRACE_ANALYSIS` |
| D Decode-ready Waiting 평균 | 0 | 0 | `TRACE_ANALYSIS` |
| D Active Decode 평균 / p95 / 최대 | 3.510 / 7 / 8 | 5.978 / 10 / 12 | `TRACE_ANALYSIS` |
| Observer 비교의 최대 절대 변화 | 1.2814% | 1.5566% | Point-estimate Screen |
| Dropped / Invalid Activity Record | 0 / 0 | 0 / 0 | 계측 정합성 |

Decode-ready는 D의 마지막 Prompt Query도 포함한다. Remote-KV Waiting은 별도로 계산했다. Queue 값은 시간 가중 Host Scheduler Snapshot이다. Kernel Queue나 GPU Idle의 직접 측정값이 아니다.

두 Profile은 사전5% Observer Screen을 통과했다. 이 검사는 전체 실행과 ROI의 TTFT/TBT 및 전체 TPOT를 비교한다. Kernel Busy는 QPS당 Profile1회의 값이다. Start-level CI는 없다. 약86초의 관측을600초 이상의 안정 상태로 해석하지 않는다.

QPS1에서는 D가 P보다 더 바빴다. 해당 구간을 P-prefill-heavy라고 부르지 않는다. QPS2.35에서는 P가99.34% Busy였고 P Waiting이 평균36.77개였다. 이 두 관측은 같은 구간의 P Compute 병목을 지지한다.

## Native TTFT의 Host 경계 분해

Main 요청별 Host Timeline도 분해했다. 아래 첫4개 구간의 합은 요청의 TTFT와 일치했다. 최대 합산 오차는0ms였다. 아래 값은 평균이며, 앞의 TTFT p95와 다른 통계량이다.

| Offered QPS | Starts | Arrival→P 첫 Prepare | P 첫 Prepare→Ready | Ready→D Prepare | D Prepare→첫 Delivery | Mean TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 1 | 84.42ms | 751.26ms | 209.01ms | 61.04ms | 1105.73ms |
| 1.0 | 3 | 559.15ms | 1022.02ms | 234.43ms | 67.07ms | 1882.68ms |
| 1.65 | 3 | 11481.26ms | 1301.69ms | 238.72ms | 69.31ms | 13090.97ms |
| 2.35 | 1 | 25299.17ms | 1313.26ms | 229.34ms | 63.14ms | 26904.91ms |

고부하의 증가 대부분은 P의 첫 실행 전 대기에 있었다. QPS2.35의 Ready→NIXL Receive Notification은 평균224.80ms였다. 해당 Notification→D Prepare는4.54ms였다. Ready→D 전체229.34ms를 D Admission만의 대기나 순수 DMA라고 부르지 않는다.

P 첫 Prepare→Ready에는 Chunk Scheduling과 Host 경계가 포함된다. GPU-exclusive Prefill Time이 아니다. 구간별 p95는 더하지 않았다. [전체 Start별 경계와 CI](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_host_critical_path_start_statistics.json)를 보존했다.

## D Kernel Inactivity는 안전 Window가 아니다

QPS2.35의 D Kernel Inactivity는14.7763%였다. DMA Activity와 SDK 내부 상태는 Kernel-only Capture에 포함하지 않았다. 따라서 이 값을 Physical Idle이나 Offloading Budget으로 사용할 수 없다.

추가 Trace 분석은20ms 이상의 D Kernel Gap을 분리했다. 20ms는 분석용 구분점이다. 안전 실행시간을 뜻하지 않는다.

| D Kernel Gap 분석 | QPS1.0 | QPS2.35 |
|---|---:|---:|
| 20ms 이상 Gap 수 | 66 | 109 |
| 해당 Gap 총시간 | 8.7173s | 12.1208s |
| `submit_transfer_host`와 겹친 시간 | 8.4489s | 11.7842s |
| 해당 Gap 시간 중 겹친 비율 | 96.9208% | 97.2231% |
| 최대 Gap | 368.894ms | 411.622ms |

같은 Host Clock으로 정렬한 API 구간과 Kernel Gap이 겹쳤다. 이것은 `TRACE_ANALYSIS`다. 별도 원인 제거 실험은 아니다. API 구간은 순수 DMA나 CPU-exclusive Time이 아니다. 겹쳤다는 이유만으로 SDK 대기가 Gap의 유일한 원인이라고 확정하지 않는다.

NIXL1.4.1의 UCX Thread-pool 경로에는 `future.wait()`가 있다. 호출 Thread CPU Time이 작고 API Wall Time이 길다는 관측과 일치한다. 그러나 내부 대기, 실제 Transport와 GPU 의존성의 완전한 분해는 `NOT_IDENTIFIED`다. [SDK Source](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_backend.cpp)를 확인했다.

## Decode SLO와 요청 길이의 영향

Frozen SLO는 TTFT≤5000ms, 모든 ITL≤34ms, TPOT≤33ms다. QPS1.65의 Main3회에서 Native ITL p99는149.02ms였다. TPOT p95는39.07ms였다. Joint Attainment는3.33%였다. 평균 TBT32.11ms만으로 Decode SLO 통과를 판단할 수 없다.

QA120개는 전체 Input Tokens의70.17%를 차지했다. Output Tokens 비중은3.60%였다. Summarization80개는 Output Tokens의96.40%를 차지했다. 이는 Token 수의 `STRUCTURAL_ACCOUNTING`이다. GPU Compute 비중이나 같은 비율의 Kernel Time을 뜻하지 않는다.

QA와 Summarization이 섞이면서 긴 Decode의 Residency가 생겼다. QPS2.35의 D Active Decode 평균5.98개도 함께 관측했다. Decode-ready Waiting이0이어도 D는 계속 계산할 수 있다.

QPS1.65의 Task별 결과도 분리했다. 각 값은 독립 Start3회의 평균이다. Task마다 Start당40개 요청이 있다.

| Task | Native TTFT p95 s | Agg TTFT p95 s | Native ITL p99 ms | Agg ITL p99 ms | Native Joint % | Agg Joint % |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 20.190 | 2.056 | 122.21 | 400.78 | 6.67 | 50.00 |
| MuSiQue | 20.679 | 2.171 | 106.29 | 412.78 | 2.50 | 35.83 |
| NarrativeQA | 20.041 | 2.856 | 118.90 | 453.63 | 7.50 | 25.00 |
| GovReport | 18.712 | 2.306 | 154.12 | 410.06 | 0.00 | 0.00 |
| QMSum | 18.796 | 1.988 | 140.42 | 423.67 | 0.00 | 5.83 |

Native는 모든 Task의 ITL p99가 더 낮았다. 그러나 긴 TTFT 때문에 Joint SLO에서 우세하지 않았다. 두 방법 모두 GovReport의 Joint SLO는0%였다. 위 표만으로 요청 수와 Token 수의 효과를 합치지 않는다. [Task별 Start-level CI와 전체 QPS](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Main_task_start_level_statistics.json)를 별도로 보존했다.

## 이전 고정 길이 결과와의 차이

기존8192/16 Batch1/QPS4는 P Busy82.1181%, D Busy약90.69%였다. Ready→D Submit 평균은1250.9623ms였다. 기존16384/16 Batch1/QPS4는 P Busy96.9523%, D Busy58.2035%였다. 이 값들은 이전 Runtime의 별도 구간에서 측정했다. 이번 Natural Mixed200과 같은 실행 계약이 아니다.

더 오래된8000/16 P/D Slots4/16 결과도 원시 기록으로 확인했다. Exact Eager Family의 Qualified Start1회에서 P Busy96.3775%였다. D의 관측 Kernel Inactivity는14.4193초 중 최소61.5572%였다. D Capture는 빈 Engine Epoch에 한정됐다. 전체 D Busy는 `NOT_MEASURED`다.

이전 Family는 Eager, Sync Scheduler와 Whole-cohort 계약을 사용했다. 새 Main은 Compile3, Async Scheduler, Standard Continuous Scheduling을 사용한다. Dataset, Output, Batch, Admission과 관측 방식도 다르다. 따라서 차이를 Input 길이 하나의 효과로 설명하지 않는다.

## 최종 Gate

| Gate | 판정 | 근거와 제한 |
|---|---|---|
| 실제 Long-context Dataset | PASS | 200개, Input 평균16674.06 Tokens |
| 높은 QPS의 P Compute 병목 | PASS | 같은 구간의99.34% Kernel Busy와 P Queue |
| D Decode-ready 누적 없음 | 관측됨 | D Active Decode와 Remote-KV Waiting은 별도 존재 |
| D의 큰 Compute 여유 | NOT_ESTABLISHED | Kernel Busy85.22%, Gap의 다수가 Transfer Host 구간과 겹침 |
| Frozen Decode SLO 보호 | FAIL_ON_TESTED_PRIMARY_GRID | Native Tail이34/33ms 기준 초과 |
| 실행 가능한 Decode-safe Window | NOT_MEASURED | Deadline과 비선점 실행을 검증한 Intervention 없음 |
| Layer/Token 또는 Dynamic 효과 | NOT_RUN | 이전 Child의 Safe-stop 이후 재개하지 않음 |
| 향후 Offloading 재개 | HOLD | Concurrent Import Tail과 안전 Window를 먼저 검증해야 함 |

이 판정은 해당 모델, Numerical 경로, Dataset과 Runtime에 한정한다. 모든 PD 환경에서 안전 Headroom이 없다는 결론은 아니다.

## 재현 자료

- [QPS1 Activity 및 Observer Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_online_kernel_activity_q1_v1.json)
- [QPS2.35 Activity 및 Observer Gate](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/Native_online_kernel_activity_q2.35_v1.json)
- [관측 구간 설명](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/activity_window_scope_clarification.json)
- [이전 Local Capacity 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/historical_Native_local_capacity_audit.json)
- [이전 Slots4/16 Activity 감사](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/historical_L8000_slots4_16_activity_audit.json)
- [P/D Kernel Busy Figure](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_native_kernel_busy.png)
- [P/D Scheduler Queue Figure](dense_ar_native_pd_real_workload_reproduction_20261119/figures/longbench200_native_scheduler_queues.png)
