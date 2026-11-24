# Native MLLM EPD: Prefill 병목과 Decode Idle 판정

이 문서는 Native EPD의 Kernel·Queue 근거를 종합한다.
Main400은 보존했다. 사용자의 변경 지시에 따라 후속 실험은 동결된 Main100을 사용했다.
추가 Prefill의 실제 안전성은 별도의 Online Intervention으로 검사한다.
이 문서의 Physical Idle 수치를 Safe Headroom이나 Serving Speedup으로 해석하지 않는다.

## 핵심 판정

| 항목 | 판정 | 근거와 한계 |
|---|---|---|
| E/D Physical Kernel Idle | 관측됨 | QPS0.5의 C25 계측 Gate 통과 |
| E/D 동시 Physical Idle | 관측됨 | 같은 C25 GPU Timeline의 교집합 |
| P 포화와 E 비병목의 동시 조건 | PREFILL_HEAVY_NOT_CONFIRMED | 고부하 계측 Gate 실패와 독립 Capacity 근거 부족 |
| P Service Proximity≥85% | NOT_MEASURED | 독립적인 P Capacity Reference 미확보 |
| 실제 미래 Decode-safe Budget | NOT_MEASURED | 물리적 Gap과 현재 HTTP 상태만으로 예측하지 않음 |
| 모든 Idle의 원인별 분류 | NOT_MEASURED | Host·Transfer·작업 내부 Gap·Backpressure를 완전히 구분하지 못함 |
| SM Active·Tensor Core·HBM Bandwidth | NOT_MEASURED | Kernel Busy나 NVML을 해당 Counter로 대체하지 않음 |

미확정은 해당 현상의 부재를 증명한 판정이 아니다.
원하는 결과를 만들기 위해 Main100의 요청이나 Natural Output 계약을 바꾸지 않았다.

## 공통 실행 계약

Original BF16 Qwen2.5-VL-7B-Instruct를 사용했다. LLM은 Full28이다.
Model·Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
원래 Vision Encoder·Processor·Causal Attention·MRoPE를 유지했다. Video와 Pruning은 제외했다.
GPU4/5/6은 같은 H100 80GB의 E/P/D다.
P/D는 B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
실제 Online Encode와 vLLM Continuous Scheduler·NIXL KV Handoff를 사용했다.
Kernel Busy는 Kernel Interval의 합집합이다. Copy·Memset은 별도 기록했다.
Actual Scheduled Requests와 B/S 상한을 같은 Batch 값으로 취급하지 않았다.

Main100 Manifest SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`이다.
전체 Task Quality는 미확정이다. Runtime Timeline을 품질 PASS로 바꾸지 않는다.

## 적격 저부하 계측: C25

QPS0.5에서 같은 Main100의 Output5269 Tokens를 고정한 Replay를 실행했다.
실제 E Encoding은 매 요청에서 실행했다. 사전 계산 Embedding Replay가 아니다.
독립 초기화는 1회다. Natural EOS Serving과 결과를 합쳐 평균내지 않았다.
Clean/Profile/Clean의 다섯 Metric과 Collector 검사가 모두 고정한 5% Gate를 통과했다.

분모는 예정 Arrival p10–p90 구간의 157.172988초다.
초기·마지막 Drain을 제외한 분석 구간이다. Sustainable Steady State를 증명한 것은 아니다.

| Stage | Kernel Busy % | Kernel Idle % | Kernel+Copy Quiet % | Waiting Mean/관측 | Waiting Max | Scheduled Requests Max |
|---|---:|---:|---:|---:|---:|---:|
| E | 12.732604 | 87.267396 | 87.152670 | 0.509677 | 1 | 1 |
| P | 14.068244 | 85.931756 | 85.889734 | 0.041166 | 2 | 3 |
| D | 14.371586 | 85.628414 | 85.197001 | 0.019316 | 1 | 4 |

Waiting Mean은 Scheduler 관측별 평균이다. 정확한 시간 가중 Queue 평균이 아니다.
P Busy는 80% 기준보다 낮다. 이 저부하 조건을 Prefill-heavy로 부르지 않는다.

### 같은 Timeline의 E/D 동시 Idle

| 관측 | 비율 | 분모 |
|---|---:|---|
| E와 D의 동시 Kernel Idle | 74.237585% | 중앙 Arrival 구간 전체 |
| 위 Idle과 P Busy의 교집합 | 9.994642% | 중앙 Arrival 구간 전체 |
| 위 Idle과 P Busy의 교집합 | 71.043991% | P Kernel Busy 시간 |

P Busy와 P Queue의 지속은 다르다.
동시 Idle은 같은 요청을 세 GPU에서 병렬 실행할 수 있다는 증거가 아니다.
Stream·KV 준비·Deadline·비선점 Kernel·새 Arrival의 위험을 추가로 검사해야 한다.

### 물리적 Kernel Gap의 분포

| Stage | Count | Idle 총합 s | Mean ms | p50 ms | p90 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 72439 | 137.160774 | 1.893466 | 0.001566 | 0.006792 | 0.041539 | 0.243080 | 6769.806614 |
| D | 816866 | 134.584737 | 0.164757 | 0.000096 | 0.000288 | 0.001088 | 0.023008 | 6501.268365 |

Count 기준으로는 매우 짧은 Gap이 많다. 전체 Idle Duration은 드문 긴 Gap에 집중된다.
100ms 이상 Gap은 E Idle Time의 99.339073%, D Idle Time의 98.922669%다.
같은 Threshold의 Window Count 비율은 각각 0.091111%와 0.007100%다.
짧은 Gap과 GPU 간 선후관계에는 원시 Clock Anchor 오차를 고려해야 한다.
위 평균·Percentile을 안전한 Prefill Quantum의 길이로 바로 사용하지 않았다.

## 고부하 근거의 한계

C27은 QPS3에서 완료 요청 30–70번만 짧게 Capture했다.
ITL p99의 Profile 차이는 +28.1761%였다. TPOT p95 차이도 +5.2074%였다.
전체 5% Gate가 실패했다. 관측하지 않은 시간을 Idle로 채우지 않았다.
공통 관측 ROI에서 E/P/D Busy는 76.9893%/73.1256%/49.9453%였다.
이 값은 모두 DIAGNOSTIC이다. E 비병목이나 안전한 D 여유를 확정하지 않는다.

별도 C39는 원래 Parent Discovery의 MMLongBench-Doc37개를 모두 사용했다.
Main100을 이 요청으로 교체하지 않았다. Source Stratum은 성능 결과 전에 동결했다.
QPS1/2/3 모두 CPC Gate를 실패했다. 아래 수치는 DIAGNOSTIC이다.

| QPS | E Busy % | P Busy % | D Busy % | E Waiting Max | P Waiting Max | D Waiting Max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.769 | 58.444 | 82.397 | 2 | 3 | 1 |
| 2 | 84.568 | 75.902 | 52.196 | 6 | 6 | 1 |
| 3 | 84.231 | 67.180 | 30.566 | 13 | 4 | 1 |

긴 문서가 많아도 E 공급이 항상 충분하다고 가정할 수 없다.
E와 P의 Queue가 함께 관측됐다. 이 표만으로 병목의 인과 기여율을 정하지 않았다.
Clean Spread 실패만으로 Profiler가 모든 차이를 일으켰다고 확정하지 않는다.

## 재현 자료

- [적격 C25 저부하 분석](mllm_epd_low_load_kernel_idle_validation_v1.md)
- [C27 고부하 Short Capture와 Gate 실패](mllm_epd_high_load_short_capture_diagnostic_v1.md)
- [C39 문서 Stratum과 모든 CPC 결과](mllm_epd_secondary_document_prefill_profile_v1.md)
- [Natural C3 Baseline Motivation](mllm_epd_main100_baseline_motivation_v1.md)

Evidence Namespace는 `mllm_epd_prefill_heavy_token_split_20261124/`다.
C25의 원시 Trace는 `raw/profiles/ec_LOW_CUPTI_C25_CPC_q0.5_v0/`에 있다.
집계·Clock Anchor·Window CSV는 대응 `analysis/*idle_analysis.json`과 `analysis/*idle_windows.csv`에 있다.
Figure와 대응 CSV는 `figures/low_load_C25_*_v0`에 있다.
GPU UUID·명령·Source Archive SHA256은 각 `state/server_*.json`에 보존했다.
