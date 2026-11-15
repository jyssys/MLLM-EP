# Direct Deadline-aware Layer Admission

이 문서는 이번 캠페인의 Admission 계약과 검증 한계를 기록한다.
독립 확인 결과와 최종 판정은 [최종 요약](dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md)에 모은다.

## 실행 계약

모델은 Qwen3-8B의 원본 BF16이다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
후보는 Last0/2/4/8/12다.
모든 요청은 전체 36 Layer의 Prefill과 정상 Full36 Decode를 수행한다.

의사결정은 P Prefix를 제출하기 전에 발생한다.
D는 기존 Decode Cohort가 실제로 종료된 이후에 Suffix를 실행한다.
예측상 Cohort가 종료됐다는 이유만으로 실행을 시작하지 않는다.
GPU Kernel의 선점을 가정하지 않는다.
이 Cohort Guard는 Dynamic만의 기능이 아니다.
Native와 Static의 같은 실행 경로에도 적용한다.

실제 P Prefix, D Suffix, Hidden/Residual, 유효 KV와 ACK는 별도로 검증한다.
Source Buffer는 ACK 전에 해제하지 않는다.
Commit 이전에는 Native를 선택할 수 있다.
Commit 이후에는 계산한 Prefix를 버리거나 같은 Layer를 재실행하지 않는다.

## 직접 Deadline Predicate

[deadline.py](../dense_ar_slo_layer_pd/deadline.py)는 다음 항목을 후보별로 검사한다.

- 후보의 P Cut Ready와 D Import Ready.
- 현재 Cohort의 종료 상한과 이미 알려진 Pending Work.
- 실제로 영향을 받는 요청의 ITL 또는 TTFT Deadline.
- Suffix, Commit, 중간 Cohort Decode와 다음 알려진 Ingress.
- KV/Hidden Credit, 상태의 최신성, 체류 상한과 Completion Benefit.

Suffix가 시작하기 전에 종료되는 Cohort의 과거 Token Deadline은 해당 Suffix의 Deadline이 아니다.
반대로 영향을 받는 Decode의 완료 상한이 Deadline을 넘으면 해당 후보를 거절한다.
Last12가 거절돼도 Last8/4/2를 각각 평가한다.
작은 Cut도 안전성과 이득을 충족해야 한다.
선택 빈도를 만들기 위해 작은 Cut을 강제로 수락하지 않는다.

미래 Arrival은 입력하지 않는다.
알려진 P Pending Request의 다음 Frontier는 모델 추정이다.
실제로 D-ready인 요청과 같은 상태로 집계하지 않는다.
Host 상태로 계산한 Budget을 물리적 GPU Idle로 표현하지 않는다.

## 목적함수와 불확실성

온라인 Cost는 실측 Lookup과 Frontier 모델을 사용한다.
Roofline은 구조적 하한을 설명한다.
Roofline 값을 실제 Completion Time으로 사용하지 않는다.

별도 옵션은 기존 Queue Waiting을 새 GPU Service Demand와 구분한다.
Waiting은 Deadline과 Completion 검사에는 남는다.
Queue Benefit을 허용하는 모드와 요청 자체의 이득만 허용하는 모드는 다르다.
주요 목적함수는 Pipeline Rate의 경험적 Surrogate다.
정확한 미래 SLO-goodput을 예측한다고 주장하지 않는다.

Discovery에서 기존 110 ms 상한에 대한 최대 양의 Group Residual은 423.952849 ms였다.
두 실행의 관측치로 추가 Margin을 440 ms로 계산했다.
총 Margin은 550 ms다.
계산 규칙은 최대 Residual에 10 ms를 더한 뒤 10 ms 단위로 올림하는 방식이다.
근거는 [discovery_residual_guard_v1.json](dense_ar_slo_throughput_dynamic_layer_pd_20261114/configs/discovery_residual_guard_v1.json)이다.
한 Start 안의 Group들은 독립 표본이 아니다.
이 Margin은 WCET나 미래 Deadline의 보장이 아니다.

새 Canonical/Async Runtime에서는 Decode Table도 다시 계산했다.
각 Batch의 실제 15개 cap2 Group에서 Worker Interval을 관측했다.
그다음 새 Predictor로 N128 Calibration Start 3회를 실행했다.
Seed는 2026111464/2026111465/2026111466이다.
최대 명목 Group Completion 과소예측은 125.296004 ms였다.
사전 규칙에 따라 새 Margin은 150 ms다.
근거는 [canonical_completion_residual_v1.json](dense_ar_slo_throughput_dynamic_layer_pd_20261114/configs/canonical_completion_residual_v1.json)이다.
이 파일은 Held-out 성능 평가 전에 생성했다.

150 ms가 모든 미래 Request를 포함한다는 보장은 없다.
특히 Group 최대 오차와 개별 Request 오차는 구분해야 한다.
단독 Calibration의 종속 Group을 독립 표본으로 사용하지 않는다.
장기 실행과 Held-out에서 Upper Coverage를 별도로 검사한다.
옛 550 ms와의 차이는 Runtime과 모델을 함께 변경한 결과다.
이를 KV 정렬 하나의 인과 효과로 표현하지 않는다.

## 작은 Cut의 전송량 — STRUCTURAL_ACCOUNTING

8000-token Prompt의 KV는 Layer당 31.25 MiB다.
계산은 `8000 × 2(K/V) × 8(KV Heads) × 128 × 2(BF16 Bytes)`다.
Partial Cut은 Hidden과 Residual을 합해 125 MiB를 추가로 전달한다.
새 Token의 Decode KV는 이 Prompt Handoff 표에 포함하지 않는다.
Control Message와 Registration Metadata도 이 Data Payload 표에서 제외한다.

| Cut | Prefix KV | Hidden + Residual | 합계 | Native 대비 |
|---|---:|---:|---:|---:|
| Last0 | 1125 MiB | 0 MiB | 1125 MiB | 1.000000× |
| Last2 | 1062.5 MiB | 125 MiB | 1187.5 MiB | 1.055556× |
| Last4 | 1000 MiB | 125 MiB | 1125 MiB | 1.000000× |
| Last8 | 875 MiB | 125 MiB | 1000 MiB | 0.888889× |
| Last12 | 750 MiB | 125 MiB | 875 MiB | 0.777778× |

Last2는 이 구조에서 Native보다 적은 Bytes를 전달하지 않는다.
Last4는 같은 Bytes에 별도 Hidden Protocol이 필요하다.
따라서 작은 Cut이 항상 Handoff에 유리한 것은 아니다.
이 표는 Byte Accounting이다.
전송시간, Queue Benefit 또는 Serving Speedup은 아니다.
실제 NIXL Bytes와의 대조는 Geometry Audit에 별도로 남긴다.

## 지난 TTFT Deadline과 Queue Recovery

Strict 정책의 QPS 4.8 장기 실행에서는 지난 TTFT Deadline도 계속 거절 조건으로 남았다.
초반에는 Migration이 발생했다.
360초 이후에는 해당 구간의 2,050개 Request Proposal이 모두 Last0이었다.
이는 같은 실행의 Host Decision Trace 분석이다.
GPU Slack 부족의 증거는 아니다.

`QUEUE_RECOVERY`는 결정 시점에 이미 지난 TTFT Deadline만 구분한다.
그 요청은 기존 SLO 평가에서 계속 실패한다.
ITL과 아직 지나지 않은 TTFT Deadline은 그대로 검사한다.
요청을 삭제하거나 다시 계산하지 않는다.
N64 정합성 시험은 이 분기를 실제로 2회 통과했다.
이 시험의 동시 Arrival은 분기 검증용이다.
Serving 성능 근거가 아니다.

Streaming Recovery의 QPS 4.8 실행도 600초 안정성 기준에는 실패했다.
실제 도착률은 4.805 requests/s였다.
완료율은 4.453333 requests/s였다.
Raw Output은 71.253333 tokens/s였다.
Outstanding Request는 29개에서 240개로 증가했다.
Decode-priority의 Arrival-cohort Joint Attainment는 30.107527%였다.
이 수정만으로 충분한 Capacity를 확보하지 못했다.

근거는 [해당 Sustainable Window 분석](dense_ar_slo_throughput_dynamic_layer_pd_20261114/analysis/recovery_wait0_DIRECT_O16_q4.8_s2026111458_W600_v1_sustainable_v2.json)이다.
고정 상태를 재사용한 Offline 대안 수와 실제 완료 Migration은 구분한다.

## 증거 범위

CPU Predicate 시험은 Ready Decode, 안전한 Quantum, 종료 Cohort와 미래 Ready Work를 구분한다.
추가 시험은 지난 TTFT의 예외가 ITL이나 미래 TTFT로 확장되지 않는지 검사한다.
전체 Output/KV/Hidden 정합성은 별도 GPU Gate로 검증한다.

실제 Kernel Idle, 개별 GEMM/Attention 실행시간과 HBM Traffic은 이 문서에서 `NOT_MEASURED`다.
실제 선택 분포와 겹치는 Gate 수는 최종 Evidence의 Candidate Audit에 남긴다.
같은 후보가 여러 Gate를 위반할 수 있다.
Gate 수를 Request 수처럼 더하지 않는다.

## 최종 Online 선택과 검증

QPS4.4의 Measurement Cohort는 8073 requests다.
실제 완료 Cut은 Last0 2003건, Last2 0건, Last4 7건, Last8 137건과 Last12 5926건이었다.
작은 Cut의 실제 실행은 확인했다.
그러나 Last12 대신 작은 Cut을 고른 관측 사유는 `NO_AGGREGATE_COMPLETION_BENEFIT`였다.
이 결과는 Decode Deadline만으로 Last12가 위험했던 경우의 추가 Serving 이득을 입증하지 않는다.

QPS4.8의 같은 Rate Native 비교에서는 ITL p99와 TPOT p95의 5% CI Gate를 통과했다.
Ratio CI 상한은 각각 1.018860과 1.039966이었다.
반면 Dynamic의 Queue·Joint SLO 기준은 3회 모두 실패했다.
Maximum ITL도 최대 1151.496112 ms였고, 큰 Host Delivery 성분이 남았다.
Quantile Gate 통과를 모든 Decode Deadline의 보장으로 해석하지 않는다.

새 Margin의 Group Upper Coverage는 확인 Run별로 99.2~100%였다.
그럼에도 최악 Group Upper Underprediction은 775.965188 ms였다.
큰 절대오차와 실제 SLO는 별도로 보고한다.
Seed별 원값과 작은 Cut의 실제 사유는 최종 Prediction/Candidate Audit에 있다.
