# BI0 Prefill-heavy 및 Safe Window 검증

상태: P Prefill 병목은 새 실행에서 확인했다. D Host Quiescence와 미래 Decode-safe 실행 예산은 구분한다. Active-decode Dynamic Scheduling은 최종적으로 `HOLD`다.

## 깨끗한 P Kernel Activity

ShortAnswer200-v2, QPS2.35, Arrival Seed2026112051을 사용했다. 일반 Continuous Native1P1D를 실행했다. 모든 Engine은 B8192/S1024, KV42GiB, Block128, BI0와 공식 UCX 초기화를 사용했다. Source/Destination은 GPU0/1이다.

처음 Two-role CUPTI 계측은5% Observer Gate를 실패했다. P-only 개입은 D Collector만 제거했다. 기준은 바꾸지 않았다. Clean2개와 Profile1개의 독립 Start를 비교했다. Whole TTFT Mean, Whole Mean TBT, TPOT p95와 두 ROI 지표의 변화는 모두5% 이내였다. 최대 절대 변화는0.6901%였다. Timestamp Bracket 최대값은0.024961ms였다. 이는 Point Gate다. Observer 영향의95% 상한을 검증한 것은 아니다.

| 항목 | 같은86.1447초 Host Window의 값 | 근거 수준 |
|---|---:|---|
| P Kernel Busy | 99.6264% | `MEASURED_P_KERNEL_ACTIVITY` |
| P Waiting 평균 / p95 / 최대 | 28.3903 / 41 / 43개 | `TRACE_ANALYSIS` |
| D Decode-ready Waiting 평균 / 최대 | 0 / 0개 | `TRACE_ANALYSIS` |
| D Remote-KV Waiting 평균 / 최대 | 0.0699 / 5개 | `TRACE_ANALYSIS` |
| D Active Decode 평균 / 최대 | 0.4012 / 5개 | `TRACE_ANALYSIS` |
| D Clean Kernel Busy | `NOT_MEASURED_IN_P_ONLY_INTERVENTION` | 해당 없음 |

P Queue만으로 Compute 포화를 추정하지 않았다. 같은 구간의 Kernel Union이 높은 Busy를 확인했다. 단, 이 고부하 조건은 Finite Cohort다. 장기 Sustainable Capacity를 입증하지 않는다.

## D의 관측된 비활성 상태

D Scheduler와 Engine이 비었고 Current-stream CUDA Event가 완료된 구간을 기록했다. 같은 Window에서121개 구간을 관측했다. 구간 합은52.6953초였다. Window의61.1707%다. 길이 p50/p95/p99는317.929/1290.027/1566.878ms였다.

이 값은 D의 모든 Stream에 대한 Kernel Idle 비율이 아니다. GPU Copy나 Background 작업을 모두 배제한 물리 Timeline도 아니다. 끝점은 다음 Control Message의 실제 수신 시각이다. 미래 Window의 예측 끝점이 아니다.

이 Window에서 Full Native P-ready는171개였다.137개가 관측된 D-empty 구간에 들어왔다. 가상의 Layer Cut은 더 일찍 Ready가 될 수 있다. Action을 바꾸면 이후 Queue도 바뀐다. 따라서137/171을 실제 Layer/Token Safe-fit 비율로 사용하지 않는다.

## 실행 가능한 안전 계약

별도 Mechanism Control은 P1/D-suffix1/D-decode1을 사용한다. D Scheduler Request가0이어야 한다. Unfinished Engine Work, Pending NIXL Handle과 Active Read도 없어야 한다. Device-wide Synchronization이 완료돼야 한다. 이후 Coordinator가 다른 D 요청을 넣지 않는 동안에만 새 Suffix를 수락한다.

이 계약은 이미 실행 중인 Decode를 Suffix로 중단하지 않는다. 이는 Quiescent-only Admission이다. Active Decode 사이의 Gap 삽입이나 선점은 구현하지 않았다. Window-aware Size 또는 Axis Controller가 독립적인 이득을 냈다는 뜻도 아니다.

일반 Continuous Native/Agg는 이 Batch1 Control로 대체하지 않는다. 두 실행 계약의 성능은 별도 표로 보고한다. Dynamic 후보는 실제 Tail Cost, 준비 상태, Queue 변화와 Decode CI 검사를 통과해야 한다.

## 현재 판정

P-prefill-heavy 조건은 이 고부하 Family에서 확인했다. D-ready Queue가 쌓이지 않는다는 사실도 확인했다. D의 관측된 Quiescence는 추가 검증할 기회를 제공한다. 그러나 이를 모두 미래 Decode-safe Budget으로 합산하지 않는다.

Physical Gap Distribution, 준비된 Partial State와의 정렬, 새 Decode 도착의 불확실성은 별도 Gate다. 그 Gate가 충분하지 않으면 위험한 Active-decode Offloading을 실행하지 않는다.

## 근거

- [P-only Observer 검사](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_P_only_ContinuousNative_observer_v2.json)
- [같은 Window의 P/D 상태](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_prefill_heavy_opportunity_v2.json)
- [D Collector 제거의 사전 설계](dense_ar_bi0_shortanswer200_20261120/configs/frozen_P_only_activity_intervention_v2.json)
- [실패한 Two-role 계측](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_quiescent_window_observer_gate.json)

## 공식 초기화 Full200 Profile의 반례

공식 UCX 초기화와 기본 KV Block16에서 Two-role Profile을 추가 검사했다. QPS2.35, Seed2026112051과 Natural200을 유지했다. Clean2개와 Profile1개를 비교했다. Mean TBT는10.2432%, TPOT p95는46.0327% 증가했다. ROI Mean TBT도11.5910% 증가했다. 동결된5% Observer Gate는 `FAIL`이다.

이 Profile의 D Kernel Busy는27.6648%였다. 이 수치는 `DIAGNOSTIC`이다. 깨끗한 D Busy나 Safe-window 성공 근거로 사용하지 않는다. Profile의 지연 증가를 새 Method의 Decode 비용으로 계산하지도 않는다. Raw Kernel Timeline과 실패한 Gate는 [공식 초기화 Observer 결과](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_official_init_online_observer_v3.json)에 보존했다.

P-only 검사에서 확인한 높은 P Busy와 D Host-empty 상태는 여전히 별도의 근거다. D 전체 Kernel Timeline의 Primary 정량 검증은 미완료다. 이 제한을 NVML Utilization이나 Host Queue 값으로 대신하지 않는다.
## C0 Actual Batch1 경로의 추가 계측

빠른 C0/Async-on Native의 별도 P-only Clean/Profile/Clean도 검사했다. QPS2.35, Seed2026112051과 KV Block128을 고정했다. Whole Mean TTFT는5.2908% 증가했다. ROI Mean TTFT는5.2682% 증가했다. Mean TBT는1.6889%, TPOT p95는0.0436% 증가했다.

사전 Observer 기준은 모든 검사 항목의 절대 변화≤5%다. 따라서 이 계측은 `DIAGNOSTIC_OBSERVER_GATE_FAILED`다.5.29%를5%로 반올림해 통과시키지 않는다. 같은 조건을 반복해 우연한 PASS를 찾지도 않는다.

이 실패가 기존 C3 Continuous Native의 P-only PASS를 취소하지는 않는다. 두 실행 계약을 분리한다. C0 Actual Batch1의 Clean Kernel Busy는 이번 추가 Profile로 인증하지 못했다. 해당 Profile의 Kernel 수치와 Timeline은 Primary Serving 설명의 GPU 비중으로 사용하지 않는다.
