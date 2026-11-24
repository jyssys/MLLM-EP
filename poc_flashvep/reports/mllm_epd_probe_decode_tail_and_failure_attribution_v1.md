# E/D Probe: ITL p99와 요청별 Decode 실패의 차이

증거는 실제 C40 N3의 TRACE_ANALYSIS다. 새 GPU 실행이나 Profiler는 추가하지 않았다.
원래 8개 Natural/Replay 보호 Gate 판정은 변경하지 않았다. 모든 조건은 전체 Gate 미통과다.
측정한 Probe는 Main 요청을 완료하지 않는다. 별도의 Throughput 개선은 아니다.

## 다른 분모와 통계량

ITL p99는 해당 Start의 모든 Client Token Interval을 모은 99th Percentile이다.
All-ITL Pass는 한 요청의 모든 Interval이 34ms 이하여야 한다.
Absolute Decode는 All-ITL과 TPOT≤33ms/token의 요청별 AND다. 각 Start에서 95% 이상이어야 한다.
따라서 일부 요청의 드문 큰 Gap은 전체 ITL p99에 나타나지 않아도 해당 요청을 실패시킨다.
실제 Probe Cohort24개 중 4개는 ITL p99≤34ms였지만 요청별 Decode 95% Gate를 실패했다.

| Start | Mode | Probe | ITL p99 ms | Client Max ITL ms | All-ITL AND TPOT Pass % |
|---:|---|---|---:|---:|---:|
| 0 | natural | D128 | 15.912 | 183.361 | 94.0 |
| 1 | natural | E128 | 15.074 | 133.336 | 92.0 |
| 1 | replay | D128 | 16.439 | 49.904 | 93.0 |
| 2 | natural | D128 | 16.546 | 133.477 | 92.0 |

전체 보호 Gate 실패에는 Absolute 실패뿐 아니라 Relative CI 상한 실패도 포함된다.
Point Ratio가 1.05 이하여도 CI 상한이 1.05를 넘으면 비열등성이 입증되지 않는다.
그 결과를 모든 Metric의 유의한 악화라고 바꾸지 않는다. N3 CI 해상도도 제한적이다.
E512 Natural의 일부 Point·CI 개선을 선택해 다른 실패 Gate를 삭제하지 않았다.

## 확정하지 못한 원인

실제 GPU Kernel 경쟁, Remote READ, Host Scheduling, Client Delivery를 Token별로 분해하지 못했다.
Clean C40에서는 해당 상세 계측을 활성화하지 않았다. 다른 Diagnostic 실행의 Operator 비율로 대체하지 않는다.
E128·E512·D128·D512 순서는 고정했다. 기간 효과와 다른 Batch·Natural Output 차이가 남아 있다.
따라서 이 표는 Metric 실패 원인을 설명한다. 드문 Gap의 단일 Runtime 또는 Hardware 원인을 확정하지 않는다.
Counterfactual Gate 제거, GPU Cut 변경 또는 미래 안전 Window의 효과로 해석하지 않는다.

## 원시 자료

analysis/probe_failure_tail_C40_v0.csv는 48개 실제 Control/Probe Cohort를 모두 포함한다.
각 요청·Token의 8개 실패 조합은 원래 interference_PROBE_C40R4_failure_categories_v0.csv에 있다.
대응 Figure와 CSV는 figures/probe_C40_actual_decode_attainment_v0 및 probe_C40_client_max_ITL_v0다.
