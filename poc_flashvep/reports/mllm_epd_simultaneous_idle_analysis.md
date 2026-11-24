# Simultaneous E/D Idle: MLLM100 Diagnostic

실측 Kernel Timeline을 공통 Host Monotonic Clock으로 정렬했다.
각 Worker의 CUPTI 시작·종료 Anchor로 Affine 변환을 계산했다.
이 결과는 `TRACE_ANALYSIS`다. Observer Gate와 Model Quality는 아직 미확정이다.

| 구간 | 비율 | 분모 |
|---|---:|---|
| E와 D의 동시 Kernel Inactivity | 55.747% | 중앙 Arrival 구간 전체 시간 |
| P Busy 중 E와 D의 동시 Kernel Inactivity | 17.426% | 중앙 Arrival 구간 전체 시간 |
| P Busy 중 E와 D의 동시 Kernel Inactivity | 59.386% | 실제 P Kernel Busy 시간 |

P Busy와 P Queue는 다르다. 위 표는 P Queue가 지속되는 시간의 비율이 아니다.
Scheduler Queue는 Step별 관측값이다. HTTP Pending을 실제 GPU Queue로 대체하지 않았다.
동시 Inactivity는 같은 요청을 세 GPU에서 병렬화할 수 있다는 증거가 아니다.
미래 Encode·Decode Arrival과 비선점 Kernel의 비용은 이 표에 반영되지 않았다.
따라서 Actionable Slack과 Encode/Decode-safe Headroom은 `NOT_MEASURED`다.
E/D가 동시에 쉬는 현상은 진단 Timeline에서 관측했다. 정식 Headroom GO 판정은 보류한다.

Idle Window Count와 Duration 가중 Threshold Coverage는 JSON에 따로 보존했다.
0.1·1·5·10·20·50·100ms 기준을 모두 계산했다.
PNG/PDF의 제목은 실제 Admission 가능 구간이 아닌 Physical Kernel Gap이라고 표시한다.
Co-residency와 추가 Prefill Intervention을 실행한 상태가 아니다.
