# E/D Probe: 기준선 변동과 인과 해석의 한계

증거는 완료된 C40 N3의 TRACE_ANALYSIS다. Serving 경로와 SLO를 변경하지 않았다.
원래 8개 Probe 조건의 전체 보호 Gate 미통과 판정도 유지한다.

## 실제 No-probe 실패

추가 Probe가 없는 실제 Cohort24개 중 4개가 Absolute Decode 95% Gate를 실패했다.
Absolute Decode는 요청별 All-ITL≤34ms와 TPOT≤33ms/token의 AND다.
각 Start의 100개 요청을 분모로 사용했다. Start를 합쳐 실패를 숨기지 않았다.

| Mode | Control | B0 Decode % | B1 Decode % | B2 Decode % |
|---|---|---:|---:|---:|
| natural | NativeStrong | 98.0 | 93.0 | 96.0 |
| natural | NativeCommon | 97.0 | 97.0 | 97.0 |
| natural | clean_before | 93.0 | 98.0 | 99.0 |
| natural | clean_after | 97.0 | 97.0 | 98.0 |
| replay | NativeStrong | 95.0 | 99.0 | 94.0 |
| replay | NativeCommon | 99.0 | 95.0 | 95.0 |
| replay | clean_before | 98.0 | 97.0 | 100.0 |
| replay | clean_after | 95.0 | 94.0 | 96.0 |

따라서 Probe가 드문 Client Gap의 전부를 만들었다고 단정할 수 없다.
이 변동을 이유로 95% 기준을 낮추거나 실패 Start를 제외하지 않는다.
ITL p99가 34ms보다 낮다는 사실도 모든-ITL 요청 통과를 대신하지 않는다.

## 서로 다른 기준선 계약

NativeStrong은 별도의 새 초기화와 원래 Stock Compiler를 사용했다.
NativeCommon은 공통 Compiled 경로와 Probe용 Coordinator를 사용했다.
이 비교는 Compiler만 바꾼 One-factor Intervention이 아니다.
각 Block에서 Stock Control을 먼저 실행했다. Family 효과와 실행 기간 효과를 완전히 분리하지 못했다.
Co-resident No-probe는 Helper 적재 전후의 완벽한 One-factor Control도 아니다.
추가 Calibration Priming, Warm-up, Allocator 상태와 측정 시각이 함께 달라졌다.
Before/After 평균은 같은 초기화의 파생 대조군이다. 독립 Start를 두 배로 계산하지 않았다.

| Output | NativeCommon/NativeStrong ITL p99 Ratio | Paired 95% CI | Relative 5% NI |
|---|---:|---|---|
| natural | 1.001164 | [0.970391, 1.042496] | True |
| replay | 1.054585 | [0.960547, 1.139385] | False |

CI가 보호 기준을 넘었다는 사실과 유의한 악화를 입증했다는 주장은 다르다.
다른 Metric의 개선과 미통과 CI를 모두 원래 CSV에 보존했다.
E512 Natural의 E HTTP 개선이나 D512 Natural의 일부 Decode 보호만으로 전체 GO를 부여하지 않는다.
반대로 전체 Gate 미통과를 모든 GPU에 물리적 여유가 전혀 없다는 증거로 확대하지 않는다.

## 남은 인과 문제

Clean Cohort에는 Token별 Worker·ASGI 상세 Observer를 활성화하지 않았다.
드문 Gap을 GPU Kernel, Remote READ, Host Scheduler 또는 Client 전달에 고유하게 배분하지 못했다.
다른 Diagnostic 실행의 비율로 이 비용을 대체하지 않는다.
고부하 Prefill-heavy와 안전한 미래 Window도 미확인이다. Full Task Quality도 PENDING이다.
추가 Safe Admission 성공이나 실제 요청 Offload Speedup은 이 분석으로 주장하지 않는다.

## 재현 파일

analysis/probe_reference_controls_C40_v0.csv는 24개 실제 No-probe Cohort를 보존한다.
analysis/probe_reference_pairs_C40_v0.csv는 40개 Paired Control 비교를 보존한다.
대응 Raw·Source SHA·GPU 정보는 원래 C40 Config와 Lease 기록에 있다.
분석은 원래 SLO·Metric·출력·원시 응답을 덮어쓰지 않았다.
