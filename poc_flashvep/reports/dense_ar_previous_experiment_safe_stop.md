# 이전 Layer/Token 캠페인 안전 중단

판정: `STOPPED_BY_RESEARCH_REDIRECTION`.

사용자가 Native PD 재현으로 연구 방향을 변경했다. 이후 Layer/Token 성능 측정을 재개하지 않는다. 기존 원시파일과 실패 로그는 삭제하지 않았다.

## 소유권과 진행 상태 확인

중단 전에 UID, PID, Parent PID, Session ID, Command와 Child 목록을 확인했다. GPU UUID와 현재 Quiet-lane Lease도 확인했다. Config, Source ZIP 및 Source Hash Manifest의 SHA256을 저장했다.

확인 대상 Controller는 다음과 같다.

| PID | 역할 | 최초 확인 시 상태 |
|---:|---|---|
| 1375507 | Held-out Confirmation | Child 없음. Quiet-lane Lock 대기 |
| 1375508 | Main Sweep | 현재 GPU Job의 Dispatcher |
| 1412455 | Quantum Probe | Child 없음. 종료 정리용 Lock 대기 |
| 1441133 | Mixed Graph Probe | Child 없음. Quiet-lane Lock 대기 |

이 네 Controller만 먼저 정지시켜 신규 Dispatch를 차단했다. 현재 GPU Child는 즉시 정지하지 않았다.

## 현재 실행의 종료

최초 확인한 현재 Job은 `sweep_Layer_L16384_Last8_N36_O16_q8_B1_v1`이었다. Coordinator PID는 1449167이었다. GPU0/1을 사용했다.

이 Child는 현재 실행 종료까지 허용했다. Raw `status.json`은 `COMPLETE`였다. Child Exit Status는 0이었다. Config, Output, Worker Log와 Lifetime 기록을 보존했다. 이 실행을 새 Native 연구의 성능 결과로 사용하지 않는다.

## 종료 신호 처리 문제

기존 Dispatcher는 SIGTERM을 즉시 종료로 처리하지 않았다. 내부 Interrupt Flag로 바꿨다. 이미 끝난 Child를 수거한 뒤 Controller의 다음 Loop가 진행됐다.

그 결과 `sweep_Token_L16384_Last4_N36_O16_q8_B1_v1`이 중단 요청 이후 예기치 않게 시작됐다. 이를 정상적인 추가 실험으로 인정하지 않는다.

이 상황을 확인한 뒤 Controller와 Coordinator를 다시 정지시켰다. Source, Config, 소유권과 Session을 다시 확인했다. SIGTERM을 가로채는 CPU-only Controller는 SIGKILL로 종료했다. Coordinator에는 SIGINT로 정상 `finally` 정리를 요청했다.

Coordinator의 Raw 상태는 `FAILED / KeyboardInterrupt()`다. Worker 두 개는 기존 Cleanup Timeout 뒤 SIGTERM으로 종료됐다. Raw Lifetime Gate를 정상 통과로 표시하지 않는다. 별도 운영 상태를 `STOPPED_BY_RESEARCH_REDIRECTION`으로 기록했다.

해당 Token Job의 부분 Output과 모든 로그는 보존했다. 성능 집계에서는 전부 제외한다. 다른 사용자의 GPU Process에는 신호를 보내지 않았다.

## 재발 방지

이전 Namespace에 `STOPPED_BY_RESEARCH_REDIRECTION.json`을 추가했다. 이전 실행 진입점은 이 Marker가 있으면 새 Job을 거절한다.

새 Native-only Dispatcher는 Lock 획득 전후에 Stop Marker를 검사한다. 종료 신호가 들어오면 Marker를 저장한다. 현재 Job 정리 뒤에는 Controller Loop를 계속하지 않는다.

새 Dispatcher는 Native 또는 공식 Reference Module만 허용한다. Token Axis와 양수 Static Layer Cut은 거절한다.

## 자원 복원

이전 GPU Worker가 모두 종료된 뒤 승인된 Resource Guard로 GPU0–7 버닝을 복원했다. 복원 직후 8개 GPU 모두 100% Utilization을 관측했다.

Native Reference Timing 중에는 같은 Quiet-lane Guard가 승인된 Burner를 정지한다. 실험 사이와 종료 시에는 승인된 전체 8개 Burner를 복원한다.

## 증거

- [중단 Inventory와 원본 SHA256](dense_ar_native_pd_real_workload_reproduction_20261119/analysis/previous_campaign_stop_inventory.json)
- [최초 Process Snapshot](dense_ar_native_pd_real_workload_reproduction_20261119/state/research_redirection_pre_stop_snapshot.json)
- [현재 Job Drain](dense_ar_native_pd_real_workload_reproduction_20261119/state/research_redirection_current_job_drain.json)
- [신호 Interception 감사](dense_ar_native_pd_real_workload_reproduction_20261119/state/research_redirection_intercepted_signal_audit.json)
- [추가 Dispatch 차단과 정리](dense_ar_native_pd_real_workload_reproduction_20261119/state/research_redirection_intercepted_signal_stop.json)

완료된 과거 결과는 원래 상태를 유지한다. 중단한 캠페인은 `STOPPED_BY_RESEARCH_REDIRECTION`이다. 실행하지 않은 조건은 `NOT_RUN`이다. 미실행 점이나 독립 반복을 임의로 채우지 않는다.
