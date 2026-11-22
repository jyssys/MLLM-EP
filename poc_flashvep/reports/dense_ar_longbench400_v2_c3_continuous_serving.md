# Strong C3 Continuous Serving

상태: ACTIVE_SNAPSHOT.
생성 시각: 2026-11-22T18:39:15.240077+00:00.
경과 연구 시간: 11.540시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 실행 계약

증거 수준은 `MEASURED_ONLINE`이다. Main은 Natural EOS와 Output Cap2048을 사용한다.
C3는 실제 Continuous Batching을 사용한다. 실제 Batch는 각 실행의 CSV에 기록한다.
B8192/S1024는 Scheduler 상한이다. 이 값은 실제 Batch가 아니다.
Stock Token은 원래 Compiled Full-model 경로를 사용한다. C0 Mode0 Token으로 대체하지 않았다.
Hybrid는 두 축을 모두 분할한다. Layer-only 또는 Token-only를 Hybrid로 이름만 변경하지 않았다.
고정 후보는 Discovery에서 선택했다. Held-out 성능을 본 뒤 후보를 변경하지 않는다.
Main400은 Discovery와 Held-out 요청을 모두 포함한다. Held-out192 결과는 별도 CI로 표시한다.
Natural EOS의 Output Token 총수가 5%를 초과해 다르면 Work-unmatched로 표시한다.
Fixed-length Replay는 별도 Diagnostic이다. 이를 Natural Serving 이득으로 사용하지 않는다.
Finite-cohort Throughput은 Sustainable Capacity가 아니다.

## 동결 후보

| Family | 설정 | D Cap | Credit | N1 Decode Point Screen |
| --- | --- | --- | --- | --- |


## c3_main_serving_v2

`NOT_RUN` 또는 분석 대기 상태다. 계획 파일과 실행 Receipt를 함께 확인한다.

## c3_heldout_serving_v2

`NOT_RUN` 또는 분석 대기 상태다. 계획 파일과 실행 Receipt를 함께 확인한다.

## c3_heldout_replay_v2

`NOT_RUN` 또는 분석 대기 상태다. 계획 파일과 실행 Receipt를 함께 확인한다.

## c3_continuous_iso8_output_bridge_v2

`NOT_RUN` 또는 분석 대기 상태다. 계획 파일과 실행 Receipt를 함께 확인한다.
