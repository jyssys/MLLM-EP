# BI0 Decode-safe Window와 Split-size Scheduler

## 안전 계약

물리적인 Kernel Gap과 실행 가능한 Safe Window를 구분한다. 과거의 빈 구간은 미래 실행 예산이 아니다. 추가 Prefill의 상태가 준비되지 않은 구간도 Opportunity로 세지 않는다.

Active Decode가 있는 Window의 후보는 다음 조건을 충족해야 한다.

`Tail(D Forward + Handoff + Admission + Control) + Margin <= Next Decode Deadline − Candidate Ready Time`

Tail Cost의 중앙값만 사용하지 않는다. 새 Decode-ready 요청과 비선점 Kernel도 고려해야 한다. 현재 수집한 Host Quiescence만으로 이 조건을 통과했다고 판단하지 않는다.

## 구현한 제한적 안전 경계

`CORRECTNESS_ONLY`와 후속 `MEASURED_ONLINE`을 구분한다. Controlled Runtime은 P Prefill Batch1, D Suffix Batch1, D Decode Batch1을 사용한다. Source Credit은4다. 모든 Worker는 B8192/S1024와 KV42GiB를 사용한다. Source는 D Reservation ACK를 기다린 뒤 P Prefill을 시작하는 예전 경로를 사용하지 않는다.

D는 다음 조건을 모두 확인한 뒤 새 요청을 받는다.

1. 기존 Scheduler Request가 없다.
2. Engine에 미완료 요청이 없다.
3. NIXL Pending Receive와 Active Read가 없다.
4. Device-wide CUDA Synchronization이 완료됐다.
5. Parent는 현재 요청이 끝날 때까지 다른 D 요청을 Dispatch하지 않는다.

이 경계는 이미 실행 중인 Decode 사이에 Suffix를 삽입하지 않는다. 현재 Decode를 임의로 선점하지도 않는다. 다음 요청의 Admission과 TTFT는 지연될 수 있다. 따라서 Quiescent-only 실행을 Active-decode Safe-window Scheduler로 표현하지 않는다.

## Window Branch Control

초기 `WINDOW` 정책은 P 시작 시점의 현재 상태만 읽는다. D Active, D Pending 또는 다른 Source Work가 있으면 Native로 돌아간다. P Backlog가 없으면 Offloading하지 않는다. 유효한 상태에서는 Device-wide Idle Certificate를 추가로 확인한다.

이 정책은 미래 Arrival이나 실제 EOS를 읽지 않는다. 그러나 현재 D 상태와 미래 P Cut-ready 시점은 다르다. D가 P Prefix 실행 중 끝날 수 있다. 현재 상태만으로 거절하면 이런 후보를 놓친다. 반대로 과거의 빈 구간을 미래 Window로 사용하면 안전성을 과장한다.

Burst N8 검사는 Layer/Token Branch와 Native Fallback의 상태·수명 계약을 확인한다. 이 검사의 CPU State Capture는 성능을 교란한다. 해당 Timing을 Online Speedup으로 사용하지 않는다. 성능 결과와 Full200 Task Quality는 별도 Gate다.

## 구현하지 않은 기능

Active Decode의 Next-deadline 직접 삽입, 안전한 Tail Coverage를 갖춘 Dynamic Size, 복잡한 Online Axis Controller는 최종적으로 `NOT_RUN`이다. 미래 Safe Budget과 Tail Coverage Gate가 통과되지 않았다. 조건을 확인하지 않은 상태에서 Guard를 제거하지 않았다. Quiescent-only Static의 실제 Online 성능은 별도로 측정했다.

동일한 Decode Safety에서 고정 Cut보다 좋은 결과가 필요하다. 작은 고립 Forward 비용이나 Host Queue 감소만으로 Dynamic Size의 독립적 이득을 주장하지 않는다. Native-relative ITL p99와 TPOT p95의95% CI 상한≤1.05도 별도로 확인한다.

## 최종 Gate 분리

QPS2.35의 Quiescent-only Static Layer/Token은 각각 독립 Start3회에서 Native-relative Decode5% Gate를 통과했다. 이는 이미 실행 중인 Decode 사이에 작업을 삽입한 결과가 아니다. Active-decode Window의 물리적 측정과 미래 예측 검증은 `HOLD`다.

작은 Validation Set에서 Layer8 D Service의 예측 Upper Coverage는11/15였다. 안전한 Tail Bound나 WCET로 사용하지 않았다. Conditional Replay에서도 검증된5% Axis 추가 Headroom을 확보하지 못했다. 따라서 Dynamic Size와 Dynamic Axis의 Online 이득을 주장하지 않는다.
