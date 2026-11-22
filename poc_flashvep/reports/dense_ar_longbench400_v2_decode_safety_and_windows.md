# Decode Tail 보호와 Window 판정

상태: ACTIVE_SNAPSHOT.
생성 시각: 2026-11-22T18:39:15.240981+00:00.
경과 연구 시간: 11.540시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 동결 계약

TTFT 제한은 5000ms다. 모든 Client-visible ITL 제한은 34ms다. Request TPOT 제한은 33ms/token이다.
상대 Gate는 ITL p99와 TPOT p95의 Paired CI 상한이 모두 1.05 이하여야 한다.
NativeStrong과 같은 D 정책을 사용하는 Common-policy Native를 모두 대조군으로 사용한다.
기존 절대 Decode Attainment 95% 조건도 별도로 검사한다.
TTFT 개선으로 Decode 실패를 상쇄하지 않는다. N1 또는 N2는 CI Gate를 통과할 수 없다.

## 두 Native 대비 Tail Gate

| Trace | QPS | Method | Starts | ITL/Strong [CI] | TPOT/Strong [CI] | ITL/Common [CI] | TPOT/Common [CI] | 상대 PASS | 절대 PASS | 통합 PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


## Kernel 및 Window

| Method | QPS | Role | 증거 | Kernel Busy % | ROI 초 | Observer Gate |
| --- | --- | --- | --- | --- | --- | --- |

Kernel Busy는 Clean/Profile/Clean Gate가 통과한 Capture만 적격 수치로 사용한다.
Observer Gate는 전체 Trace와 ROI 노출 Request를 각각 검사한다. 허용 변화는 5%다.
Kernel Gap은 모든 Engine의 Physical Idle 또는 실행 가능한 Decode-safe Window와 같지 않다.
Worker Deadline Residual은 마지막 Token 시각+34ms의 관측 Proxy다.
이 Proxy는 후보 준비, KV Credit, Stream 가용성 및 미래 Arrival 안전성을 보장하지 않는다.
Online Dynamic Insertion은 검증된 Window와 Tail Predictor가 없으면 `NOT_RUN`이다.
GPU Operator-exclusive GEMM/Attention, SM 및 HBM 비중은 적격 계측이 없으면 `NOT_MEASURED`다.
NIXL Host API와 Kernel Gap의 시간 중첩은 인과 관계를 입증하지 않는다.

[c3_kernel_copy_cpc_results_v2.json](dense_ar_longbench400_v2_20261122/analysis/c3_kernel_copy_cpc_results_v2.json)

[c3_kernel_busy_and_queue_v2.csv](dense_ar_longbench400_v2_20261122/figures/c3_kernel_busy_and_queue_v2.csv)
