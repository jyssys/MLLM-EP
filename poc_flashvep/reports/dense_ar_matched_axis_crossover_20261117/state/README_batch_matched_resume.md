# 배치 일치 연구 재개 메모

## 2026-11-18 최종 상태

이 절이 아래의 이전 진행 메모보다 우선한다.
실제 Batch1 성능은 98/98회 완료했다.
실제 준비1/Decode2 성능은 42/42회 완료했다.
390개 적격 성능 Start의 45,000개 요청을 검증했다.
GPU 실험은 `CAMPAIGN_CLOSED.json`으로 종료했다.
보류된 원래 Matrix는 완료로 표시하지 않았다.
기존 GPU Controller를 다시 시작하지 않는다.
GPU0–7의 승인된 Burner를 복원했고 양의 사용률을 3회 연속 검사했다.
완료 시 복원 누락의 회귀 검사도 추가했다.
최종 결과는 `../../dense_ar_matched_axis_crossover_final_summary.md`에서 확인한다.
Publication의 실제 Push 여부와 SHA는 `publication_receipt.json`에서 확인한다.
Receipt가 아직 없으면 보고서·Figure·Archive의 Private Index Publication만 남은 상태이다.

## 현재 사용자 요청

작은 Cut의 Layer 이득과 큰 Cut의 Token 이득이 Online에서도 존재하는지 검사한다.
실제 배치를 맞춰야 한다.
원하는 역전을 강제하지 않는다.
GEMM, Attention, 전송과 Queue의 기여를 구분한다.
사용자는 실행을 유지하고, 중간보고를 최소화하기를 원한다.

Last2 전체 Sweep은 취소 상태이다.
Last2의 제한된 대조를 추가할지 비동기 질문을 보냈다.
아직 답을 받지 않았다. 승인 없이 Last2 Serving을 추가하지 않는다.

## 기존 범위 전환

`scope_revision.py`가 Quiet-lane Lock을 확보한 뒤 기존 Controller 4개만 종료했다.
진행 중인 GPU 작업은 정상 종료했다. 종료한 GPU 작업 수는 0이다.
기존 Discovery는 198/232개이고 Admission Control은 52/52개이다.
기존 120개 Confirmation과 나머지 Discovery는 보류 상태이다.
기존 완료 Marker를 가짜로 생성하지 않는다.
전환 기록은 `user_batch_matched_scope_revision_20261118.json`이다.
이전 외부 SIGTERM 실패 1개와 대체 실행도 그대로 보존한다.

기존 최종화 코드는 아직 이 범위 전환을 처리하지 못한다.
`closeout.py`와 `validate.py --final`은 이전 모든 Marker를 요구한다.
최종화 전에 새 범위를 명시적으로 지원해야 한다.

## 실제 배치1 캠페인

Controller PID는 3616418이다.
모듈은 `poc_flashvep.dense_ar_matched_axis_pd.batch_matched`이다.
현재 모니터 PID는 3760668이다. 이전 PID 3616665는 종료했다.
모니터 모듈은 `batch_matched_monitor`이다.
둘 다 독립 Session에서 실행한다.

설정은 `batch_matched_design_20261118.json`에 동결했다.
P 준비, D Prefill 준비, D Decode의 실제 요청 수는 모두 1이다.
Source Macro도 1이다. 기존 Macro8 대조와 합치지 않는다.
B8192/S1024, Source Window32, 기존 KV/NIXL/정확성 계약을 유지한다.

Arm은 Native와 Last4/8/12의 Layer/Token이다.
성능 Matrix는 98개 Start이다.
Discovery 56개는 N72이다.
Output16의 QPS는 2/4/6/8/10이다.
Output64의 QPS는 2/6/10이다.
Discovery Seed는 2026111810이다.
Confirmation 42개는 N144, QPS6, Output16/64이다.
Confirmation Seed는 2026111820/2026111821/2026111822이다.

N18 Full-state Gate는 7개 Arm 모두 통과했다.
실제 배치 검사는 각 실행의 `_matched_batch_gate.json`으로 한다.
추가 실행 결과를 확인할 때 `batch_matched_monitor.snapshot()`을 사용한다.
이전 모니터의 완료 조건은 보류한 Marker 때문에 계속 False이다.

## Projection API 원인 분석

기존 고립 측정은 H100에서 F.linear를 호출했다.
실제 Serving은 Batch-invariant Linear를 직접 호출한다.
새 고립 대조는 `isolated_serving_projection_API_20261118_v1`이다.
원래 BF16 Weight를 사용했다. Serving Forward는 변경하지 않았다.
2개 GPU, 19개 M Shape, 4개 Projection의 152개 결과를 완료했다.
16개 고립 A/B/A Profiler Gate도 통과했다.
Kernel 이름은 `matmul_kernel_persistent`이다.
Online Profiler Gate를 통과한 것은 아니다.

GPU1의 대표 Layer0 모델에서 Token/Layer 비용 비율은 다음과 같다.
Last4는 1.1044103259408606이다.
Last8은 1.0748595252699766이다.
Last12는 1.0485607232607337이다.
이는 구조적 모델이며 Online GEMM 실측 합이나 TTFT가 아니다.
기존 측정과 새 측정의 차이를 전부 API 효과로 해석하지 않는다.

API 보고서는 `dense_ar_serving_projection_api_measurement_audit.md`이다.
모델 결과는 `analysis/actual_projection_API_geometry_model.json`이다.
`batch_matched_costs.py`는 이제 이 실제 API 데이터만 Projection 모델에 사용한다.
기존 Contiguous FA3 값도 Serving Attention 실측이 아닌 모델로 남긴다.

## 고정 Decode2 정합성 Prototype

Controller PID는 3744687이다.
모듈은 `poc_flashvep.dense_ar_matched_axis_pd.fixed_cohort_probe`이다.
현재는 정합성만 검사한다. 성능 실행은 아직 없다.
설정은 `fixed_cohort2_qualification_design.json`에 있다.
새 Adapter는 `fixed_matched_cohort.py`이다.
새 Adapter의 Freeze SHA256은 다음과 같다.
`1b141ac26210bf6928448588a6fb9630922fc60c328ddd5e09611de6058bb530`

새 조건은 P 준비1, D Suffix 준비1, D Decode2이다.
Native는 원래 Cached Query를 Decode2로 실행한다.
Native 초기 Input Flag가 prefilling이어도 Prompt Suffix 계산은 아니다.
상태가 준비된 요청은 같은 Cohort의 다른 Prompt가 제출될 때까지 Decode를 기다린다.
Token 준비도 요청 1개씩 실행한다. Token만 GEMM Batch2를 받지 않는다.
첫 토큰 알림은 Cohort 전체의 첫 알림이 존재할 때 전달한다.
생성 Timestamp는 변경하지 않는다. 대기 비용은 Client TTFT에 포함한다.
Native는 같은 Graph Replay에서 첫 알림 2개를 보낸다.
Native의 정상 첫 ITL을 인위적으로 0으로 만들지 않는다.

CPU 검사 6개가 통과했다.
Native N18 State Gate와 실제 Geometry Audit는 통과했다.
Native 실제 Decode 입력은 초기 9개와 후속 126개 모두 Batch2이다.
초기 Prefill Flag는 Cached Query 분류이며, 모두 Full36 Decode이다.
First-token Barrier는 9개이고 Native Hold는 수십 us 수준이었다.
이 N18 시간은 정합성 계측이며 성능 결과로 사용하지 않는다.
Layer/Token 상태는 진행 중이므로 새 진행 파일을 읽는다.

### Version1 실패와 Version2 재검사

Version1의 Layer Last4는 정상 종료하지 못했다.
실제 Scheduler Output은 q0 Decode1과 q1 Suffix8000을 함께 제출했다.
기존 Partial Prefill 안전 검사가 이를 차단했다.
실패 이름은 `fixed_cohort2_state_Layer_Last4_N18_v1`이다.
실패 Receipt와 Config의 정확한 Hash를 `preserved_fixed_cohort_prototype_failures.json`에 보존했다.
이 실패를 성능 또는 Clean Correctness 결과로 사용하지 않는다.
같은 Hash의 이 실패만 배치1의 별도 실패로 분류한다.
다른 Prototype 실패나 배치1 실패는 자동으로 제외하지 않는다.
기존 8개 Frozen Inference 파일은 여전히 같다.

Version2 Controller PID는 3763176이다.
모듈은 `fixed_cohort_probe_v2`이다.
새 Runtime 파일은 `fixed_matched_cohort_v2.py`이다.
Version1의 Runtime 파일은 변경하지 않았다.
새 Frontier는 Remote KV의 Computed Hint가 아니라 실제 로컬 Suffix 제출 Receipt를 사용한다.
3개 Frontier 회귀 검사와 기존 6개 Prototype 검사가 통과했다.
Version2도 정합성 전용이다. 성능은 아직 시작하지 않았다.

새 Design과 진행 파일은 `fixed_cohort2_v2_qualification_*.json`이다.
`fixed_cohort_audit --version 2`로 실제 Geometry와 두 Runtime 파일의 Source ZIP Hash를 검사한다.
Version2도 실패하면 새 실패를 보존한다.
근거 없이 같은 Runtime 실패를 반복하지 않는다.
Version2 성능 실험은 모든 State/Geometry 검사가 통과한 경우에만 설계한다.

`fixed_cohort_audit.py`는 실제 입력, Source ZIP의 Adapter Hash와 Barrier를 검사한다.
모든 State/Geometry 검사가 통과하기 전에 성능을 시작하지 않는다.
Prototype 실패는 보존하고 원인을 검사한다.
배치1의 Frozen Inference 경로와 Prototype을 혼동하지 않는다.

Prototype이 통과하면 별도 성능 설계를 먼저 동결한다.
권장 첫 대조는 Output16, N72, QPS2/6/10의 7개 Arm이다.
배치1과 같은 Arrival Seed 2026111810을 사용하면 Batch 효과를 짝지을 수 있다.
긍정적 근거가 있을 때 QPS6/N144/독립 Start3회를 추가한다.
별도 Stage를 사용하고 배치1 결과와 CI를 합치지 않는다.
이는 권장 후속 설계이다. 아직 실행하거나 성과로 보고하지 않았다.

## 분석과 검증

`statistics.py`가 batch1_discovery와 batch1_confirmation을 별도로 처리한다.
`quantile_audit.py`도 같은 `analysis_stage`를 사용한다.
새 Stage를 이전 Discovery에 잘못 묶던 KeyError를 수정했다.
회귀 검사를 실패 상태에서 실행한 뒤 수정했다.
CI의 Type7 정의와 역CDF 정의 자체는 변경하지 않았다.
이전 CI Snapshot도 보존한다.

분석 실행 순서는 다음과 같다.

1. statistics
2. quantile_audit
3. batch_matched_costs
4. batch_matched_figures
5. validate

위 순서로 실행하면 Figure의 통계 파일 읽기와 쓰기가 충돌하지 않는다.
Figure는 읽은 통계 Byte의 SHA256을 기록한다.
N72 Discovery와 N144 Confirmation을 같은 Curve에 합치지 않는다.

새 Figure는 Repository의 `poc_flashvep/`에 PNG/PDF로 생성한다.
이름은 `dense_ar_batch_matched_four_metrics_o16`과 `o64`이다.
정합성 실패나 미실행 Point를 정상 측정값으로 표시하지 않는다.

## 최종화의 남은 작업

새 Scope에 맞는 최종 검증과 자원 종료 조건을 구현한다.
이전 198개 결과와 새 98개 결과를 명확히 구분한다.
Prototype 성능을 추가하면 그 Manifest도 따로 검증한다.
보류한 원래 Matrix를 완료로 표시하지 않는다.
최종 보고서와 기존 요청 Figure에는 Scope 변경을 명시한다.
Offline 모델과 실제 Serving 우위를 구분한다.
NI 검사는 Native-relative ITL p99와 TPOT p95 CI 상한 1.05를 유지한다.
경계 결과는 Type7과 역CDF 민감도를 함께 보고한다.

이후 자원 복원, 보고서, Archive, Private Index Commit/Push를 수행한다.
기존 Publication Branch는 `flashvep/dense-ar-matched-axis-crossover-20261117`이다.
아직 새 Commit/Push는 하지 않았다.
User Checkout이나 Staged Index는 변경하지 않는다.
최종 GPU0~7 Burner를 복원하고 양의 Utilization을 3회 검사한다.
Remote SHA를 검사한 뒤 최종 보고한다.

사용자의 원래 선호 종료 시각은 2026-11-19 00:26:36 KST이다.
`user_finish_target_23_24h.json`의 Epoch를 유지한다.
전환했다고 종료 시계를 다시 시작하지 않는다.
