# Native와 Split의 요청별 Critical Path

## 범위와 검증

아래 표는 `TRACE_ANALYSIS`다.
LongBench400-v2 Main400, Natural EOS, QPS2.0, 기존 독립 Start3회만 사용했다.
새 정책 실험과 합쳐 평균을 계산하지 않았다.
기존 분석은 별도 Family209개 Start를 보존한다.
기존 PD 경로는191개 Start,41040개 요청 관측값이다.
새 Main39개와 Heldout30개 실행을 별도 Family로 추가했다.
전체 PD Host 경로는260개 Start,62400개 요청 관측값이다.
Agg와 Whole-prefill Deflection에는 존재하지 않는 P→D 경계를 만들지 않았다.

각 요청은 `Family + Start + Request ID + Prompt ID`로 연결했다.
같은 Start의 Native와 Split은 같은 Prompt, 순서, 예정 Arrival Trace를 사용했다.
Natural EOS의 실제 출력 길이는 Runtime과 Batch에 따라 다를 수 있다.
요청별 출력 수가 같은지도 별도 필드로 보존했다.
Equal-output Replay는 별도 Family다.
Heldout192는 Discovery208에서 분리했다. Main400의 일부이기도 하다.
따라서 Heldout192를 Main400과 요청이 겹치지 않는 새 Test Set이라고 부르지 않는다.

Correctness 및 Serial Calibration의 요청·Context 겹침도 검사했다.
Qualification32와 Prime용8개는 Main400과 요청·Context가 모두 겹치지 않는다.
두 Control Set도 서로 겹치지 않는다.
Main400의 고유 Context는 287개다. 고유 Request는 400개다.
Heldout192의 Context147개와 Discovery208의 Context140개는 서로 겹치지 않는다.
[Dataset Audit](dense_ar_native_ttft_joint_slo_20261124/analysis/dataset_identity_and_overlap_audit.json)에 Count와 Hash를 보존했다.

원본, Config, Model Revision, Dataset Hash와 Source SHA는 [실행별 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/metrics_per_start.csv)에 있다.
GPU UUID와 Scheduler 설정은 실행별 `actual_runtime_prefill.json` 및 `actual_runtime_decode.json`에 있다.
GPU Pair와 Arrival Seed도 CSV에 있다.

## Timestamp 계약

다음 Host 경계를 순서대로 사용했다.

`Arrival → P Submit → P Accept → P Prepare → P Ready → D Submit → D Accept → D Prepare → First Worker → First Client`

| 경계 | 실제 관측 | 해석 제한 |
|---|---|---|
| Arrival | 사전 생성한 Poisson 예정 시각 | Coordinator가 실제 Enqueue한 시각과 구분 |
| P Submit | Coordinator의 Ingress 전송 Queue 등록 | Worker 수락이나 GPU 시작이 아님 |
| P Accept | Worker가 요청을 수락한 시각 | Scheduler가 즉시 실행한다는 뜻이 아님 |
| P Prepare | 첫 실제 Query 준비 기록 | CUDA Kernel 시작이 아님 |
| P Ready | P 결과 메시지의 Worker 시각 | 요청의 GPU-exclusive Service가 아님 |
| D Submit | Coordinator의 D 전송 Queue 등록 | NIXL DMA 시작이 아님 |
| D Accept | D Worker의 요청 수락 | KV 수신 완료가 아님 |
| D Prepare | D의 첫 실제 Query 준비 | CUDA Kernel 시작이 아님 |
| First Worker | D 첫 Token 결과의 Worker 시각 | Client 전달 시각과 구분 |
| First Client | Coordinator가 첫 Token을 받은 시각 | 외부 HTTP 전달 시간은 포함하지 않음 |

각 요청의 인접 구간은 음수가 아닌지 검사했다.
아홉 구간의 합계는 해당 요청의 Client TTFT와 일치한다.
허용 수치 오차는 0.001ms다.
Host 평균 구간은 더할 수 있다. 각 구간의 p95를 더해서 TTFT p95를 만들 수는 없다.

Main400의 요청별 GPU Forward Start/End는 `NOT_INSTRUMENTED`다.
Host Prepare를 GPU Start로 바꾸지 않았다.
같은 시간에 다른 요청이 사용한 GPU 시간을 현재 요청의 전용 Compute로 배분하지 않았다.
Serial32 GPU Capture의 별도 결과는 [GPU·Intervention 보고서](dense_ar_pd_critical_path_controlled_interventions.md)에 있다.
Serial Batch1 비용을 Natural C3의 Operator 기여율로 대체하지 않는다.

## 공통 First Token 경로

검사한 기존 Main400 Native와 Split은 모두 `OFFICIAL_PROXY` 경로다.
첫 Client Token은 D Worker에서 왔다.
Native만 P의 첫 Sample을 바로 전달했다는 설명은 이 Family에서 반증된다.
다른 과거 `FIRST_TOKEN_REUSE` Family를 이 결과에 섞지 않았다.
[First-token Audit](dense_ar_native_ttft_joint_slo_20261124/figures/first_token_source_audit.csv)에 실행별 Source Count가 있다.

## Main400 Host 경로

단위는 ms다. 각 Start의 요청별 평균을 계산한 뒤 Start3회의 평균을 구했다.
아래 값은 Kernel 시간이나 DMA 시간이 아니다.

| Host 구간 | NativeStrong | NativeCommon0 | NativeCommon2 | Stock Token | Layer | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| Arrival→P Submit | 0.532 | 7.514 | 0.368 | 0.982 | 1166.290 | 3469.234 |
| P Submit→Accept | 119.811 | 118.691 | 120.260 | 74.818 | 113.866 | 84.809 |
| P Accept→Prepare | 2471.978 | 2662.084 | 2449.226 | 249.659 | 1440.218 | 253.076 |
| P Prepare→Ready | 851.370 | 859.011 | 850.510 | 538.703 | 804.920 | 615.984 |
| P Ready→D Submit | 1.041 | 11.483 | 0.890 | 1.203 | 1.264 | 1.237 |
| D Submit→Accept | 7.885 | 426.541 | 349.980 | 779.262 | 552.331 | 362.123 |
| D Accept→Prepare | 39.907 | 60.143 | 144.106 | 2920.414 | 1804.583 | 1515.083 |
| D Prepare→First Worker | 21.767 | 26.498 | 22.956 | 440.647 | 219.637 | 320.475 |
| Worker→Client | 0.283 | 0.283 | 0.313 | 0.471 | 0.688 | 0.282 |
| Arrival→P Ready | 3443.691 | 3647.300 | 3420.363 | 864.162 | 3525.294 | 4423.103 |
| P Ready→First Client | 70.883 | 524.948 | 518.246 | 4141.997 | 2578.503 | 2199.199 |

이 표는 비교 기준을 구분해야 함을 보여준다.

- Stock−NativeStrong: P Ready 이전은 약2579.529ms 짧다. 이후는 약4071.115ms 길다.
- Layer−NativeStrong: P Ready 이전은 약81.603ms 길다. 이후는 약2507.621ms 길다.
- Hybrid−NativeStrong: P Ready 이전은 약979.413ms 길다. 이후는 약2128.316ms 길다.
- Layer−Stock: P Ready 이전은 약2661.132ms 길다. 이후는 약1563.494ms 짧다.

이 값은 평균 Host 경로의 위치를 식별한다.
P GPU 절약량이나 D GPU 추가 비용을 이 값으로 확정하지 않는다.
Native 대비 회귀를 Layer-vs-Stock의 P Attention 설명만으로 대체할 수 없다.
비교 기준에 따라 관측되는 병목 위치가 다르다.

## 기존 표와 새 Timing Family의 Runtime Bridge

새 Timing은 기준 Commit의 최근 Runtime을 사용한다.
기존 v2 Main400 Raw의 Byte-for-byte GPU 재실행이라고 부르지 않는다.
Prompt·Arrival Offset·Seed·Output Contract·BI0·B/S는 기존과 일치하는지 검사했다.
Native·Stock·Layer·Hybrid의 기본 Action, Credit, D Cap과 Compiler 옵션도 유지했다.
다만 최근 공통 Coordinator 설정은 기존 v2와 다르다.

- Coordinator Evidence Write를 Buffering한다.
- 측정 중 Coordinator Cyclic GC를 지연한다.
- Worker Source Path와 NIXL Control Port를 고정한다.
- P Critical Trace를 사용한다. GPU Event Observer는 사용하지 않는다.

이 설정은 새 Family의 모든 Arm에 동일하게 적용한다.
새 Family 내부의 정책 비교만 One-factor Controlled Intervention으로 해석한다.
기존 v2와 새 Family 사이의 전체 지연 변화는 여러 차이가 함께 바뀐 비교다.
그 변화 전체를 Registration, GC 또는 Credit 하나의 인과 효과로 확정하지 않는다.
Source ZIP 차이와 Config 차이는 [Runtime Bridge Audit](dense_ar_native_ttft_joint_slo_20261124/analysis/historical_current_runtime_bridge_audit.json)에 있다.

## 새 Main400과 Equal-output의 위치 분해

새 Family도 동일한 아홉 Host 경계의 항등식을 검증했다.
각 행은 독립 Start3회의 평균이다. 단위는 ms다.
Main은 Natural400이다. Heldout은 고정 길이 Replay192다.
두 Family의 차이를 Output 하나만의 효과로 해석하지 않는다.

| Family·Method | Arrival→P Ready | P Ready→첫 Client | Mean TTFT |
|---|---:|---:|---:|
| Main NativeStrong | 3427.277 | 73.697 | 3500.975 |
| Main NativeCommon0 | 3385.197 | 463.084 | 3848.281 |
| Main NativeCommon2 | 3409.446 | 668.156 | 4077.602 |
| Main Stock Token | 793.547 | 2648.133 | 3441.680 |
| Main StockCap2 | 866.127 | 5147.951 | 6014.078 |
| Main Layer | 3087.235 | 2397.752 | 5484.988 |
| Main Hybrid | 4639.125 | 2248.951 | 6888.075 |
| Replay NativeStrong | 2517.976 | 69.310 | 2587.287 |
| Replay NativeCommon0 | 2522.229 | 628.909 | 3151.139 |
| Replay NativeCommon2 | 2512.023 | 980.876 | 3492.899 |
| Replay Stock Token | 828.209 | 2795.138 | 3623.347 |
| Replay StockCap2 | 824.383 | 3739.514 | 4563.897 |
| Replay Layer | 2624.596 | 2088.326 | 4712.921 |
| Replay Hybrid | 3019.834 | 2212.065 | 5231.899 |

새 Main Stock은 NativeStrong보다 P 이전을2633.730ms 줄였다.
P 이후는2574.436ms 증가했다. 두 효과가 평균에서 거의 상쇄됐다.
따라서 Mean TTFT는 약59.294ms 짧았다.
Native-relative p95 우위나 회귀는 CI가1을 포함했다.
원래 v2의 Post-P 추가 대기와 새 결과를 같은 값으로 취급하지 않는다.

새 Main Layer는 NativeStrong보다 P 이전이340.042ms 짧았다.
P 이후는2324.055ms 길었다. Mean TTFT 회귀는1984.013ms였다.
반면 Layer−Stock은 P 이전+2293.688ms, 이후−250.381ms였다.
비교 기준에 따라 설명해야 할 위치가 다시 다르다.
이는 실제 요청 경계의 평균 분해다. GPU 비용의 인과 비율이 아니다.

## Queue 위치 변경을 확인한 One-factor 개입

아래 부호는 Candidate−Control이다. 단위는 ms다.
각 행은 같은 Request·Arrival을 사용한 paired N3의 Host 평균 차이다.
독립적인 GPU 비용이나 p95의 가법 분해가 아니다.

| Family·개입 | P 이전 전체 | P 이후 전체 | D Submit→Accept | D Accept→Prepare | D Prepare→첫 Worker |
|---|---:|---:|---:|---:|---:|
| Main Stock Cap0→2 | +72.580 | +2499.818 | +7.437 | +2638.049 | −145.590 |
| Replay Stock Cap0→2 | −3.826 | +944.377 | −23.259 | +1107.700 | −140.055 |
| Main Layer RegOpen | +5.435 | −17.740 | −472.351 | +453.942 | +0.587 |
| Replay Layer RegOpen | +4.140 | +31.529 | −568.674 | +593.029 | +7.089 |
| Main Hybrid RegOpen | −129.145 | −56.861 | −350.905 | +306.199 | −12.143 |
| Replay Hybrid RegOpen | −6.252 | −6.212 | −504.799 | +475.713 | +22.937 |
| Main Native RegOpen | +16.341 | +19.329 | −425.515 | +445.630 | −0.948 |
| Replay Native RegOpen | +9.654 | −5.970 | −645.657 | +639.380 | +0.472 |

Cap 변경의 Replay에서는 P 이전 시간이 거의 같았다.
동일한 출력11208개를 생성했는데 P 이후 대기는 약944.377ms 증가했다.
짧아진 D Prepare→첫 Worker 경계가 긴 D 준비 대기를 만회하지 못했다.
따라서 이 회귀를 추가 P 계산이나 더 많은 출력만으로 설명할 수 없다.
Actual Batch와 Cohort 변화는 해당 정책의 총 효과에 포함된다.

Registration 지연을 해제하면 D Submit→Accept가 실제로 줄었다.
그러나 대기의 상당 부분이 D Accept→Prepare로 이동했다.
등록 경계만 짧아졌다고 E2E TTFT가 개선됐다고 판단할 수 없다.
논리적 Decode 상태, READ 준비와 Bulk Prefill 조건이 이후에도 남았다.
여러 조건이 겹치므로 이 내부 대기를 전부 한 Branch의 전용 비용으로 배분하지 않는다.

Main Layer Retirement는 P 이전−51.522ms, 이후+83.178ms였다.
Main Hybrid Retirement는 P 이전+1106.210ms, 이후+385.404ms였다.
Hybrid의 Coordinator Submit 이전 평균도1127.567ms 증가했다.
이것은 이후 Queue·Batch·Natural 출력이 함께 변한 총 효과다.
고정된 미래 상태에서 Credit 하나만 일찍 반환한 이론 효과와 구분한다.
안전한 반환이 항상 시스템 전체의 P 대기를 줄인다는 가설은 지지되지 않았다.

## D Accept→Prepare의 내부 경계

NIXL Host READ Call과 Engine의 Receive ACK를 별도로 연결했다.
아래 값은 세 Start의 모든 요청에서 경계가 순서대로 관측된 경우의 평균이다.
Main400의 각 실행은 400/400개 요청에서 이 조건을 만족했다.

| 구간 ms | NativeStrong | NativeCommon0 | NativeCommon2 | Stock Token | Layer | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| D Accept→READ Start | 3.414 | 15.984 | 20.140 | 1332.991 | 751.406 | 661.229 |
| READ Host Enclosure | 16.290 | 15.883 | 16.170 | 14.300 | 16.065 | 15.260 |
| READ End→Receive ACK | 14.659 | 17.750 | 18.266 | 170.162 | 49.861 | 62.241 |
| Receive ACK→D Prepare | 5.544 | 10.525 | 89.530 | 1402.961 | 987.251 | 776.353 |

Stock의 READ Host Call 자체는 평균14.300ms다.
Stock의 D Accept→Prepare는 평균2920.414ms다.
따라서 전체 D 대기를 READ API 실행시간 하나로 설명할 수 없다.
READ 이전과 ACK 이후의 대기가 더 크다.
그러나 READ와 Decode의 의존성은 정책 개입 없이 위 표만으로 인과 확정할 수 없다.
Receive ACK 시각에는 Engine Poll 지연이 포함될 수 있다.
해당 구간을 전용 DMA 시간이나 GPU Idle로 부르지 않는다.

이 내부 표는 바깥 D Accept→Prepare 구간의 세부 분해다.
바깥 구간과 내부 구간을 다시 더하지 않는다.
Hidden Import의 의존성도 KV READ와 항상 같지는 않다.
실제 Kernel과 Copy의 미관측 경계는 그대로 남긴다.

## KV 여유와 Queue의 구분

같은 Main400 실행의 D Scheduler Snapshot에서 Free KV Block을 검사했다.
아래 값은 Start별 관측 최솟값이다. Start-mean이나 GPU Memory 사용률이 아니다.

| Method | Start0 | Start1 | Start2 |
|---|---:|---:|---:|
| NativeStrong | 1918 | 1871 | 1810 |
| Stock Token | 619 | 392 | 406 |
| Layer | 404 | 404 | 319 |
| Hybrid | 1373 | 1139 | 1344 |

검사한 측정 구간의 Snapshot에서 Free Block0은 관측되지 않았다.
첫 상태가 측정 시작보다 늦으면 그 앞 구간을 0으로 채우지 않았다.
해당 Field의 시간 Coverage는 99.746%–100%다.
따라서 단순한 KV Pool 고갈만으로 수초의 D 대기를 설명할 근거는 없다.
특정 Allocation이나 Reservation의 모든 대기를 반증한 것은 아니다.
Free KV Block은 GPU Compute Headroom이나 Decode-safe Window를 뜻하지 않는다.
Field별 Coverage, 관측 최솟값과 조건부 평균은 [Policy CSV](dense_ar_native_ttft_joint_slo_20261124/figures/policy_credit_batch_per_start.csv)에 있다.

## Source Credit와 D 정책

Source Credit는 Coordinator의 Outstanding Window다.
Actual GPU Batch, Hidden Buffer Slot, KV Block 수와 다른 값이다.
기존 Stock은 Credit0이다. 여기서 0은 무제한 Coordinator Window를 뜻한다.
Layer는 Credit16, Hybrid는 Credit8이다.
두 Axis는 기본적으로 첫 Client Token에서 Credit를 반환했다.
따라서 D에서 첫 Token이 지연되면 P의 새 요청 제출도 지연될 수 있다.

기존 Layer의 Arrival→P Submit은 평균1166.290ms다.
기존 Hybrid는 평균3469.234ms다.
Credit Window가 가득 찬 구간과 이 대기의 겹침도 요청별로 보존했다.
이 겹침을 전용 Credit 인과 지연으로 이름 바꾸지 않았다.
원인 분리는 안전한 Source Retirement 반환 정책과 비교한다.

Coordinator가 예정 Arrival을 발견한 시각도 별도 검사했다.
기존 Main400에서 Layer의 Loop 발견 지연은 평균0.163ms였다.
발견→Credit Acquire는1165.881ms, Acquire→P Submit은0.246ms였다.
Hybrid는 각각0.542ms,3468.455ms,0.237ms였다.
큰 Submit 대기는 주로 Coordinator가 Arrival을 발견한 뒤에 발생했다.
긴 Credit 경계의 위치가 확인됐다는 뜻이다.
전체 경계를 전용 Credit 인과 비용으로 확정한 것은 아니다.
내부 경계는 바깥 Arrival→P Submit의 세부 구간이다.
아홉 Stage와 다시 더하지 않는다.

NativeCommon0은 Stock의 uncapped D 정책과 맞춘 Native 대조군이다.
NativeCommon2는 Layer/Hybrid의 D Cap2 정책과 맞춘 Native 대조군이다.
NativeStrong은 기존 검증된 강한 Native 경로다.
Common-D Policy가 Source Credit까지 모두 같다는 뜻은 아니다.
또한 D Cap2는 Actual Decode Batch2를 보장하지 않는다.
[Policy CSV](dense_ar_native_ttft_joint_slo_20261124/figures/policy_credit_batch_per_start.csv)에 실제 Geometry와 정책 차이가 있다.

## Artifact와 재현

요청별 Queue·Input·Arrival·실제 Batch와 TTFT의 기술 통계도 별도 생성한다.
[Feature Association CSV](dense_ar_native_ttft_joint_slo_20261124/figures/request_feature_associations.csv)는 Start별 Pearson·Spearman 값을 보존한다.
[Pass·Fail 조건부 CSV](dense_ar_native_ttft_joint_slo_20261124/figures/pass_fail_queue_features.csv)는 분모와 Missing Count를 함께 보존한다.
이 값에 인과 기여율이나 독립 표본의 p-value를 부여하지 않는다.
실제 Output Length와 실행된 Batch는 사후 결과다. 미래를 읽지 않는 Selector 입력으로 사용하지 않는다.
P Scheduler Snapshot에는 Coordinator가 아직 제출하지 않은 요청이 포함되지 않는다.
따라서 관측한 P Waiting Queue0을 전체 P 앞 Queue0으로 해석하지 않는다.

- [Main400 요청별 Host 경계](dense_ar_native_ttft_joint_slo_20261124/figures/main400_request_critical_paths.csv).
- [Native·Stock과 요청별 차이](dense_ar_native_ttft_joint_slo_20261124/figures/paired_native_stock_requests.csv).
- [요청별 Trace와 SLO 결합 CSV.gz](dense_ar_native_ttft_joint_slo_20261124/figures/request_trace_and_slo.csv.gz).
- [D 등록 지연의 직접 관측](dense_ar_native_ttft_joint_slo_20261124/figures/D_registration_request_breakdown.csv).
- [Host Stage Figure](dense_ar_native_ttft_joint_slo_20261124/figures/native_split_host_stage_breakdown.png).
- [Host Queue Timeline](dense_ar_native_ttft_joint_slo_20261124/figures/P_D_service_queue_host_timeline.png).

아래 명령은 기존 Raw를 읽는다. 기존 Raw를 다시 쓰지 않는다.
큰 공개 CSV는 검증한 무손실 Gzip으로 제공한다.
로컬 원본 CSV는 삭제하지 않는다. 압축 해제하면 같은 SHA256의 CSV가 나온다.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.request_trace
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.policy_trace
/home/esjung/.venvs/looped-pd-abc/bin/python -B -m poc_flashvep.dense_ar_native_ttft_slo.causal_analysis
```
