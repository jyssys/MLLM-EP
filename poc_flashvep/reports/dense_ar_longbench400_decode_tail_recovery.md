# LongBench400: Native와 Offloading의 Decode Tail 복원

상태: Cut4의 독립 Start 3회 Safety Confirmation은 `PASS`다. 최종 Static 비교는 진행 중이다.

## 1. 실패 조건

Prefill Offloading이 Decode ITL을 자동으로 보존하지는 않는다.
D의 Prefill Continuation은 비선점 GPU 작업을 추가한다. Host의 Transfer Setup도 다음 Decode 제출을 지연시킬 수 있다.
따라서 TTFT 개선만으로 성공을 판정하지 않는다.

초기 C3 Discovery200, QPS1.8에서 Native-matched ITL p99는 16.656ms였다.
Hybrid Last4+Suffix 약 L/8의 ITL p99는 48.590ms였다.
SLO-goodput은 각각 약 55.00과 21.93 output tokens/s였다.
이 값은 독립 Start 1회 결과다. Hybrid의 초기 Decode Protection은 실패했다.

초기 값과 후속 실패를 삭제하지 않는다. Client Timestamp와 Frozen SLO도 변경하지 않는다.
SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms다.
별도로 Native-relative ITL p99와 TPOT p95의 95% CI 상한≤1.05를 요구한다.

## 2. 경쟁 가설과 검사

1. D의 긴 Prefill과 Decode가 같은 Step에 섞여 Decode를 지연시킬 수 있다.
2. 큰 Remote KV READ Setup이 Decode Host 경로를 지연시킬 수 있다.
3. D Decode Concurrency와 Credit이 Tail을 바꿀 수 있다.
4. Layer/Token의 Numerical 또는 KV 오류가 잘못된 실행시간을 만들 수 있다.

각 가설은 별도로 검사했다. State·Coverage·Lifetime 검사와 성능 검사를 분리했다.
Host 구간의 겹침만으로 GPU DMA나 특정 UCX 함수의 인과성을 확정하지 않는다.

## 3. Intervention A: Prefill 계산 지연

Installed Scheduler의 할당 전 경로를 사용한다.
이미 Decode 중인 요청이 있으면 큰 Prefill의 실행을 지연한다.
원래 Scheduler 상태는 호출 후 복원한다. 예외 발생 시에도 복원한다.
이미 할당된 Query를 삭제하거나 Output을 덮어쓰지 않는다.

Native의 한 Query Continuation은 별도 예외를 유지한다.
Remote KV 예약과 Control 처리는 계속 허용한다.
이 정책은 Whole-request 직렬 Admission과 다르다.
실제 Decode Batch는 고정하지 않는다. 따라서 이를 Batch1/1/1 Mechanism Control이라고 부르지 않는다.

Natural32/400의 Native/Layer/Stock Token/Hybrid State·Lifetime·Quality Gate를 통과했다.
이 Gate는 성능 우위나 ITL 비열등성을 뜻하지 않는다.

## 4. Intervention B: Remote READ Setup 지연

Decode 중에는 새로운 큰 Remote READ Setup을 보류한다.
Decode가 없을 때 기존 READ 경로로 전달한다.
ACK, Heartbeat와 Retirement 관련 Metadata는 보존한다.
기존 Metadata를 직접 변경하지 않고 복사한다.

Installed NIXL Worker에서 `_read_blocks_for_req`의 호출 위치를 감사했다.
이번 경로의 두 호출은 `start_load_kv`에 있었다.
Handshake Thread가 준비한 요청도 이 경계에서 소비한다.
이는 Source Audit이다. READ API Enclosure를 순수 DMA 시간으로 해석하지 않는다.
READ Setup 지연은 관련 Transfer와 Stream 실행 시점도 바꾼다.
따라서 이 Intervention을 Host Wait만 바꾼 실험으로 해석하지 않는다.

정상 완료 경로의 Natural32/400 State·Lifetime·Quality Gate를 통과했다.
Cancel, 외부 실패 복구와 모든 Transport 조합의 안전성을 검증한 것은 아니다.
해당 범위는 `NOT_RUN`이다.

## 5. 같은 Trace의 복원 결과

다음 값은 Discovery200, QPS2.8, Seed2026112291, GPU6/7의 실측이다.
모든 행은 같은 Source Capsule과 Arrival Trace를 사용한다.
각 행은 독립 Start 1회다. 표의 비용과 비율에는 아직 독립 Start CI가 없다.

| Method | TTFT p95 | ITL p99 | TPOT p95 | SLO-goodput |
|---|---:|---:|---:|---:|
| Unchanged NativeStrong | 10,245.877ms | 19.630ms | 8.542ms/token | 41.620 output tokens/s |
| Native + Compute/READ 정책 | 10,396.762ms | 9.585ms | 8.206ms/token | 42.021 output tokens/s |
| Layer Last4 + Compute/READ 정책 | 9,536.662ms | 8.742ms | 8.222ms/token | 46.450 output tokens/s |
| Stock Token Last4-equivalent + Compute/READ 정책 | 6,499.060ms | 8.561ms | 7.935ms/token | 55.374 output tokens/s |
| Hybrid Last4+L/8 + Compute/READ 정책 | 9,212.702ms | 9.124ms | 8.293ms/token | 46.597 output tokens/s |

증거 수준은 `MEASURED_ONLINE`이다.
네 Method의 복원된 ITL p99는 8.561–9.585ms였다.
TPOT p95는 7.935–8.293ms/token이었다.
이 단일 Start에서는 각 Offloading Method가 공통 정책 Native 대비 두 Decode Point Gate를 통과했다.
그러나 이는 95% CI 기반 Safety PASS가 아니다.

Layer와 Token은 이 표에서 Last4-equivalent Rows를 목표로 했다.
16-Token Alignment의 작은 Aggregate Rows 오차는 Config에 기록했다. 이 표를 Exact Iso-work라고 부르지 않는다.
Hybrid Last4+L/8은 더 많은 D Token-Layer Rows를 실행한다.
따라서 이 표를 세 축의 Iso-work 성능 비교로 사용하지 않는다.

실제 Natural Output 합계도 같지는 않았다.
Native/Layer/Token/Hybrid는 각각 6043/6048/6053/5997 Tokens를 생성했다.
BI0 Batch Geometry와 Numerical Path 차이를 별도 Quality 계약으로 검사했다.
Task Quality PASS를 모든 Batch의 Greedy Token 동일성이라고 부르지 않는다.

전체 Config와 원시 Timestamp는 `RemoteSetupAB_Discovery200_q2.8_*_v1` 실행에 있다.
[Recovery Ledger](dense_ar_longbench400_2d_20261121/analysis/decode_tail_recovery_discovery_all_v1.json)는 Control과 불리한 결과도 포함한다.

## 6. 반증과 부작용

READ Setup 지연이 모든 TTFT를 개선한 것은 아니다.
Hybrid의 같은 Trace TTFT p95는 Compute-only 약 8405.197ms에서 약 9212.702ms로 악화됐다.
따라서 ITL 복원과 TTFT 개선을 같은 성과로 처리하지 않는다.

Whole-request D Quiescent Admission도 ITL을 낮췄다.
그러나 해당 정책은 Pipeline 동시성을 줄였다. TTFT 비용을 함께 보고한다.
공통 Native Control이 없는 Quiescent 후보는 최종 공정한 Static 선택에서 제외한다.

Default Compile을 복원한 Token도 Prefill/Decode를 그대로 섞으면 Tail이 나빴다.
따라서 Compile만으로 이번 Tail을 설명하거나 해결했다고 주장하지 않는다.
Compiled Layer와 Compiled Hybrid도 구분한다.
Layer의 Stage-matched Gate는 통과했다. Compiled Hybrid의 엄격한 Hidden Gate는 통과하지 못했다.
실패한 Compiled Hybrid를 Primary 성능 후보에 넣지 않는다.

## 7. 독립 반복과 최종 Gate

Cut4의 NativeStrong/NativeREAD/Layer/Stock Token/Hybrid를 새 Arrival Seed 3개로 검사했다.
이 검사는 기존 Discovery200을 사용한다. 최종 Held-out200 검사를 대체하지 않는다.
해당 TTFT 결과를 Static Cut 선택에 사용하지 않는다.

QPS는 2.8이다. GPU Pair는 6/7이다. Arrival Seed는 2026112491/2492/2493이다.
모든 Arm은 같은 Source Capsule, Request와 Arrival Trace를 사용한다.
실제 Continuous Batch를 기록했다. Batch1/1/1로 강제한 검사는 아니다.
Native와 Stock Token은 Installed Default Compile을 사용한다. Layer와 Hybrid의 Partial Path는 No-compile 경로다.
Native/Stock Token의 Source Credit은 0이다. Layer/Hybrid는 4다. 이 차이를 순수 GPU Axis 효과로 해석하지 않는다.

| Method | Start-mean ITL p99 | Start-mean TPOT p95 | NativeREAD 대비 ITL 비율 [95% CI] | NativeREAD 대비 TPOT 비율 [95% CI] |
|---|---:|---:|---|---|
| NativeREAD | 9.371477ms | 8.088167ms/token | Reference | Reference |
| Layer Last4 | 8.812412ms | 8.246946ms/token | 0.940344 [0.912351,0.969502] | 1.019631 [1.012937,1.027420] |
| Stock Token Last4-equivalent | 9.098622ms | 8.036325ms/token | 0.970885 [0.962896,0.980811] | 0.993590 [0.987247,1.000990] |
| Hybrid Last4+L/8 | 9.004169ms | 8.260455ms/token | 0.960806 [0.945380,0.979135] | 1.021301 [1.015808,1.026417] |

세 Offloading Arm 모두 두 Metric의 CI 상한≤1.05를 통과했다.
변경하지 않은 NativeStrong 대비 Gate와 Absolute Decode Gate도 통과했다.
NativeStrong의 Start-mean ITL p99와 TPOT p95는 17.303278ms와 8.436507ms/token이었다.
나쁜 Native Tail만 기준으로 Safety를 판정하지 않았다. 복원된 공통 정책 Native도 대조군으로 유지했다.

CI는 Paired Independent Server Start 3개를 27개 재표본으로 완전 열거했다.
Pointwise Empirical CI이며 다중 비교 보정을 하지 않았다. Request나 Token을 Start로 재표본하지 않았다.
이 결과는 선언한 정상 완료 경로의 Decode 복원을 지지한다.
모든 Cut, Cancel 경로, 새로운 Transport 또는 Best-static 우위를 검증한 결과는 아니다.

근거: [독립 Safety Confirmation](dense_ar_longbench400_2d_20261121/analysis/READ_policy_decode_protection_confirmation_v1.json).

최종 Static 후보는 Discovery에서 동결한다.
새 Source의 Full400 State·Lifetime·Quality Gate를 다시 요구한다.
그 뒤 Held-out200과 Main400에서 독립 Start 3회 비교를 수행한다.
Unchanged NativeStrong과 정확히 같은 D 정책의 Native를 모두 남긴다.

QPS2.8에서 선택한 Token과 같은 Credit8/Cut8의 Cap2 대조군도 최종 Test 전에 추가로 동결했다.
이 대조군은 D Running Admission 상한만 비교하기 위한 것이다.
Running Cap2를 Actual Decode Batch2라고 부르지 않는다.
원래 선택한 빠른 Token을 삭제하거나 대조군으로 교체하지 않는다.
Held-out200/Main400의 QPS2.8은 각각 독립 Start 3회를 사용한다.
Main400의 QPS4.5는 단일 Start 보조 측정이다.
공통 정책 Native와 두 Token 설정을 모두 보고한다.
이를 Online Dynamic Axis Controller라고 부르지 않는다.

근거: [사전 동결한 Token Tail 대조군](dense_ar_longbench400_2d_20261121/configs/final_token_tail_control_extension_v1.json).

Main400의 Start별 GPU Pair는 0/1, 2/3, 4/5다.
Primary Held-out200은 GPU0/1을 사용한다.
Hardware/Seed Block을 기록한다. Request나 Token을 독립 Server Start로 재표본하지 않는다.

Cut4 Safety 판정은 `PASS_ON_TESTED_CONTRACT`다. 최종 Best-static Serving 판정은 `PENDING_CONFIRMATION`이다.
향후 CI가 기준을 넘으면 해당 Cell은 실패로 남긴다.
Threshold, Client Timestamp 또는 Dataset을 바꾸어 PASS를 만들지 않는다.

## 8. 연구 범위

이번 정책은 Decode가 없을 때 큰 Continuation을 실행하는 보수적 Admission이다.
미래 Decode Deadline을 예측하는 검증된 Window Scheduler는 아니다.
Kernel Inactivity도 실행 가능한 미래 Decode-safe Window와 다르다.
따라서 Static Decode 복원 결과를 Dynamic Axis Selection의 성공으로 표현하지 않는다.

Native/Layer/Token/Hybrid의 ITL이 비슷한 것은 목표다.
측정값을 같게 만드는 것은 목표가 아니다.

## 9. Main400에서 재발한 Fused Hybrid Tail

QPS1.8의 동결된 Hybrid 후보는 Compute-only 정책을 사용했다.
새 Main400 Start2에서 ITL p99는 14.449714ms였다. TPOT p95는 9.572ms/token이었다.
같은 Trace의 NativeStrong은 각각 13.095227ms와 8.212ms/token이었다.
이 값은 독립 Start 1회다. 최종 3-start Safety CI의 실패 판정은 아직 아니다.

그러나 변경하지 않은 Native의 ITL Point Limit 13.749988ms를 넘었다.
재현 명령의 Point Assertion도 실패했다. 해당 실패는 보존한다.
원래 140-start 계획과 Static 선택은 바꾸지 않았다.

실제 Interval은 10,632개였다. Point Limit을 넘은 Interval은 112개였다.
그중 104개는 기록한 NIXL Host 구간과 겹쳤다.
Mixed Prefill/Decode Host Step과 겹친 Interval은 0개였다.
전체 Mixed Step도 0개였다. Absolute 34ms 위반 Interval은 14개였다.

이 결과는 Compute Mixing만으로 Tail을 설명하는 가설에 반한다.
NIXL Host 구간의 겹침은 추가 근거다. 그 자체로 Host 또는 DMA의 인과성을 확정하지 않는다.
Native와 Offloading의 실제 Decode Batch 및 Host Gap도 계속 검사한다.

별도 Adaptive Repair 설계를 새 실행 전에 동결했다.
같은 Fused Hybrid에서 READ 준비 지연만 바꾼다.
Cut, Source Credit4, Hidden Pool4, Model, Output, Timestamp와 SLO는 그대로다.
새 조합의 Natural32/400 State·Lifetime·Quality Gate를 먼저 요구한다.

같은 Main400 Trace의 Pilot 1개와 새로운 Arrival Seed 3개의 Confirmation을 계획했다.
새 Seed는 2026112661/2662/2663이다. Confirmation Pair는 GPU6/7이다.
각 새 Seed는 NativeStrong, NativeCompute, NativeREAD, NativeCompletion을 포함한다.
HybridCompute, HybridREAD와 강한 Compiled Stock Token도 모두 포함한다.
총 신규 Serving Start는 22개다. Proof 실행은 이 수에 포함하지 않는다.

이 Follow-up은 관측 후 설계한 Repair다. 원래 맹검 Main Test라고 표시하지 않는다.
Pilot 방향과 관계없이 동결한 새 Seed와 대조군을 유지한다.
수정 전 실패 결과를 새 정책으로 교체하지 않는다.

새 Fused/READ 조합은 Natural32/400 State·Lifetime·Quality Gate를 통과했다.
같은 Trace Pilot의 ITL p99는 14.449714→9.176005ms로 낮아졌다.
TPOT p95는 9.572→8.311ms/token으로 낮아졌다.
TTFT p95는 2737.017→2795.315ms로 늘었다.
이는 단일 Start의 READ-only Intervention이다. 독립 Start CI의 복원 PASS는 아니다.
최대 ITL은 62.165→71.288ms로 악화됐다. 모든 Outlier를 해결한 것은 아니다.
Absolute 34ms 위반 Interval은 14개에서 1개로 줄었다. 남은 위반도 삭제하지 않는다.

기존에 동결한 Layer 후보도 새 Seed 3개의 Control로 추가했다.
이 확장은 새 Repair Confirmation 성능을 보기 전에 동결했다.
Layer 3개 Start는 원래 7-arm Block 뒤에 실행한다. 같은 Randomized Order에 속하지 않는다.
같은 Source, Pair, Prompt와 Arrival을 유지한다. 이 별도 Time Block도 기록한다.
신규 Serving 수는 원래 Repair22개와 Layer Control3개를 합한 25개다.
원래 140-start Main 계획과 Repair22개 계획은 덮어쓰지 않았다.

현재 Repair 상태는 `CORRECTNESS_PASS_CONFIRMATION_PENDING`이다.
Pilot 복원을 모든 Seed나 모든 Method의 안전성으로 일반화하지 않는다.
근거: [동결 Repair 설계](dense_ar_longbench400_2d_20261121/configs/hybrid_READ_repair_frozen_v1.json),
[재발 Tail 진단](dense_ar_longbench400_2d_20261121/analysis/FinalC3_Main400_q1.8_s2_BestHybrid_v1_decode_tail_audit.json),
[별도 Repair 결과](dense_ar_longbench400_2d_20261121/analysis/hybrid_READ_repair_results_v1.json).
추가 Layer의 범위는 [동결한 Layer Control](dense_ar_longbench400_2d_20261121/configs/hybrid_READ_repair_layer_control_extension_v1.json)에 있다.
최종 Safety 판정은 변경하지 않은 Client-visible Metric과 독립 Start CI를 사용한다.

## 10. Adaptive Repair의 독립 Confirmation

이 검사는 원래 140-start Main의 결과를 교체하지 않는다.
Main400의 단일 Tail 증상 이후 설계한 별도 Adaptive Follow-up이다.
모든 새 Arrival Seed와 대조군은 새 성능 측정 전에 동결했다.
QPS1.8, Natural EOS, Cap2048, BI0와 B8192/S1024를 사용했다.
새 Seed는 2026112661/2662/2663이다. 각 Arm은 독립 Start 3회다.
각 Start의 원본 Request는 400개다. 모든 새 Start는 GPU6/7을 사용했다.
Layer Control은 원래 Discovery에서 동결한 후보다.
Layer Control 3회는 7-arm Block 뒤에 실행했다. 같은 Randomized Order로 처리하지 않는다.
실제 Continuous Batch는 강제하지 않았다. Runtime과 Actual Work를 별도로 기록한다.

| Arm | TTFT p95(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | SLO-goodput(output tokens/s) | 실제 Output Tokens/Start |
|---|---:|---:|---:|---:|---:|---:|---|
| NativeStrong | 4185.528179 | 14.927791 | 69.660590 | 71.528528 | 8.338425 | 47.675191 | [10975, 11078, 11051] |
| NativeCompute | 4167.884204 | 15.206545 | 138.537119 | 213.090625 | 8.362622 | 46.868015 | [11007, 10954, 11034] |
| NativeREAD | 4249.753211 | 8.786864 | 73.401802 | 83.844386 | 7.936119 | 48.184255 | [11011, 11034, 11076] |
| NativeCompletion | 4248.408158 | 8.970609 | 549.993959 | 1060.002280 | 7.969876 | 48.164484 | [11062, 11060, 11066] |
| LayerControl | 2967.271854 | 9.095082 | 408.851584 | 1089.738455 | 8.299148 | 48.881195 | [11024, 10979, 10977] |
| StockToken | 2302.863664 | 9.014037 | 204.129795 | 491.610855 | 8.244158 | 49.612368 | [11188, 10988, 11054] |
| HybridCompute | 2792.208691 | 11.007910 | 162.917829 | 358.003408 | 9.209552 | 47.588989 | [10999, 10984, 10987] |
| HybridREAD | 2834.753089 | 9.158424 | 65.270386 | 67.978878 | 8.319373 | 49.171031 | [10951, 10983, 10947] |

표는 Start별 Metric의 평균이다. Max 열도 3개 Start Max의 평균이다.
관측 전체 Max 열은 별도로 모든 유효 Start의 단일 최댓값이다. 모집단의 최악 지연 상한은 아니다.
Start-mean Max를 관측 전체 Max 또는 Pooled p99로 해석하지 않는다.
정확한 각 Start의 Max와 원시 Token Event는 Raw에 보존했다.
Natural Output 수가 다르면 동일한 생성 작업량을 가정하지 않는다.

| Arm / Reference | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| HybridREAD / NativeStrong | 0.613515 [0.560632,0.683950] | 0.997715 [0.987654,1.010913] | 0.677275 [0.584995,0.779175] | 1.031376 [1.019402,1.045387] | True | True |
| HybridREAD / NativeREAD | 1.042286 [1.039800,1.045405] | 1.048292 [1.044526,1.053354] | 0.667039 [0.575920,0.773670] | 1.020479 [1.006213,1.032685] | False | True |
| HybridREAD / HybridCompute | 0.831986 [0.776087,0.917384] | 0.903342 [0.867744,0.934138] | 1.015237 [0.999693,1.029765] | 1.033244 [1.026609,1.040813] | True | True |
| HybridCompute / NativeStrong | 0.737411 [0.643546,0.855908] | 1.104471 [1.073514,1.142007] | 0.667110 [0.583200,0.763965] | 0.998192 [0.991047,1.004883] | False | True |
| HybridCompute / NativeCompute | 0.723893 [0.619069,0.871906] | 1.101276 [1.049679,1.163756] | 0.669934 [0.579429,0.776463] | 1.015383 [0.989977,1.040776] | False | True |
| HybridREAD / StockToken | 1.016018 [1.011331,1.020951] | 1.009123 [1.001040,1.016338] | 1.230969 [1.132285,1.326660] | 0.991104 [0.981726,1.001015] | True | True |
| StockToken / NativeStrong | 0.603843 [0.554345,0.669894] | 0.988695 [0.980370,0.997112] | 0.550197 [0.516540,0.587220] | 1.040633 [1.036930,1.044725] | True | True |
| StockToken / NativeCompletion | 1.004841 [0.990637,1.016438] | 1.034415 [1.027273,1.040861] | 0.542053 [0.509183,0.581844] | 1.030061 [1.011829,1.054581] | True | True |
| LayerControl / NativeStrong | 0.609272 [0.553330,0.681600] | 0.995290 [0.983279,1.006660] | 0.708936 [0.613612,0.825288] | 1.025296 [1.013416,1.035337] | True | True |
| LayerControl / NativeREAD | 1.035077 [1.025623,1.044558] | 1.045744 [1.036371,1.058397] | 0.698222 [0.604116,0.819508] | 1.014464 [0.997877,1.030866] | False | True |
| HybridREAD / LayerControl | 1.006964 [1.000608,1.014339] | 1.002437 [0.995205,1.007961] | 0.955340 [0.933402,0.987078] | 1.005929 [0.997578,1.012649] | True | True |

비율 CI는 같은 Seed Block을 재표본한 Pointwise Empirical 95% CI다.
N3에서 가능한 Bootstrap 27개 재표본을 완전 열거했다. 다중 비교 보정은 하지 않았다.
HybridREAD와 LayerControl은 NativeStrong과 NativeREAD를 모두 검사한다.
Stock Token은 NativeStrong과 같은 Completion Guard를 가진 NativeCompletion을 모두 검사한다.
빠른 TTFT만으로 안전한 Method로 처리하지 않는다. 두 Decode Metric과 Absolute Gate를 함께 사용한다.

### Runtime과 Actual Geometry

| Arm | Source Credit | Compile Family | P/D 최대 Request Batch | P/D 최대 Padded Query |
|---|---|---|---|---|
| NativeStrong | [0] | INSTALLED_DEFAULT_COMPILE | 3/4 | 8192/4 |
| NativeCompute | [0] | INSTALLED_DEFAULT_COMPILE | 3/4 | 8192/4 |
| NativeREAD | [0] | INSTALLED_DEFAULT_COMPILE | 3/4 | 8192/4 |
| NativeCompletion | [0] | INSTALLED_DEFAULT_COMPILE | 3/4 | 8192/4 |
| LayerControl | [4] | MATCHED_NO_COMPILE | 3/3 | 8192/8192 |
| StockToken | [0] | INSTALLED_DEFAULT_COMPILE | 4/4 | 8192/8192 |
| HybridCompute | [4] | MATCHED_NO_COMPILE | 3/3 | 8192/8192 |
| HybridREAD | [4] | MATCHED_NO_COMPILE | 3/3 | 8192/8192 |

이 비교는 Best-versus-best Continuous Family 비교다. Actual Batch1/1/1 또는 Exact Iso-work Control이 아니다.
READ 지연은 관련 Transfer와 Stream 시점도 바꾼다. 순수 Host 또는 DMA 비용의 단독 원인으로 해석하지 않는다.
같은 Source의 READ-only A/B는 Runtime 경로의 인과적 근거를 제공한다.
모든 GPU Operator의 비용 비중이나 미래 Decode-safe Window를 식별한 것은 아니다.
이 Repair의 성능을 원래 맹검 Held-out Main의 성능으로 표시하지 않는다.
동일 Request의 새 Arrival Seed 검증이다. 완전히 새로운 Request Population의 검증은 아니다.

전체 Source, Request, 실제 Geometry와 Host Tail 근거는 [Repair 분석](dense_ar_longbench400_2d_20261121/analysis/hybrid_READ_repair_results_v1.json)에 있다.
