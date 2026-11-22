# LongBench400: Prefill Norm과 Compiled Decode를 분리한 검증

상태: 독립 Start의 Terminal 결과를 확보했다.

## 변경과 대조군

실제 설치 소스의 CUDA Platform은 Compile 여부에 따라 IR Norm 우선순위를 바꾼다.
Mode0는 `vllm_c,native`를 사용한다. 기본 Inductor Compile은 `native`를 사용한다.
두 설정은 같은 BF16 수학을 구현하지만 모든 Tensor Byte가 같지는 않았다.
`custom_ops=all`만으로 IR Norm 우선순위가 바뀌지는 않았다.
기본 D Compile의 엄격한 불일치와 Global CUDA 우선순위의 N1/N8 통과를 모두 보존한다.
이 개입은 RMSNorm과 Fused-add RMSNorm의 Provider Bundle을 함께 바꾼다.
두 Operator 각각의 수치 기여율을 별도로 분해한 것은 아니다.

모든 D Prefill은 원래 CUDA Norm을 사용해 Eager로 실행한다.
Decode-only Full36은 원래 Native Compiled Model을 유지한다.
D의 Custom-op 설정은 `all`이다. P의 기존 Mode0 경로는 바꾸지 않는다.
Tensor, Sampling Token이나 Client Timestamp를 덮어쓰지 않는다.
원래 Full36 Single-GPU Teacher도 같은 Stage별 Provider 계약으로 독립 실행했다.
Teacher에는 Partition Adapter, NIXL과 수입 KV가 없다.
Teacher의 Warm State를 평가 Request로 세었던 CPU Assertion은 수정했다.
평가 ID0–7과36개 KV Plane을 검사한다. Warm State와 실패 Controller Log는 삭제하지 않았다.

## 동결한 성능 계약

I3의 원문 Prompt16개, Mean Input28296 Tokens, QPS1.253873을 유지한다.
Arrival Seed는2026112551/2552/2553이다. 각 Arm은 독립 Server Start3회다.
모든 새 Paired 성능은 GPU0/1을 사용한다. 원래 GPU4/5 결과와 CI를 합치지 않는다.
모델은 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.
BI0, Original BF16, Exact Causal Attention과 Full36 Decode를 유지한다.
B8192/S1024, KV42GiB/Block128, Prefix Cache Disabled를 유지한다.
Output16은 `ignore_eos=True`인 Mechanism Control이다. Natural EOS 결과가 아니다.
Layer8, Rows-matched Stock Token8, Hybrid4+정확한 L/8을 사용한다.
각 Offload의 Logical P/D Rows는12676608/3621888이다.
D Cap1과 Source Credit8을 모든 Offload Arm에 유지한다.
NativeStrong은 원래 Scheduler를 유지한다. Common-policy Native의 Cap은1이다.
실제 Batch와 Padding은 JSON에 별도 기록한다. B/S를 실제 Batch로 해석하지 않는다.
Source SHA256은 `2b004c881d5c2f7caf66d6c467957d43079e27a15694cae54b8ee90971504a9f`다.
Dataset SHA256은 `6429dde7a7dd44156636e7f9c8f041d7b7d0b6f91c80bc2cce1697dc51e8c780`이다.
TTFT5000ms, 모든 ITL34ms, TPOT33ms의 Frozen SLO를 바꾸지 않았다.

## 실측 결과

| Arm | 독립 Starts | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) | Raw Output Tokens/s | 관측 Max ITL(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NativeStrong | 3 | 6109.482200 | 8.975284 | 7.966278 | 9.701630 | 15.513732 | 11.521466 |
| NativeCommon | 3 | 6109.127819 | 8.800160 | 7.964016 | 9.696048 | 15.510810 | 11.349099 |
| Layer | 3 | 3877.841457 | 9.294024 | 8.277091 | 18.364846 | 18.753963 | 75.071637 |
| StockToken | 3 | 3137.411254 | 8.923986 | 8.062524 | 19.928978 | 19.928978 | 9.388762 |
| Hybrid | 3 | 3532.473552 | 9.236101 | 8.303362 | 19.160569 | 19.160569 | 10.575166 |
| LayerScoped | 3 | 3896.066958 | 9.239217 | 8.164296 | 18.711462 | 18.711462 | 9.474479 |
| HybridScoped | 3 | 3518.385320 | 9.044181 | 8.142871 | 19.249358 | 19.249358 | 10.500399 |

## 실제 Batch와 실행 경로

아래 Max는 완료한 Start에서 관측한 최댓값이다. 고정 Batch 또는 평균 Batch가 아니다.
D Cap1은 Admission 상한이다. P Batch와 Query Padding을 같은 값으로 고정하지 않는다.

Graph 열은 Engine 설정이다. 모든 Partial Prefill이 Graph로 실행됐다는 뜻은 아니다.

| Arm | 관측 P/D 최대 Request Batch | 관측 P/D 최대 Padded Query | P/D Compile Mode | D Graph 설정 |
|---|---|---|---|---|
| NativeStrong | 2/1 | 8192/1 | 3/3 | FULL_AND_PIECEWISE |
| NativeCommon | 2/1 | 8192/1 | 3/3 | FULL_AND_PIECEWISE |
| Layer | 2/1 | 8192/8192 | 0/0 | FULL_DECODE_ONLY |
| StockToken | 2/1 | 8192/7328 | 3/3 | FULL_AND_PIECEWISE |
| Hybrid | 2/1 | 8192/8192 | 0/0 | FULL_DECODE_ONLY |
| LayerScoped | 2/1 | 8192/8192 | 0/3 | FULL_AND_PIECEWISE |
| HybridScoped | 2/1 | 8192/8192 | 0/3 | FULL_AND_PIECEWISE |

## Paired 판정

| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode5% | 매 Start Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| Layer/NativeStrong | 0.634725 [0.610720,0.669139] | 1.035513 [1.021236,1.054204] | 1.039016 [1.037574,1.040328] | 1.892965 [1.508661,2.376472] | False | False |
| Layer/NativeCommon | 0.634762 [0.612433,0.666224] | 1.056120 [1.045189,1.067123] | 1.039311 [1.037274,1.042081] | 1.894055 [1.511814,2.374072] | False | False |
| Layer/StockToken | 1.236000 [1.139469,1.322007] | 1.041465 [1.025875,1.060339] | 1.026613 [1.023960,1.028719] | 0.921515 [0.915038,0.930581] | False | False |
| Hybrid/NativeStrong | 0.578195 [0.549213,0.616968] | 1.029059 [1.015062,1.045137] | 1.042314 [1.041109,1.043209] | 1.974984 [1.593661,2.458185] | True | True |
| Hybrid/NativeCommon | 0.578229 [0.550754,0.614278] | 1.049538 [1.033507,1.066067] | 1.042610 [1.038581,1.046392] | 1.976122 [1.596995,2.455703] | False | True |
| Hybrid/StockToken | 1.125920 [1.046715,1.188837] | 1.034975 [1.018784,1.053602] | 1.029871 [1.028816,1.031292] | 0.961443 [0.946496,0.983312] | False | True |
| LayerScoped/NativeStrong | 0.637708 [0.610261,0.676165] | 1.029407 [1.024739,1.034680] | 1.024857 [1.022558,1.026730] | 1.928693 [1.578789,2.375704] | True | True |
| LayerScoped/NativeCommon | 0.637745 [0.611973,0.673218] | 1.049892 [1.036595,1.065921] | 1.025148 [1.022009,1.029122] | 1.929803 [1.582097,2.373305] | False | True |
| LayerScoped/StockToken | 1.241809 [1.150306,1.321003] | 1.035324 [1.032931,1.038503] | 1.012623 [1.009029,1.015572] | 0.938907 [0.914368,0.974468] | True | True |
| LayerScoped/Layer | 1.004700 [0.999246,1.010471] | 0.994103 [0.974181,1.012312] | 0.986373 [0.985284,0.987564] | 1.018874 [0.998408,1.047423] | True | True |
| HybridScoped/NativeStrong | 0.575889 [0.541115,0.620855] | 1.007676 [1.005926,1.009377] | 1.022168 [1.019187,1.025767] | 1.984136 [1.599406,2.474488] | True | True |
| HybridScoped/NativeCommon | 0.575923 [0.542633,0.618146] | 1.027729 [1.017684,1.037632] | 1.022458 [1.021049,1.023752] | 1.985279 [1.602753,2.471990] | True | True |
| HybridScoped/StockToken | 1.121429 [1.050122,1.179215] | 1.013469 [1.009614,1.017550] | 1.009965 [1.006666,1.014003] | 0.965898 [0.949395,0.988077] | True | True |
| HybridScoped/Hybrid | 0.996012 [0.985239,1.006252] | 0.979221 [0.965802,0.991001] | 0.980672 [0.976983,0.985258] | 1.004634 [1.002212,1.007159] | True | True |
| StockToken/NativeStrong | 0.513531 [0.461977,0.588061] | 0.994285 [0.991968,0.996348] | 1.012082 [1.010987,1.013408] | 2.054189 [1.621555,2.597144] | True | True |
| StockToken/NativeCommon | 0.513561 [0.463273,0.585467] | 1.014071 [1.003547,1.026398] | 1.012369 [1.009491,1.014641] | 2.055371 [1.624942,2.594521] | True | True |

## 판정의 제한

NI는 NativeStrong과 정확한 Common-policy Native에 각각 적용한다.
두 Native에 대한 ITL p99와 TPOT p95 CI 상한이 모두1.05 이하여야 한다.
Absolute Decode Attainment≥95%도 모든 Start에서 별도로 요구한다.
실패한 Gate를 반올림하거나 큰 Client Interval을 제거하지 않는다.
CI는 Paired Start-level Empirical 95% CI다. N3의27개 Bootstrap 조합을 열거한다.
다중 비교 보정은 하지 않았다. 48개 Request를48개 독립 Start로 사용하지 않는다.
Within-axis 개입에는 D의 Compiler Dispatch와 Stage별 Kernel 정책이 함께 포함된다.
후속 Admission, Batch와 Queue도 바뀔 수 있다. 이를 순수 GPU 시간의 인과 효과로 표현하지 않는다.
GPU-exclusive Operator 비중과 SM/HBM은 이 Source에서 NOT_MEASURED다.
이 결과를 Natural400 또는 다른 Cut·QPS의 성능으로 확대하지 않는다.
원래54개와 등록 수정54개의 실패를 새 Source로 덮어쓰지 않는다.
Online Dynamic Window나 Axis Selection은 이 검사에 없다.
유한16개 Throughput을 Sustainable Capacity로 표현하지 않는다.

- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/eager_prefill_native_decode_confirmation_frozen_v2.json)
- [전체 결과와 CI](dense_ar_longbench400_2d_20261121/analysis/eager_compile_confirmation_v2_results_v1.json)
- [기존 수치 불일치와 Negative Control](dense_ar_longbench400_decode_compile_residual.md)

## 같은 실행의 Host 경계 진단

아래 값은 Start별 통계량의 평균이다. GPU-exclusive 시간은 아니다.
첫 표는 기록한 Query가 모두 DECODE인 Host Step이다. 첫 Worker 출력 전의 Step도 포함할 수 있다.
비동기 Engine의 첫 출력 대기와 후속 Decode를 같은 구간으로 해석하지 않는다.
이 Step의 Wall Time도 Kernel 시간이나 Physical Idle과 같지 않다.

| Arm | Client ITL p99(ms) | Worker Emission p99(ms) | Delivery p99(ms) | Decode-labeled Host Step p50(ms) | Decode-labeled Host Step p99(ms) | 연속 Labeled Step 사이 Host Gap p99(ms) |
|---|---:|---:|---:|---:|---:|---:|
| NativeStrong | 8.975284 | 8.880846 | 0.478861 | 8.041934 | 35.848807 | 0.750252 |
| NativeCommon | 8.800160 | 8.714259 | 0.471035 | 8.067977 | 31.408568 | 0.703613 |
| Layer | 9.294024 | 9.141199 | 15.310189 | 8.304894 | 102.822737 | 1.004327 |
| StockToken | 8.923986 | 8.849756 | 0.526356 | 8.014273 | 327.087214 | 1.098610 |
| Hybrid | 9.236101 | 9.085323 | 0.544302 | 8.323634 | 206.019628 | 0.888980 |
| LayerScoped | 9.239217 | 9.054026 | 0.479511 | 8.147433 | 109.226356 | 1.076485 |
| HybridScoped | 9.044181 | 8.955405 | 0.434011 | 8.201226 | 200.369759 | 0.738051 |

### 첫 Worker 출력 이후의 별도 Host Step 부분집합

각 Request의 첫 Worker 출력 이후 시작한 Step만 아래 부분집합에 넣었다.
Query는 Request당1개이며 Position은 원래 Prompt 길이 이상이어야 한다.
이것은 후속 Streaming Host 경계다. 모든15개 GPU Decode Step의 완전한 비용 합은 아니다.
기존 첫 표와 모든 Client Interval은 그대로 보존한다. Primary Safety와 Throughput은 바뀌지 않는다.

| Arm | 선택한 Host Steps/3 Starts | Labeled 중 부분집합 밖 Steps/3 Starts | 이후 Step p50(ms) | 이후 Step p99(ms) | 연속 이후 Step 사이 Gap p99(ms) |
|---|---:|---:|---:|---:|---:|
| NativeStrong | 672 | 48 | 8.030983 | 8.247261 | 0.232660 |
| NativeCommon | 672 | 48 | 8.052687 | 8.232504 | 0.215601 |
| Layer | 672 | 48 | 8.271294 | 8.689650 | 0.266014 |
| StockToken | 672 | 48 | 7.980995 | 8.336188 | 0.286714 |
| Hybrid | 672 | 48 | 8.285851 | 8.671585 | 0.260732 |
| LayerScoped | 672 | 48 | 8.124776 | 8.453219 | 0.278048 |
| HybridScoped | 672 | 48 | 8.161592 | 8.523289 | 0.249440 |

각 Percentile의 차이를 비용의 합으로 해석하지 않는다.
비교에는 Compiler·Kernel 정책과 이후 Admission·Batch Feedback이 포함된다.
한 Operator의 GPU 비용 감소가 원인이라고 단정하지 않는다.
실제 GPU-exclusive Operator Breakdown과 SM/HBM은 이 Source에서 NOT_MEASURED다.

- [Start별 값과 Paired CI](dense_ar_longbench400_2d_20261121/analysis/eager_compile_host_boundaries_v2.json)
