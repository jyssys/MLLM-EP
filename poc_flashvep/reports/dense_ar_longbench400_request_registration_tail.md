# LongBench400: D Request 등록과 첫 Decode Tail — QPS1.8

Evidence: `MEASURED_ONLINE_ADAPTIVE_REPAIR` 및 `TRACE_ANALYSIS`.
원래 Main140과 READ Repair는 유지했다. 이 문서는 별도 후속 검사다.
먼저 관측한 Heldout200 Host Gap을 사용해 가설을 세웠다.
새 성능 측정 전에 3개 새 Seed와 7개 대조군을 동결했다.
Confirmation Seed는 [2026112771, 2026112772, 2026112773]다. 모두 GPU Pair [2, 3]를 사용했다.
Natural400, QPS1.8, BI0, BF16, B8192/S1024와 KV42GiB/Block128을 유지했다.
Model Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.
Source Capsule SHA256은 `07cdfb84be94fc4ae3b1e778d58c8c1d03ff7a99452b0cd3ee667966600447cd`이다.
Prefix Cache를 사용하지 않았다. 모든 Decode는 Full36 Layers다.
Primary Latency는 Coordinator Client-visible 값이다. HTTP Server Latency가 아니다.
이 비교의 GPU-exclusive Operator 비중은 `NOT_MEASURED`다.
이는 같은 Request의 새로운 Arrival 검사다. 새로운 Request Population은 아니다.

QPS2.8 Extension은 첫 QPS1.8 Pilot 관측 후 동결했다.
그러나 모든 QPS2.8 성능 측정 전에 Seed와 Arm을 고정했다.
이를 원래 Main의 사전 맹검 Test와 같은 설계로 표시하지 않는다.

## 변경과 경쟁 가설

H1은 새 Request 등록과 Input Processing이 다음 Decode 제출을 지연한다는 가설이다.
H2는 IPC Receive 비용이 대기를 지배한다는 가설이다. 새 정책도 IPC Receive는 계속 수행한다.
H3은 첫 Decode 준비 또는 Batch 변동이 지배한다는 가설이다.
새 정책은 Output이 시작된 Running Request가 있으면 새 `engine.add_request` 호출을 미룬다.
기존 Decode Step과 NIXL ACK 및 Retirement는 계속 실행한다.
받은 Request는 버리지 않는다. Stop 전에 모든 추가 Request를 처리한다.
새 Admission 시점은 이후 Batch도 바꾼다. 단독 Python 비용의 인과 효과로 단정하지 않는다.

같은 Source의 등록-only A/B 대조군은 TokenRegistration과 TokenOldControl이다.
Hybrid의 READ-only A/B는 별도 READ Repair에서 수행했다.
이번 Hybrid와 원래 Compute-only Main의 차이를 등록-only 효과라고 부르지 않는다.

## 독립 Start 3회의 실제 Metric

| Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO-goodput(output tokens/s) | Mean Joint SLO(0–1) | Output Tokens/Start |
|---|---:|---:|---:|---:|---:|---:|---|
| NativeStrong | 3682.195197 | 14.685411 | 8.402706 | 47.020895 | 46.173476 | 0.990000 | [11058, 11013, 11044] |
| NativeRegistration | 3759.741055 | 8.742812 | 7.946945 | 47.117444 | 46.539887 | 0.990000 | [11058, 11103, 11029] |
| NativeCompletionRegistration | 3759.331107 | 8.746956 | 7.968561 | 47.065270 | 46.729328 | 0.993333 | [11078, 11042, 11038] |
| LayerRegistration | 2640.257311 | 8.873075 | 8.225188 | 46.854836 | 45.652340 | 0.990833 | [10984, 11024, 10966] |
| TokenRegistration | 2432.918372 | 8.629237 | 7.982386 | 47.139687 | 46.938302 | 0.997500 | [11012, 10972, 11195] |
| HybridRegistration | 2668.047908 | 8.888570 | 8.246027 | 46.730512 | 45.958081 | 0.994167 | [10957, 10966, 10976] |
| TokenOldControl | 2484.940215 | 8.893794 | 8.220281 | 47.205490 | 46.837971 | 0.995000 | [11010, 11001, 11215] |

Joint SLO는 TTFT와 Request별 Decode 조건을 함께 만족한 비율이다.
표는 Start별 비율의 평균이다. 평균 Joint≥95%가 모든 Start의 Joint≥95%를 보장하지 않는다.
Decode 보호와 Joint SLO를 구분한다. Raw Throughput과 SLO-goodput도 다른 Metric이다.
처리량은 유한400개 Arrival부터 Drain까지 측정했다. Sustainable Capacity가 아니다.

### 관측한 가장 큰 Client Interval

| Arm | 각 Start의 Max ITL(ms) | 관측 전체 Max ITL(ms) |
|---|---|---:|
| NativeStrong | [63.019522, 97.816836, 70.258817] | 97.816836 |
| NativeRegistration | [360.821176, 195.873743, 43.439066] | 360.821176 |
| NativeCompletionRegistration | [61.773678, 70.813878, 59.584247] | 70.813878 |
| LayerRegistration | [641.203169, 260.954801, 156.706296] | 641.203169 |
| TokenRegistration | [52.305806, 63.270712, 66.864716] | 66.864716 |
| HybridRegistration | [67.933165, 201.539109, 66.630304] | 201.539109 |
| TokenOldControl | [383.736381, 68.189844, 67.504413] | 383.736381 |

이는 관측 Max다. 미래 지연의 최악 상한이 아니다.
p99 또는 TPOT 비열등성 통과가 모든 Interval의 동일성이나 34ms 이하를 보장하지 않는다.
Request별 모든 ITL과 TPOT를 사용하는 Absolute Decode Attainment도 별도로 검사한다.

## Paired CI와 Decode 보호

| Arm / Reference | TTFT 비율 [95% CI] | ITL 비율 [95% CI] | TPOT 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| TokenRegistration / TokenOldControl | 0.979065 [0.964571,0.993347] | 0.970254 [0.964353,0.976641] | 0.971060 [0.967403,0.974344] | 1.002142 [0.998255,1.006378] | True | True |
| TokenRegistration / NativeCompletionRegistration | 0.647168 [0.590582,0.712242] | 0.986542 [0.983779,0.989522] | 1.001735 [0.997907,1.005423] | 1.004472 [0.995422,1.015951] | True | True |
| LayerRegistration / NativeStrong | 0.717034 [0.666122,0.761499] | 0.604210 [0.567837,0.643100] | 0.978874 [0.969758,0.990109] | 0.988714 [0.987535,0.989673] | True | True |
| LayerRegistration / NativeRegistration | 0.702244 [0.647173,0.748623] | 1.014900 [1.012826,1.017283] | 1.035012 [1.033650,1.036229] | 0.980929 [0.976739,0.986437] | True | True |
| TokenRegistration / NativeStrong | 0.660725 [0.606823,0.725305] | 0.587606 [0.548999,0.627236] | 0.949978 [0.937913,0.964052] | 1.016564 [1.004836,1.030930] | True | True |
| HybridRegistration / NativeStrong | 0.724581 [0.696286,0.751820] | 0.605265 [0.568858,0.643861] | 0.981354 [0.970724,0.992902] | 0.995335 [0.993726,0.997070] | True | True |
| HybridRegistration / NativeRegistration | 0.709636 [0.676899,0.739106] | 1.016672 [1.014028,1.019108] | 1.037635 [1.036320,1.039027] | 0.987499 [0.984807,0.990956] | True | True |
| HybridRegistration / LayerRegistration | 1.010526 [0.983191,1.048376] | 1.001746 [1.000796,1.002818] | 1.002534 [1.000088,1.005202] | 1.006697 [1.004582,1.008260] | True | True |
| HybridRegistration / TokenRegistration | 1.096645 [1.036991,1.147475] | 1.030053 [1.023784,1.037278] | 1.033028 [1.029560,1.035951] | 0.979117 [0.967172,0.988945] | True | True |
| LayerRegistration / TokenRegistration | 1.085222 [1.034622,1.138262] | 1.028257 [1.021079,1.035997] | 1.030417 [1.027031,1.033954] | 0.972603 [0.959279,0.984361] | True | True |

CI는 Paired Server Start 단위다. N3에서 27개 Bootstrap 조합을 완전 열거했다.
Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.
각 Arm은 NativeStrong과 같은 D 정책의 Native를 모두 통과해야 한다.
공통 Native의 정확한 Mapping은 다음과 같다: {'LayerRegistration': 'NativeRegistration', 'TokenRegistration': 'NativeCompletionRegistration', 'HybridRegistration': 'NativeRegistration'}.
실패한 Gate를 TTFT 개선으로 대체하지 않는다.

## 같은 Request의 Host 경계 차이

아래 값은 왼쪽 Arm에서 Reference를 뺀 Start별 평균의 Paired CI다.
단위는 ms다. 같은 Request ID, Prompt, Source, Pair와 Arrival을 비교한다.
여섯 구간의 평균 차이는 Mean TTFT 차이에 합산된다. 구간별 CI는 합산하지 않는다.
이는 TTFT p95 또는 GPU-exclusive Operator 비용의 분해가 아니다.

| Arm − Reference | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 | 첫 Output 이후 Residency 차이 |
|---|---|---|---|---|---|---|---|---|
| LayerRegistration − TokenRegistration | 148.842 [125.605,177.730] | 39.828 [34.873,45.710] | 0.346 [-0.590,1.241] | -52.686 [-60.808,-44.176] | -8.731 [-13.070,-3.091] | -0.262 [-0.566,-0.021] | 127.338 [94.976,166.502] | 5.704 [3.886,7.109] |
| HybridRegistration − TokenRegistration | 109.359 [87.420,132.622] | 21.337 [16.226,25.664] | -0.250 [-0.701,0.094] | -40.892 [-51.697,-31.726] | 32.409 [26.493,38.303] | -0.175 [-0.550,0.174] | 121.788 [97.698,147.769] | 5.011 [3.680,6.356] |
| HybridRegistration − LayerRegistration | -39.483 [-48.168,-30.180] | -18.491 [-23.528,-12.911] | -0.596 [-1.176,-0.099] | 11.794 [0.673,23.182] | 41.139 [37.966,44.565] | 0.087 [-0.008,0.203] | -5.550 [-23.902,16.240] | -0.693 [-1.592,0.116] |
| TokenRegistration − TokenOldControl | -1.124 [-1.711,-0.459] | 0.024 [-1.236,1.577] | -0.283 [-0.823,0.247] | 4.886 [-1.706,11.904] | -0.676 [-3.134,1.394] | 0.259 [0.013,0.560] | 3.086 [-3.647,10.452] | -1.274 [-1.703,-0.850] |

P Prepare는 Host Input 준비 경계다. 실제 GPU Kernel 시작 시각이 아니다.
C3 Prepare→Ready는 Chunk Packing, 다른 Request와 Host 작업도 포함한다.
Arrival→P Prepare 차이를 Credit만의 효과 또는 Attention만의 효과로 단정하지 않는다.
첫 Output 이후 Residency는 TTFT에 더하지 않았다.
Natural Output 수가 Arm별로 다를 수 있다. 같은 Output 길이의 Controlled 실험과 구분한다.
Credit, Compile과 실제 Batch가 다른 Best-versus-best 비교다. 순수한 Axis-only 인과 효과가 아니다.

## 실제 Batch와 Runtime

| Arm | Source Credit | Compile Family | D Cap | P / D-prefill / D-decode 최대 Request Batch |
|---|---|---|---|---|
| NativeStrong | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 3/2/4 |
| NativeRegistration | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 3/4/5 |
| NativeCompletionRegistration | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 3/3/5 |
| LayerRegistration | [4] | ['MATCHED_NO_COMPILE'] | ['None'] | 3/3/2 |
| TokenRegistration | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 3/5/4 |
| HybridRegistration | [4] | ['MATCHED_NO_COMPILE'] | ['None'] | 3/4/3 |
| TokenOldControl | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 3/5/4 |

위 Batch는 3개 Start 전체의 최대값이다. 모든 Step의 고정 Batch가 아니다.
C3 Source Credit0은 별도 Coordinator Outstanding 제한을 끈 값이다. 요청0개를 허용한다는 뜻이 아니다.
Worker B/S, KV Capacity와 Admission 제한은 그대로 적용한다. C0의 별도 Credit 설정과 구분한다.
이 후속 검사는 Discovery에서 동결한 Best-versus-best C3 설정을 유지한다.
Actual Batch1/1/1, 같은 Source Credit 또는 정확한 Iso-work를 강제한 Mechanism Control은 아니다.
같은 B/S만으로 Axis별 실제 Geometry가 같다고 주장하지 않는다.
Native의 D-prefill 열은 Last-prompt Query 재계산도 포함한다. Layer/Token Offload라고 부르지 않는다.
모든 Batch Histogram과 Policy Flag는 전체 JSON에 있다.

## 별도 Host 등록 Queue

| Arm | Deferred Requests / 1200 Requests | 평균 Pending Requests | 최대 Pending Requests | Start-mean Conditional Wait p95(ms) |
|---|---:|---:|---:|---:|
| NativeStrong | 0 | 0.000000 | 0 | N/A |
| NativeRegistration | 317 | 0.100304 | 5 | 587.985648 |
| NativeCompletionRegistration | 318 | 0.104544 | 5 | 590.098000 |
| LayerRegistration | 466 | 0.136718 | 4 | 586.964958 |
| TokenRegistration | 606 | 0.183746 | 6 | 671.377666 |
| HybridRegistration | 522 | 0.158065 | 4 | 641.422725 |
| TokenOldControl | 0 | 0.000000 | 0 | N/A |

Queue 값은 이 Host 등록 Queue만 측정한다.
정책을 사용하지 않은 Arm의 0은 vLLM Admission 또는 Remote-KV Queue가 비었다는 뜻이 아니다.
Conditional Wait p95는 실제로 미룬 Request만 사용한 Start별 p95의 평균이다.
이를 모든 Request 또는 모든 Start를 합친 p95로 바꾸어 읽지 않는다.

## 관측 범위

첫 Transition 진단은 첫 두 Worker Output Notification을 비교한다.
비동기 Output과 Co-delivery가 이 진단에 영향을 줄 수 있다.
미룬 Request는 별도 Host 등록 Queue에 남는다. 이를 빈 vLLM Queue에 숨기지 않는다.
Receive→Release 대기와 Queue 길이는 전체 JSON의 `D_host_registration_queue`에 기록했다.
이 대기는 앞선 Coordinator 또는 IPC 대기를 포함하지 않는다.
Primary ITL은 모든 실제 Client Interval을 유지한다.
Host Step과 Host Gap은 GPU-exclusive 시간 또는 Physical Idle이 아니다.
Natural Output 수와 실제 Continuous Batch는 Arm별로 다를 수 있다.
Normal Completion의 State와 Lifetime을 검사했다. Cancellation과 Remote Failure는 별도 미검증이다.
새 정책은 미래 EOS, Arrival 또는 Decode-safe Window를 예측하지 않는다.

- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/request_registration_deferral_frozen_v1.json)
- [전체 Metric, 같은 ID 경계와 첫 Transition](dense_ar_longbench400_2d_20261121/analysis/request_registration_deferral_results_v1.json)
