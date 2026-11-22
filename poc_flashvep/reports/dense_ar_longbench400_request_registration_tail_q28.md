# LongBench400: D Request 등록과 첫 Decode Tail — QPS2.8

Evidence: `MEASURED_ONLINE_ADAPTIVE_REPAIR` 및 `TRACE_ANALYSIS`.
원래 Main140과 READ Repair는 유지했다. 이 문서는 별도 후속 검사다.
먼저 관측한 Heldout200 Host Gap을 사용해 가설을 세웠다.
새 성능 측정 전에 3개 새 Seed와 7개 대조군을 동결했다.
Confirmation Seed는 [2026112781, 2026112782, 2026112783]다. 모두 GPU Pair [4, 5]를 사용했다.
Natural400, QPS2.8, BI0, BF16, B8192/S1024와 KV42GiB/Block128을 유지했다.
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
| NativeStrong | 26484.191104 | 16.938246 | 8.475937 | 61.686933 | 9.755980 | 0.177500 | [11003, 11029, 11036] |
| NativeRegistration | 26649.548378 | 8.925276 | 8.004097 | 61.867057 | 10.172318 | 0.175833 | [11074, 11047, 11071] |
| NativeCap2Registration | 26709.308993 | 8.779906 | 7.957054 | 61.677295 | 10.083014 | 0.175000 | [10974, 11047, 11084] |
| LayerRegistration | 6583.601082 | 8.948487 | 8.239755 | 70.345654 | 52.488352 | 0.761667 | [10993, 10996, 10974] |
| TokenRegistration | 4390.895747 | 8.947635 | 8.050413 | 71.526013 | 67.709592 | 0.959167 | [11150, 11027, 11077] |
| HybridRegistration | 5902.597826 | 8.943290 | 8.231843 | 70.387073 | 57.243633 | 0.825833 | [10978, 11006, 11001] |
| TokenOldControl | 4298.469925 | 9.461037 | 8.365078 | 71.443487 | 67.355281 | 0.958333 | [11144, 11011, 11066] |

Joint SLO는 TTFT와 Request별 Decode 조건을 함께 만족한 비율이다.
표는 Start별 비율의 평균이다. 평균 Joint≥95%가 모든 Start의 Joint≥95%를 보장하지 않는다.
Decode 보호와 Joint SLO를 구분한다. Raw Throughput과 SLO-goodput도 다른 Metric이다.
처리량은 유한400개 Arrival부터 Drain까지 측정했다. Sustainable Capacity가 아니다.

### 관측한 가장 큰 Client Interval

| Arm | 각 Start의 Max ITL(ms) | 관측 전체 Max ITL(ms) |
|---|---|---:|
| NativeStrong | [272.036476, 165.378178, 188.06212] | 272.036476 |
| NativeRegistration | [51.07568, 69.587188, 69.456652] | 69.587188 |
| NativeCap2Registration | [59.175048, 64.816136, 64.140113] | 64.816136 |
| LayerRegistration | [150.8648, 241.807004, 56.045877] | 241.807004 |
| TokenRegistration | [146.96406, 68.180932, 12.781827] | 146.964060 |
| HybridRegistration | [55.749763, 72.35596, 238.404531] | 238.404531 |
| TokenOldControl | [64.627277, 356.622558, 66.976432] | 356.622558 |

이는 관측 Max다. 미래 지연의 최악 상한이 아니다.
p99 또는 TPOT 비열등성 통과가 모든 Interval의 동일성이나 34ms 이하를 보장하지 않는다.
Request별 모든 ITL과 TPOT를 사용하는 Absolute Decode Attainment도 별도로 검사한다.

## Paired CI와 Decode 보호

| Arm / Reference | TTFT 비율 [95% CI] | ITL 비율 [95% CI] | TPOT 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |
|---|---|---|---|---|---|---|
| TokenRegistration / TokenOldControl | 1.021502 [0.997176,1.054691] | 0.945735 [0.913529,0.982325] | 0.962383 [0.956918,0.967238] | 1.005260 [1.002747,1.007590] | True | True |
| TokenRegistration / NativeRegistration | 0.164764 [0.145949,0.183171] | 1.002505 [0.980183,1.031733] | 1.005786 [1.001359,1.011719] | 6.656260 [5.782309,7.963148] | True | True |
| LayerRegistration / NativeStrong | 0.248586 [0.227626,0.266291] | 0.528301 [0.499539,0.563440] | 0.972135 [0.966736,0.978530] | 5.380121 [4.381489,6.502833] | True | True |
| LayerRegistration / NativeCap2Registration | 0.246491 [0.223662,0.265537] | 1.019201 [1.013256,1.026533] | 1.035528 [1.034601,1.036656] | 5.205621 [4.204173,6.595154] | True | True |
| TokenRegistration / NativeStrong | 0.165793 [0.147263,0.184022] | 0.528250 [0.501144,0.554803] | 0.949796 [0.947474,0.952758] | 6.940317 [6.254772,7.882108] | True | True |
| HybridRegistration / NativeStrong | 0.222872 [0.201748,0.246342] | 0.527994 [0.501218,0.560856] | 0.971202 [0.965329,0.977989] | 5.867543 [4.692357,7.184772] | True | True |
| HybridRegistration / NativeRegistration | 0.221490 [0.199945,0.245166] | 1.002018 [0.992665,1.009701] | 1.028454 [1.021074,1.035706] | 5.627393 [4.441419,7.163407] | True | True |
| HybridRegistration / LayerRegistration | 0.896561 [0.837692,0.941169] | 0.999419 [0.995427,1.003367] | 0.999040 [0.998545,0.999447] | 1.090597 [1.070608,1.104739] | True | True |
| HybridRegistration / TokenRegistration | 1.344281 [1.312721,1.378300] | 0.999514 [0.975974,1.020588] | 1.022537 [1.018393,1.026654] | 0.845429 [0.698429,0.953074] | True | True |
| LayerRegistration / TokenRegistration | 1.499375 [1.437462,1.573933] | 1.000095 [0.975367,1.024203] | 1.023520 [1.019877,1.027222] | 0.775198 [0.649467,0.870114] | True | True |

CI는 Paired Server Start 단위다. N3에서 27개 Bootstrap 조합을 완전 열거했다.
Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.
각 Arm은 NativeStrong과 같은 D 정책의 Native를 모두 통과해야 한다.
공통 Native의 정확한 Mapping은 다음과 같다: {'LayerRegistration': 'NativeCap2Registration', 'TokenRegistration': 'NativeRegistration', 'HybridRegistration': 'NativeRegistration'}.
실패한 Gate를 TTFT 개선으로 대체하지 않는다.

## 같은 Request의 Host 경계 차이

아래 값은 왼쪽 Arm에서 Reference를 뺀 Start별 평균의 Paired CI다.
단위는 ms다. 같은 Request ID, Prompt, Source, Pair와 Arrival을 비교한다.
여섯 구간의 평균 차이는 Mean TTFT 차이에 합산된다. 구간별 CI는 합산하지 않는다.
이는 TTFT p95 또는 GPU-exclusive Operator 비용의 분해가 아니다.

| Arm − Reference | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 | 첫 Output 이후 Residency 차이 |
|---|---|---|---|---|---|---|---|---|
| LayerRegistration − TokenRegistration | 1278.653 [1054.330,1558.571] | 44.002 [29.302,57.025] | 0.446 [0.067,0.943] | 75.529 [32.792,117.093] | -9.694 [-27.396,8.111] | 0.478 [0.023,1.072] | 1389.413 [1224.063,1598.965] | 2.636 [1.411,3.690] |
| HybridRegistration − TokenRegistration | 703.550 [491.480,986.578] | 36.671 [22.785,51.271] | 0.105 [-0.093,0.341] | 77.163 [54.917,101.945] | 119.725 [113.644,124.678] | -0.048 [-0.092,-0.008] | 937.166 [738.430,1195.444] | 2.601 [1.104,3.741] |
| HybridRegistration − LayerRegistration | -575.103 [-605.998,-549.848] | -7.331 [-12.050,-3.638] | -0.340 [-0.968,0.248] | 1.634 [-15.148,22.125] | 129.419 [113.993,147.772] | -0.526 [-1.096,-0.075] | -452.247 [-513.890,-392.716] | -0.035 [-0.431,0.375] |
| TokenRegistration − TokenOldControl | 4.841 [2.220,8.136] | -3.284 [-7.608,0.637] | -0.156 [-0.339,-0.018] | 29.510 [-3.546,62.127] | 7.723 [1.593,15.004] | -0.084 [-0.188,-0.007] | 38.550 [1.373,76.944] | -1.050 [-1.224,-0.867] |

P Prepare는 Host Input 준비 경계다. 실제 GPU Kernel 시작 시각이 아니다.
C3 Prepare→Ready는 Chunk Packing, 다른 Request와 Host 작업도 포함한다.
Arrival→P Prepare 차이를 Credit만의 효과 또는 Attention만의 효과로 단정하지 않는다.
첫 Output 이후 Residency는 TTFT에 더하지 않았다.
Natural Output 수가 Arm별로 다를 수 있다. 같은 Output 길이의 Controlled 실험과 구분한다.
Credit, Compile과 실제 Batch가 다른 Best-versus-best 비교다. 순수한 Axis-only 인과 효과가 아니다.

## 실제 Batch와 Runtime

| Arm | Source Credit | Compile Family | D Cap | P / D-prefill / D-decode 최대 Request Batch |
|---|---|---|---|---|
| NativeStrong | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 4/3/5 |
| NativeRegistration | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 4/4/5 |
| NativeCap2Registration | [0] | ['INSTALLED_DEFAULT_COMPILE'] | ['2'] | 4/2/2 |
| LayerRegistration | [8] | ['MATCHED_NO_COMPILE'] | ['2'] | 3/2/2 |
| TokenRegistration | [8] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 4/5/5 |
| HybridRegistration | [8] | ['MATCHED_NO_COMPILE'] | ['None'] | 4/4/2 |
| TokenOldControl | [8] | ['INSTALLED_DEFAULT_COMPILE'] | ['None'] | 4/5/5 |

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
| NativeRegistration | 416 | 0.164918 | 5 | 573.152435 |
| NativeCap2Registration | 412 | 0.156272 | 5 | 542.265612 |
| LayerRegistration | 686 | 0.314197 | 6 | 588.128516 |
| TokenRegistration | 830 | 0.403868 | 7 | 643.914741 |
| HybridRegistration | 722 | 0.339155 | 5 | 619.854293 |
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

- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/request_registration_q28_extension_frozen_v1.json)
- [전체 Metric, 같은 ID 경계와 첫 Transition](dense_ar_longbench400_2d_20261121/analysis/request_registration_deferral_q28_results_v1.json)
