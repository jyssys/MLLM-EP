# Whole-request Deflection — 독립 N3 C3 Serving

실제 Online E와 Continuous P/D를 사용했다. Main100과 새 Context Holdout50을 별도로 평가했다.
Natural EOS와 동일 Output 수의 Replay를 분리했다. 각 Method의 독립 초기화는 3회다.
전체 Task Quality는 PENDING이다. 아래 Runtime Threshold 판정을 Full GO로 바꾸지 않는다.

## 모든 Start 결과

| Subset | Output | Method | Start | Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main100 | natural | NativeOnDemand | 0 | 5198 | 2408.013 | 9571.735 | 14.910 | 6.722 | 52.224 | 56.000 |
| Main100 | replay | NativeOnDemand | 0 | 5269 | 2224.157 | 9166.969 | 15.926 | 6.915 | 53.600 | 58.000 |
| FreshHoldout | natural | NativeOnDemand | 0 | 3090 | 1420.199 | 2361.488 | 17.319 | 6.938 | 102.070 | 92.000 |
| FreshHoldout | replay | NativeOnDemand | 0 | 3074 | 1263.015 | 2215.779 | 18.508 | 6.898 | 133.105 | 100.000 |
| Main100 | natural | WholeDeflection | 0 | 5312 | 2293.274 | 8612.986 | 20.428 | 7.571 | 56.820 | 60.000 |
| Main100 | replay | WholeDeflection | 0 | 5269 | 2329.081 | 9646.187 | 17.945 | 7.276 | 56.398 | 60.000 |
| FreshHoldout | natural | WholeDeflection | 0 | 3028 | 1261.079 | 2342.107 | 19.254 | 7.230 | 128.039 | 96.000 |
| FreshHoldout | replay | WholeDeflection | 0 | 3074 | 1324.722 | 2186.762 | 18.965 | 9.565 | 121.680 | 90.000 |
| Main100 | natural | NativeStrong | 0 | 5288 | 2191.776 | 4740.674 | 14.779 | 6.945 | 82.914 | 92.000 |
| Main100 | replay | NativeStrong | 0 | 5269 | 2322.746 | 5583.600 | 14.952 | 6.721 | 56.748 | 80.000 |
| FreshHoldout | natural | NativeStrong | 0 | 3068 | 1176.863 | 2038.720 | 16.725 | 6.890 | 134.175 | 100.000 |
| FreshHoldout | replay | NativeStrong | 0 | 3074 | 1193.772 | 2031.559 | 17.398 | 6.949 | 112.765 | 98.000 |
| Main100 | natural | WholeDeflection | 1 | 5284 | 2195.469 | 9570.622 | 18.032 | 7.683 | 55.252 | 60.000 |
| Main100 | replay | WholeDeflection | 1 | 5269 | 2274.116 | 7492.109 | 17.572 | 7.491 | 59.854 | 66.000 |
| FreshHoldout | natural | WholeDeflection | 1 | 3043 | 1227.946 | 2421.187 | 20.678 | 7.910 | 99.935 | 92.000 |
| FreshHoldout | replay | WholeDeflection | 1 | 3074 | 1139.204 | 2455.432 | 17.893 | 6.889 | 112.153 | 96.000 |
| Main100 | natural | NativeStrong | 1 | 5303 | 2209.520 | 4880.967 | 15.437 | 6.870 | 92.669 | 97.000 |
| Main100 | replay | NativeStrong | 1 | 5269 | 2254.457 | 5061.351 | 14.713 | 6.844 | 92.057 | 93.000 |
| FreshHoldout | natural | NativeStrong | 1 | 3040 | 1190.296 | 2044.163 | 15.606 | 6.861 | 131.739 | 100.000 |
| FreshHoldout | replay | NativeStrong | 1 | 3074 | 1228.718 | 2070.605 | 16.360 | 6.950 | 115.978 | 96.000 |
| Main100 | natural | NativeOnDemand | 1 | 5339 | 2352.776 | 10241.290 | 15.971 | 6.874 | 52.279 | 56.000 |
| Main100 | replay | NativeOnDemand | 1 | 5269 | 2394.654 | 11635.603 | 16.529 | 6.862 | 49.246 | 54.000 |
| FreshHoldout | natural | NativeOnDemand | 1 | 3113 | 1370.833 | 2425.866 | 16.790 | 6.853 | 133.373 | 100.000 |
| FreshHoldout | replay | NativeOnDemand | 1 | 3074 | 1330.203 | 2227.806 | 18.606 | 6.777 | 133.333 | 100.000 |
| Main100 | natural | NativeStrong | 2 | 5035 | 2264.289 | 4436.345 | 15.103 | 6.840 | 90.088 | 95.000 |
| Main100 | replay | NativeStrong | 2 | 5269 | 2246.705 | 5265.188 | 15.089 | 6.853 | 90.330 | 91.000 |
| FreshHoldout | natural | NativeStrong | 2 | 3006 | 1212.690 | 2069.134 | 17.251 | 6.868 | 131.182 | 100.000 |
| FreshHoldout | replay | NativeStrong | 2 | 3074 | 1178.762 | 2039.787 | 17.113 | 7.262 | 133.653 | 100.000 |
| Main100 | natural | NativeOnDemand | 2 | 5275 | 2225.168 | 9407.653 | 15.622 | 6.930 | 53.258 | 56.000 |
| Main100 | replay | NativeOnDemand | 2 | 5269 | 2245.093 | 8601.208 | 14.980 | 6.851 | 53.942 | 58.000 |
| FreshHoldout | natural | NativeOnDemand | 2 | 3026 | 1303.360 | 2367.822 | 18.781 | 6.828 | 130.474 | 100.000 |
| FreshHoldout | replay | NativeOnDemand | 2 | 3074 | 1372.363 | 2386.205 | 19.617 | 6.836 | 109.622 | 94.000 |
| Main100 | natural | WholeDeflection | 2 | 5198 | 2181.255 | 9121.430 | 19.380 | 8.212 | 51.536 | 57.000 |
| Main100 | replay | WholeDeflection | 2 | 5269 | 1986.909 | 7724.790 | 18.503 | 7.251 | 55.943 | 63.000 |
| FreshHoldout | natural | WholeDeflection | 2 | 3095 | 1181.081 | 2261.474 | 18.080 | 6.917 | 132.602 | 98.000 |
| FreshHoldout | replay | WholeDeflection | 2 | 3074 | 1239.082 | 2352.520 | 17.315 | 6.955 | 131.985 | 100.000 |

## 고정 Gate 판정

| Subset | Output | Relative Decode | Absolute Decode | TTFT p50 | Runtime Decision |
|---|---|---|---|---|---|
| Main100 | natural | False | False | True | NO_GO_ON_TESTED_RUNTIME |
| Main100 | replay | False | False | True | NO_GO_ON_TESTED_RUNTIME |
| FreshHoldout | natural | False | False | False | NO_GO_ON_TESTED_RUNTIME |
| FreshHoldout | replay | False | False | False | NO_GO_ON_TESTED_RUNTIME |

NativeStrong은 원래 Inline E Transport와 Stock Compiler를 유지했다.
NativeOnDemand는 후보와 같은 On-demand E Transport·Coordinator·공통 Compiler를 사용하되 Deflection하지 않았다.
두 Native를 같은 실행 정책이라고 부르지 않는다. 후보는 두 기준에 대해 각각 Decode와 TTFT p50 Gate를 검사했다.
선택은 실제 E 완료 후 현재 P HTTP 작업과 D Stream 상태를 사용했다. 미래 EOS나 Idle Duration은 사용하지 않았다.
D Full Prefill은 이전 D Stream이 없을 때만 시작했다. 새 Native D는 첫 실제 생성 Token 이후에 진입할 수 있다.
이 상태 보호만으로 모든 후속 Decode의 비열등성을 가정하지 않았다. 실제 ITL/TPOT와 모든-ITL 실패를 검사했다.
필수 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token이다.
Absolute Decode는 Request별 All-ITL과 TPOT의 AND다. 모든 Start에서 95% 이상이어야 한다.
Paired CI는 같은 Block의 독립 Start3쌍을 사용했다. 요청을 독립 Start로 취급하지 않았다.
모든 27개 Empirical Resample을 사용했다. N3 CI 해상도는 제한적이다.
다른 주요 지표의 Point 감소도 Audit에 남겼다. TTFT 개선으로 Decode나 Joint SLO 악화를 상쇄하지 않는다.

## 경로·품질·적용 한계

Deflected 요청은 P Forward와 P Prefix KV Import가 없다. 해당 P 시간은 NOT_APPLICABLE로 남겼다.
다섯 공통 Host 구간은 TTFT와 정확히 합산된다. 세부 P 구간은 공통 구간과 겹치므로 다시 더하지 않는다.
Host Enclosure를 GPU-exclusive Forward나 DMA 시간으로 해석하지 않는다.
Whole-request의 성공이나 실패는 Generic Token Split의 결과가 아니다. Kairos 재현이나 새로운 Modality-aware Method도 아니다.
조건부 Same-E N1+N8에서 28개 KV Layer와 전체 Greedy 출력을 검사했다. C3의 다른 Batch에서 같은 출력을 보장하지는 않는다.
MMLongBench-Doc 공식 외부 Answer Extraction은 미실행이다. Raw Task Score를 전체 품질 PASS로 바꾸지 않았다.
FreshHoldout은 Parent400 중 Main100·Calibration 이미지/문서와 겹치지 않는 새 50개다.
원래 Single-GPU Natural 길이 Profiling은 존재한다. 이번 C3 방법 비교에는 새 Context다.
Main100이나 이전 Holdout을 성능 결과에 따라 교체하지 않았다.
실제 Clean GPU Batch는 미측정이다. HTTP Outstanding 또는 B/S를 Actual GPU Batch로 대체하지 않았다.

## 재현 자료

Original BF16 Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.
GPU4/5/6, B8192/S1024, KV42GiB, Block128, Prefix Cache Off, 원래 Vision/Processor를 유지했다.
Config는 `configs/whole_DEFLECTION_C43R4.json`이다. Dataset Hash·실행 순서·Arrival은 결과 전에 동결했다.
CSV와 Audit은 `analysis/whole_DEFLECTION_C43R4_*_v0`에 있다. Figure와 대응 CSV는 `figures/whole_DEFLECTION_C43R4_*_v0`에 있다.
모든 원시 실패와 세부 Request Token 수를 보존했다. Finite-cohort Throughput을 Sustainable Capacity로 부르지 않는다.
