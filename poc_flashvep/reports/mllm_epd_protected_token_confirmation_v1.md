# Protected Token Split — 독립 N3 Runtime 확인

실제 Online E/P/D의 Natural EOS와 Equal-output Replay를 별도로 재계산했다.
각 Method는 GPU4/5/6에서 독립 서버 초기화 3회로 실행했다. 요청을 독립 Start로 취급하지 않았다.
전체 Task Quality는 PENDING이다. 아래 Runtime 결과만으로 최종 GO를 선언하지 않는다.

## 계약

Qwen2.5-VL-7B, Original BF16, BI0, 원본 Vision/Processor, Full28을 유지했다.
Revision cc594898137f460bfe9f0759e9844b3ce807cfb5, B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
Main100 Manifest SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1다.
NativeStrong은 Stock Compiled다. NativeCommon과 후보는 공통 Deterministic Reduction 설정을 사용했다.
두 Native는 원래 NIXL과 기본 Admission을 유지했다. Token Adapter나 인위적인 D Cap을 추가하지 않았다.
후보는 기존 D HTTP Stream이 없을 때만 추가 Suffix를 시작했다.
이 정책은 미래 Decode-safe Window 또는 Dynamic Split-size Predictor가 아니다.
Main100은 Discovery50과 Heldout50을 포함한다. Heldout50은 별도의 고정 Arrival로도 검사했다.
두 Split의 Source 분포는 불균형하다. Main100 전체 대표성이 각 Split의 대표성을 뜻하지 않는다.
모든 비율, 순서, Arrival Seed와 SLO는 Heldout 결과 전에 고정했다.
측정 중 GPU Profiler와 CPU Tensor Capture는 껐다. 실제 Clean GPU Batch는 NOT_MEASURED다.
같은 B/S나 Continuous Scheduler 이름을 같은 Actual Batch로 해석하지 않았다.

## Start별 모든 결과

| Subset | Output | Method | Start | Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Main100 | natural | NativeStrong | 0 | 5189 | 2321.851 | 6241.897 | 15.560 | 7.067 | 66.620 | 76.000 |
| Main100 | replay | NativeStrong | 0 | 5269 | 2186.758 | 4680.588 | 15.150 | 6.884 | 89.238 | 92.000 |
| Heldout50 | natural | NativeStrong | 0 | 3783 | 1604.423 | 2621.345 | 15.458 | 6.811 | 110.639 | 96.000 |
| Heldout50 | replay | NativeStrong | 0 | 3568 | 1692.248 | 2735.072 | 15.499 | 6.776 | 132.629 | 100.000 |
| Main100 | natural | Static10 | 0 | 5399 | 2764.599 | 8883.382 | 8.684 | 6.382 | 65.241 | 66.000 |
| Main100 | replay | Static10 | 0 | 5269 | 2655.166 | 8152.390 | 7.933 | 6.385 | 61.295 | 68.000 |
| Heldout50 | natural | Static10 | 0 | 3551 | 3376.567 | 7425.550 | 8.636 | 6.414 | 83.890 | 58.000 |
| Heldout50 | replay | Static10 | 0 | 3568 | 3946.307 | 7883.975 | 7.918 | 6.411 | 83.528 | 58.000 |
| Main100 | natural | Idle10 | 0 | 4970 | 2130.771 | 7103.351 | 10.706 | 6.539 | 67.378 | 79.000 |
| Main100 | replay | Idle10 | 0 | 5269 | 2138.576 | 4627.452 | 15.855 | 6.825 | 95.305 | 97.000 |
| Heldout50 | natural | Idle10 | 0 | 5514 | 1727.304 | 2902.050 | 13.302 | 6.779 | 140.873 | 96.000 |
| Heldout50 | replay | Idle10 | 0 | 3568 | 1655.620 | 3762.315 | 15.144 | 6.728 | 124.190 | 98.000 |
| Main100 | natural | NativeCommon | 0 | 5247 | 2155.969 | 4632.451 | 16.260 | 6.820 | 89.073 | 93.000 |
| Main100 | replay | NativeCommon | 0 | 5269 | 2298.804 | 5060.788 | 15.859 | 6.746 | 88.613 | 90.000 |
| Heldout50 | natural | NativeCommon | 0 | 3597 | 1584.854 | 2650.256 | 15.911 | 6.998 | 127.012 | 96.000 |
| Heldout50 | replay | NativeCommon | 0 | 3568 | 1724.081 | 2693.162 | 15.349 | 6.801 | 124.893 | 96.000 |
| Main100 | natural | Idle10 | 1 | 5182 | 2233.207 | 5368.029 | 15.910 | 6.798 | 80.882 | 92.000 |
| Main100 | replay | Idle10 | 1 | 5269 | 2209.036 | 5353.052 | 13.036 | 6.620 | 92.458 | 93.000 |
| Heldout50 | natural | Idle10 | 1 | 3578 | 1683.770 | 3856.777 | 15.819 | 6.773 | 124.033 | 98.000 |
| Heldout50 | replay | Idle10 | 1 | 3568 | 1748.074 | 2785.182 | 16.005 | 6.997 | 130.502 | 100.000 |
| Main100 | natural | Static10 | 1 | 5399 | 2670.581 | 8118.941 | 8.507 | 6.380 | 63.068 | 65.000 |
| Main100 | replay | Static10 | 1 | 5269 | 2544.046 | 7913.730 | 8.307 | 6.360 | 62.653 | 67.000 |
| Heldout50 | natural | Static10 | 1 | 3557 | 3779.778 | 7794.977 | 8.787 | 6.399 | 83.065 | 58.000 |
| Heldout50 | replay | Static10 | 1 | 3568 | 3815.942 | 7804.429 | 8.124 | 6.411 | 83.844 | 58.000 |
| Main100 | natural | NativeStrong | 1 | 5289 | 2407.120 | 4629.608 | 15.835 | 6.905 | 84.949 | 94.000 |
| Main100 | replay | NativeStrong | 1 | 5269 | 2336.499 | 5339.019 | 15.583 | 6.844 | 77.938 | 86.000 |
| Heldout50 | natural | NativeStrong | 1 | 3555 | 1617.506 | 2694.539 | 15.146 | 7.054 | 133.804 | 100.000 |
| Heldout50 | replay | NativeStrong | 1 | 3568 | 1675.038 | 2715.181 | 15.761 | 6.798 | 131.760 | 100.000 |
| Main100 | natural | NativeCommon | 1 | 5214 | 2360.971 | 4688.847 | 16.044 | 6.917 | 86.153 | 90.000 |
| Main100 | replay | NativeCommon | 1 | 5269 | 2100.524 | 4919.953 | 15.030 | 6.904 | 92.424 | 92.000 |
| Heldout50 | natural | NativeCommon | 1 | 3630 | 1617.782 | 2671.548 | 17.707 | 6.947 | 119.319 | 96.000 |
| Heldout50 | replay | NativeCommon | 1 | 3568 | 1599.713 | 2612.670 | 15.825 | 6.757 | 132.926 | 100.000 |
| Main100 | natural | Idle10 | 2 | 5093 | 2241.683 | 6620.494 | 14.556 | 6.594 | 76.923 | 84.000 |
| Main100 | replay | Idle10 | 2 | 5269 | 2324.562 | 5170.599 | 13.678 | 6.663 | 90.621 | 91.000 |
| Heldout50 | natural | Idle10 | 2 | 3685 | 1597.287 | 2783.862 | 15.124 | 7.433 | 127.299 | 96.000 |
| Heldout50 | replay | Idle10 | 2 | 3568 | 1748.107 | 2751.054 | 15.596 | 6.774 | 102.908 | 96.000 |
| Main100 | natural | NativeStrong | 2 | 5256 | 1997.969 | 4631.274 | 15.719 | 6.799 | 83.988 | 92.000 |
| Main100 | replay | NativeStrong | 2 | 5269 | 2382.545 | 5539.493 | 14.473 | 6.800 | 83.742 | 87.000 |
| Heldout50 | natural | NativeStrong | 2 | 3655 | 1718.594 | 2746.495 | 16.269 | 6.777 | 104.674 | 94.000 |
| Heldout50 | replay | NativeStrong | 2 | 3568 | 1588.508 | 2679.153 | 15.532 | 6.828 | 133.006 | 100.000 |
| Main100 | natural | Static10 | 2 | 5399 | 2594.444 | 7012.053 | 8.298 | 6.419 | 69.863 | 76.000 |
| Main100 | replay | Static10 | 2 | 5269 | 2465.024 | 7604.147 | 8.021 | 6.395 | 63.619 | 69.000 |
| Heldout50 | natural | Static10 | 2 | 3551 | 3273.676 | 7309.196 | 8.736 | 6.416 | 78.716 | 58.000 |
| Heldout50 | replay | Static10 | 2 | 3568 | 3513.075 | 7508.033 | 7.867 | 6.417 | 84.416 | 58.000 |
| Main100 | natural | NativeCommon | 2 | 5183 | 2209.211 | 4840.101 | 15.734 | 7.002 | 88.132 | 92.000 |
| Main100 | replay | NativeCommon | 2 | 5269 | 2215.663 | 4446.265 | 15.150 | 6.809 | 87.392 | 93.000 |
| Heldout50 | natural | NativeCommon | 2 | 3608 | 1660.118 | 2717.413 | 13.938 | 6.918 | 113.471 | 94.000 |
| Heldout50 | replay | NativeCommon | 2 | 3568 | 1600.144 | 2689.994 | 15.087 | 6.901 | 132.858 | 100.000 |

## Runtime Gate 판정

| Subset | Output | Method | Relative Decode | Absolute Decode | TTFT p50 | Runtime Decision |
|---|---|---|---|---|---|---|
| Main100 | natural | Static10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Main100 | replay | Static10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Heldout50 | natural | Static10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Heldout50 | replay | Static10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Main100 | natural | Idle10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Main100 | replay | Idle10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |
| Heldout50 | natural | Idle10 | False | True | False | NO_GO_ON_TESTED_RUNTIME |
| Heldout50 | replay | Idle10 | True | True | False | NO_GO_ON_TESTED_RUNTIME |

Runtime의 개선 조건은 TTFT p95 최소5% 감소 또는 Goodput 최소5% 증가와 우위 방향의 Paired CI다.
두 Native 기준, 상대 Decode, TTFT p50 비열등성 및 Absolute Decode를 각각 검사했다.
품질 PENDING을 Runtime 실패의 대체 설명으로 사용하지 않았다. Runtime 실패와 품질 미확정을 분리했다.
CDF는 같은 크기의 독립 Start 요청을 합친 기술통계다. CI의 독립 단위는 여전히 Start다.

## Paired CI와 한계

각 Paired Ratio는 Start-mean의 비율이다. 같은 Block의 Baseline과 후보를 함께 Resampling했다.
N3의 가능한 27개 Empirical Bootstrap 조합을 모두 사용했다. 95% CI의 해상도는 제한적이다.
ITL p99와 TPOT p95의 상대 1.05 Gate는 NativeStrong 및 NativeCommon에 각각 적용해야 한다.
Absolute Gate는 모든 Client ITL≤34ms 및 Request TPOT≤33ms/token이다.
TTFT≤5000ms와 두 Decode Gate의 AND가 Joint다. 모든 실패 8개 조합의 요청 수와 Token 수를 저장했다.
Goodput 분모는 첫 예정 Arrival부터 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.
Natural Output 수는 실제 생성값이다. Replay는 요청별 Token 수만 같고 Answer String은 같다고 보장하지 않는다.
Task Quality의 공식 외부 답변 추출은 승인 없이 실행하지 않았다. Raw Task Score를 완전한 품질 Gate로 대체하지 않았다.
25% 후보는 이전 N8 KV 실패로 제외했다. 해당 실패를 삭제하거나 기준을 완화하지 않았다.
Raw Metric, Arrival Contract와 SLO Goodput 분자를 재계산했다. 원래 저장값과 일치했다.
Host Stage 합계도 요청 TTFT와 일치했다. HTTP Enclosure는 GPU-exclusive Compute가 아니다.
Derived 파일은 `analysis/token_confirmation_C37_*_v0`에 있다.
Figure와 대응 CSV는 `figures/token_confirmation_C37_*_v0`에 있다.
본 보고서는 최종 Motivation, Prefill-heavy 및 Encode-safe Headroom 판정을 대신하지 않는다.
