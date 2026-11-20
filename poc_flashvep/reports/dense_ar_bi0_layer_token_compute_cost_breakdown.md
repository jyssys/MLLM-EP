# BI0 Layer/Token 계산 비용

상태: `ISOLATED_KERNEL_MEASUREMENT`와`ISOLATED_FORWARD_MEASUREMENT`. Online TTFT 결과가 아니다.

## 실행 범위

Qwen3-8B Original BF16의 실제 Serving Layer와 Linear API를 사용했다. 실제 요청의4608/8064 Token Prefix를 진단 입력으로 사용했다. 두 길이는576의 배수다. 모든 Last1/2/4/6/8/12에서 `L*m=36*s`와16-Token 정렬을 정확히 만족했다. Actual Request Batch는1이다.

각 Grid를 독립 Server Start3회에서 측정했다. 표의 시간은 Start별 Clean CUDA Event p50의 평균이다. Event Enclosure는 Exclusive Kernel Time이 아니다. CI는 Start를 쌍으로 Resampling한 경험적95% CI다. 요청이나 Kernel 호출을 독립 Start로 세지 않았다. Start3개이므로 CI의 범위는 제한적이다.

## Projection-only GEMM

실제 QKV/O/Gate-up/Down Module을 사용했다. CUDA Graph Replay는 Layer별 Python 제출 Gap을 줄인다. 입력은 Shape Control용 Zero Tensor다. Attention, Norm, KV, Transfer, Queue와 Sampling은 제외한다.

| Input | Cut | Token Suffix | Layer p50 | Token p50 | Token/Layer | Paired95% CI |
|---|---:|---:|---:|---:|---:|---|
| 4608 | Last1 | 128 | 2.1873ms | 5.6001ms | 2.5602 | [2.5472,2.5778] |
| 4608 | Last2 | 256 | 4.3127ms | 6.2993ms | 1.4606 | [1.4605,1.4608] |
| 4608 | Last4 | 512 | 8.6268ms | 8.8902ms | 1.0305 | [1.0292,1.0316] |
| 4608 | Last8 | 1024 | 17.3376ms | 17.1375ms | 0.9885 | [0.9857,0.9922] |
| 8064 | Last1 | 224 | 3.7534ms | 6.4576ms | 1.7205 | [1.7163,1.7252] |
| 8064 | Last2 | 448 | 7.5365ms | 9.0750ms | 1.2041 | [1.2005,1.2074] |
| 8064 | Last4 | 896 | 15.1284ms | 17.2964ms | 1.1433 | [1.1397,1.1476] |
| 8064 | Last12 | 2688 | 45.6198ms | 50.2517ms | 1.1015 | [1.0975,1.1065] |

작은 Token M에서 Projection 비용 차이가 컸다. 큰 M에서는 차이가 줄었다. Input4608/Last8에서는 Token이 약1.15% 빨랐다. 모든 Cell을 설명하는 보편적 단일 M 임계값은 확인하지 않았다.

같은 GEMM Rows라도 D가 사용하는 Projection Parameter 집합은 다르다. Layer는 마지막m개 Layer의 Matrix를 사용한다. Token은36개 Layer의 Matrix를 사용한다. 구조적 Parameter Bytes의 비율은36/m이다. 작은 M의 비용에는 Matrix Working Set 차이가 포함될 수 있다. 실제 HBM Read와 Cache Miss는`NOT_MEASURED`다. 이 구조적 Byte 수를 측정 Bandwidth로 표현하지 않는다.

Input4608/Last1은3개 Start 전체의 Kernel Attribution Gate를 통과하지 못했다. 표의 Clean Event 시간은 유지한다. 그 Cell의 CUPTI 비용 비중은 Primary로 사용하지 않는다. 나머지 Cell별 Gate와 전체 Shape 목록은 Raw 분석에 있다.

## 전체 D Continuation Forward

이 실험은 BI0 Decoder Layer API의 Eager 경로다. CUDA Graph를 사용하지 않는다. 유효한 Prefix KV와 Layer Boundary Hidden을 준비했다. 모든 Replay의 Hidden은 같은 Geometry의 원래 Forward와 일치했다. Transport, Queue, LM-head와 Sampling은 제외했다.

고립 Forward는 기본 KV Block16을 사용했다. 후속 Online 통제 실험은 KV Block128을 사용한다. Attention Shape와 비용에 이 차이가 영향을 줄 수 있다. 따라서 고립 Forward 비율을 Online 비용 비율로 대체하지 않는다. Projection-only 실험에는 KV Access가 없다. 별도의 실제 요청 Serial 측정은 Online과 같은 Block128을 사용한다.

| Input | Cut | Layer p50 | Token p50 | Token/Layer | Paired95% CI |
|---|---:|---:|---:|---:|---|
| 4608 | Last1 | 3.1229ms | 14.6675ms | 4.6968 | [4.6006,4.8098] |
| 4608 | Last2 | 6.5043ms | 14.7975ms | 2.2750 | [2.2470,2.3017] |
| 4608 | Last4 | 13.0327ms | 15.9835ms | 1.2264 | [1.2176,1.2338] |
| 4608 | Last8 | 26.6835ms | 30.0303ms | 1.1254 | [1.1224,1.1292] |
| 4608 | Last12 | 40.1346ms | 43.2370ms | 1.0773 | [1.0748,1.0796] |
| 8064 | Last1 | 6.3154ms | 15.0872ms | 2.3889 | [2.3701,2.4029] |
| 8064 | Last2 | 12.2216ms | 17.1586ms | 1.4040 | [1.3969,1.4098] |
| 8064 | Last4 | 24.7994ms | 31.0940ms | 1.2538 | [1.2456,1.2604] |
| 8064 | Last8 | 50.1214ms | 59.0456ms | 1.1781 | [1.1704,1.1848] |
| 8064 | Last12 | 75.0744ms | 85.7265ms | 1.1419 | [1.1372,1.1460] |

이 Grid의 전체 Forward에서는 Token 우위로 역전되지 않았다. Projection-only의 작은 역전과 전체 Forward를 구분한다. 전체 Forward는 Attention과 비GEMM 실행도 포함한다.

Projection-only는 Zero 입력과 Graph Replay를 사용한다. 전체 Forward는 실제 State와 Eager Dataflow를 사용한다. Cache, 실행 경로와 Host 제출도 다를 수 있다. 두 시간의 단순 차이를 실제 Kernel 제출 비용 또는 Attention 시간으로 사용하지 않는다. 실제 Forward의 비용 비중은 같은 실행의 Observer-qualified CUPTI 범주만 사용한다.

## 원인과 제한

같은 GEMM Rows에서도 D Attention Pair 수는 다르다. Token/Layer 비율은 정수 오차를 제외하면 약`2-m/36`이다. 전체 P+D Attention Pair 수는 같다. 달라지는 것은 P/D 배치다. Pair 수의 비율을 GPU 시간 비율로 대체하지 않는다.

첫 독립 Start의 Input8064/Last4에서 Observer Gate가 통과했다. Layer의 Kernel Union은25.247ms였다. 확인한 GEMM은17.954ms, Attention은4.174ms였다. Token은29.596ms, GEMM17.976ms, Attention8.397ms였다. 이 Cell의 차이는 거대한 GEMM 효율 차이만으로 설명되지 않는다. 나머지 Start와 Cell의 범주는 원시 분석에서 따로 확인한다.

작은 Cut의 D 계산 절감이 전체 TTFT 절감을 보장하지 않는다. Last1은 P Layer 작업의1/36만 이전한다. Layer는 Hidden/Residual을 추가 전송한다. 이 Payload는 Input Token당16384Bytes다. 이는 KV4개 Layer에 해당한다. Last1의 유효 전송량은 Native36 대비39개 Layer 단위다. 작은 Cut의 고정 비용과 P Queue 효과를 Online에서 검증해야 한다.

## 근거

- [GEMM3-Start 요약](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_GEMM_API_summary.json)
- [전체 Forward3-Start 요약](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_isolated_axis_cost_summary.json)
- [구조적 작업량](dense_ar_bi0_shortanswer200_20261120/analysis/BI0_axis_structural_accounting.json)

Kernel Exclusive Time, Host Enclosure와 Online Serving 이득을 서로 대체하지 않는다. Natural200의16-Token 정렬 비교는 Aggregate Rows가 근사 일치한다. 위 두 진단 길이만 정확한 동일 GEMM Rows 비교다.
## 실제 원문20개의 Serial Service

`ISOLATED_SERIAL_SERVICE_MEASUREMENT`. 별도 Calibration20을 각 Action에서 직렬 실행했다. Actual Batch는1/1/1이다. KV Block128, C0, Async-on과 B8192/S1024를 사용했다. 다른 요청의 Queue는 없다. 각 Action은 독립 Start1회다. 요청20개를 Start20회로 계산하지 않는다.

| Cut | Layer P Prepare→Ready | Token P Prepare→Ready | Layer Ready→First Worker | Token Ready→First Worker | Layer/Token 두 구간 합 |
|---:|---:|---:|---:|---:|---:|
|1|495.09ms|489.36ms|82.46ms|102.91ms|0.9751|
|2|489.51ms|469.64ms|86.35ms|113.74ms|0.9871|
|4|453.98ms|435.46ms|113.46ms|135.61ms|0.9937|
|6|430.41ms|404.43ms|127.26ms|166.07ms|0.9775|
|8|397.80ms|370.24ms|144.43ms|197.34ms|0.9553|
|12|344.92ms|304.96ms|212.74ms|245.22ms|1.0136|

Native는 같은 두 구간에서505.43ms와66.79ms였다. 모든 값은 요청별 Host 경계의 평균이다. P IPC/초기 Admission과 First Worker→Client는 위 합에 포함하지 않았다. 따라서 위 합은 전체 Client TTFT가 아니다. Source Model v2는 누락을 막기 위해 P Submit→Ready를 따로 사용한다.

Layer의 D 경로는 모든 Cut에서 짧았다. 반대로 Token의 P 경로는 모든 Cut에서 짧았다. 이 경쟁 효과 때문에 고립 D Forward의 큰 속도비가 전체 First-output Path의 같은 속도비로 이어지지 않았다. Cut12의 합은 Token이 약1.36% 짧았다. 단일 Start의 작은 차이를 확정된 Crossover로 판단하지 않는다.

Input Length는 실제 원문의 다양한 길이다. Layer D는 전체 Input의 Suffix Layers를 B8192 Chunk로 계산한다. Token D는 짧은 Suffix Token의 Full36을 계산한다. 실제 Chunk 수, Hidden READ와 KV Descriptor는 `analysis/BI0_axis_serial_component_comparison_v1.json`에 보존한다. 이 경계로 GEMM·Attention·Transfer의 Exclusive GPU 비중을 계산하지 않는다.

## Host Cost Model의 독립 Prompt Validation

Model v2는 Calibration20으로 Fit했다. Validation15는 다른 Prompt다. 각 Action/Split은 독립 Start1회다. 아래 Coverage는 Calibration의 양의 Residual p95를 더한 예측의 Validation 포함률이다. 안전한 WCET 또는 p99 보장이 아니다.

| Action/경계 | MAE | Absolute Error p95 | Absolute Error p99 | Underprediction | Upper Coverage |
|---|---:|---:|---:|---:|---:|
|Native P Submit→Ready|6.900ms|19.761ms|23.059ms|6/15|12/15|
|Native Ready→First Worker|24.633ms|50.430ms|61.006ms|3/15|14/15|
|Layer8 P Submit→Ready|6.307ms|14.604ms|15.301ms|10/15|13/15|
|Layer8 Ready→First Worker|18.758ms|49.442ms|67.961ms|15/15|11/15|
|Token8 P Submit→Ready|3.964ms|10.516ms|12.670ms|7/15|15/15|
|Token8 Ready→First Worker|23.734ms|46.358ms|69.488ms|2/15|14/15|

특히 Layer8 D 경로는15/15개를 낮게 예측했다. 이 Model로 안전한 Window Admission을 실행하지 않았다. 작은 표본의 p99를 단단한 Tail Bound로 표현하지 않는다. 전체13개 Action의 결과는 `analysis/BI0_serial_service_model_v2.json`에 있다.
