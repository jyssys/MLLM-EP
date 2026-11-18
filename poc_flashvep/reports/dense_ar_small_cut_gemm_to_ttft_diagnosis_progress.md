# Small Layer Cut의 GEMM 이득과 TTFT 차이

## 범위와 현재 판정

이 문서는 진행 중인 배치 일치 실험의 원인 분석이다.
최종 Layer/Token 우위나 Crossover 판정이 아니다.
기존 결과와 원시 Evidence는 변경하지 않았다.
새 Dynamic Policy도 구현하지 않았다.

작은 Cut의 D Forward 이득은 실제로 관측됐다.
그러나 그 이득이 같은 비율의 전체 TTFT 개선을 뜻하지는 않는다.
현재 근거만으로 Attention과 Hidden 전송을 항상 무시할 수 있다고 판단하지 않는다.
입력 길이가 증가하면 GEMM 효율도 계속 증가한다고 가정하지 않는다.

## 1. 실제 실행 구간과 전체 TTFT

다음은 기존 Batch1 직렬 진단의 실제 측정이다.
Workload는 Input8000/Output16이다.
요청 간격은 2초이다.
각 방식은 독립 Process Start 1회이다.
18개 요청 중 처음 9개를 준비 실행으로 사용했다.
마지막 9개 요청을 평가했다.
독립 Start 기반 95% CI는 없다.

| 비교 | Layer D Forward p50 | Token D Forward p50 | Layer Client TTFT p95 | Token Client TTFT p95 |
|---|---:|---:|---:|---:|
| Last4 / Token≈889 | 30.3130ms | 38.3212ms | 318.0027ms | 329.9976ms |
| Last2 / Token≈444 | 15.2960ms | 53.8052ms | 323.9960ms | 381.2160ms |

위 D Forward는 CUDA Event 경과시간이다.
Attention, Norm, Activation과 Launch Gap도 포함할 수 있다.
순수 GEMM 시간이나 Exclusive Kernel Time이 아니다.
Handoff와 D Admission 전체를 포함하는 구간도 아니다.

Last4의 Forward p50은 Token 대비 20.8975% 낮았다.
같은 진단의 TTFT p95는 3.6348% 낮았다.
두 Metric의 Sample 분위와 측정 경계가 다르다.
두 시간 차이를 더하거나 빼서 TTFT p95를 분해하지 않는다.
Last2 결과는 이전 진단이다.
취소된 Last2 Online Sweep을 재개한 결과가 아니다.

## 2. 같은 실제 Batch1의 Online 대기

다음은 새 Discovery의 Input8000/Output16/QPS6 조건이다.
P 준비, D Suffix 준비와 Full36 Decode의 실제 Batch는 모두 1이다.
양쪽 Worker는 B8192/S1024이다.
N72, Arrival Seed2026111810이며 각 방식은 독립 Start 1회이다.
정합성 및 실제 Batch Gate를 통과한 실행이다.
과부하 Finite Cohort이며 Sustainable Capacity 판정이 아니다.

| Host Metric | Layer Last4 | Token≈889 |
|---|---:|---:|
| Mean TTFT | 7969.9829ms | 8314.4662ms |
| TTFT p95 | 14440.2333ms | 15086.1977ms |
| Arrival → P Host Submit 평균 | 3184.2273ms | 3055.8215ms |
| P Submit → Source Notification 평균 | 258.3478ms | 256.7782ms |
| Source Notification → D Host Submit 평균 | 4479.7812ms | 4946.3386ms |
| D Submit → First Worker Notice 평균 | 47.3568ms | 55.2545ms |
| First Worker Notice → Client 평균 | 0.2697ms | 0.2733ms |
| D Decode Residency/Request | 352.2107ms | 353.8372ms |

처음 다섯 Host 구간의 평균 합은 Mean TTFT와 일치한다.
D Decode Residency는 이 합의 별도 여섯 번째 항목이 아니다.
이 값은 이전 Cohort가 다음 요청의 Admission을 지연할 수 있는 실행 구조를 설명한다.
Host 경과시간을 순수 GPU 계산시간으로 표시하지 않는다.

이 조건에서 Layer의 TTFT p95는 Token보다 4.2818% 낮았다.
하지만 독립 Start 1회이므로 반복 가능한 우위는 아직 미입증이다.
수십 ms의 D 준비 비용 외에 수 초의 제출 대기가 존재한다.
따라서 자기 요청의 Prefill GEMM만으로 전체 TTFT를 설명할 수 없다.
같은 실행에서 P Compute Saturation을 확인했다고 주장하지 않는다.

## 3. 실제 Serving GEMM 경로 감사

이전 고립 GEMM은 `torch.nn.functional.linear`를 사용했다.
현재 Serving은 Batch-invariant 분기에서 `linear_batch_invariant`를 직접 호출한다.
설치된 H100 경로에서 두 호출은 같은 API가 아니다.
이 차이는 실제 Serving Forward나 위 Client Latency를 변경하지 않았다.
이전 고립 측정을 현재 Serving의 GEMM 비용으로 대체해서 사용하지 않는다.

새 고립 대조는 실제 Serving Projection API를 사용했다.
GPU1의 Layer0 Projection 중앙값을 확장한 모델은 다음과 같다.

| Pair | Layer Projection 모델 | Token Projection 모델 | Token/Layer |
|---|---:|---:|---:|
| Last4 / Token≈889 | 22.8293ms | 25.2129ms | 1.1044 |
| Last8 / Token≈1778 | 45.6586ms | 49.0766ms | 1.0749 |
| Last12 / Token≈2667 | 68.4879ms | 71.8138ms | 1.0486 |

이 표는 `STRUCTURAL_ACCOUNTING`이다.
실측 Full Forward나 Online Exclusive GEMM 시간이 아니다.
이 모델은 Cut이 커질수록 GEMM 이득이 줄어드는 경향을 보인다.
큰 Cut에서 Token이 반드시 역전한다고 예측하지는 않는다.

이전 Loop2048/R32의 Token64와 Dense8000/36의 Token889는 같은 Shape가 아니다.
Model Dimension과 Kernel 경로도 같지 않다.
이전 Loop의 속도 비율을 이번 Dense 실행에 그대로 적용하지 않는다.

## 4. Attention과 전송

같은 Token-layer Rows는 같은 D Attention Work를 뜻하지 않는다.
Causal Pair 계산에서 Last4의 Token D Attention은 Layer의 약 1.889배이다.
Last8은 약 1.778배이고, Last12는 약 1.667배이다.
이는 `STRUCTURAL_ACCOUNTING`이며 GPU 시간 비율이 아니다.
해당 차이는 Layer 이득의 후보 원인이다.
Layer 이득이 작아지는 이유라고 반대로 해석하지 않는다.

Last4의 NIXL 보고 KV Payload는 양쪽 모두 Request당 1,048,576,000 Bytes이다.
Layer에는 Hidden/Residual 131,072,000 Bytes가 추가된다.
Token에는 이 추가 Payload가 없다.
동일 GEMM Rows에서 KV Payload 차이가 있다고 가정하지 않는다.
추가 Hidden Payload가 TTFT Critical Path에서 무시 가능한지는 별도 Timing 검사가 필요하다.
전송 API 구간을 순수 DMA 시간으로 표시하지 않는다.

## 5. 순위가 있는 원인 가설과 반증 검사

1. D Cohort 대기가 Forward 이득을 희석할 수 있다.
   같은 준비 Batch1을 유지한 Decode1/Decode2 대조에서 이 영향을 검사한다.
   Decode Residency, 실제 Batch와 Client TTFT를 함께 비교한다.
   Decode2는 원래 Batch1 실행과 별도 조건이다.
   정합성을 통과하기 전에 성능 결과로 사용하지 않는다.
2. 실제 GEMM Shape 이득이 이전 고립 추정보다 작을 수 있다.
   실제 Serving API 대조는 이 가능성과 일치한다.
   Online Exclusive Projection 시간은 아직 `NOT_MEASURED`이다.
3. 추가 Hidden 전송과 Host 처리가 작은 계산 이득을 상쇄할 수 있다.
   같은 Request의 준비, Handoff, Admission 및 First Output 경계를 검사한다.
   현재 Payload 크기만으로 시간 병목을 확정하지 않는다.
4. Attention과 다른 Operator가 Forward 이득의 일부를 설명할 수 있다.
   계측이 Clean Run을 5% 넘게 변경하면 Diagnostic으로만 분류한다.
   GEMM 모델을 전체 Forward에서 빼서 나머지를 Attention 시간이라고 표시하지 않는다.

현재 Batch1 반복 실험은 계속 실행 중이다.
고정 Decode2 대조는 별도 State 및 실제 Geometry Gate를 먼저 통과해야 한다.
GEMM, 대기와 전송 중 어느 요소가 Crossover를 결정하는지는 아직 최종 판정하지 않았다.

## 근거

- [실제 Last2/Last4 Forward와 직렬 TTFT](dense_ar_last2_last4_actual_suffix_quick_comparison.md).
- [Serving Projection API 감사와 고립 모델](dense_ar_serving_projection_api_measurement_audit.md).
- [실제 Batch1 Protocol](dense_ar_batch_matched_axis_tradeoff_protocol.md).
- [Host 구간 분석](dense_ar_matched_axis_crossover_20261117/analysis/batch1_cost_and_critical_path.json).
- Layer 실행: `batch1_discovery_Layer_Last4_N72_O16_q6_s2026111810_P1D1_v1`.
- Token 실행: `batch1_discovery_Token_Last4_N72_O16_q6_s2026111810_P1D1_v1`.

각 실행의 Config, Source ZIP, Prompt IDs, Arrival 및 실제 Geometry를 원래 Namespace에 보존했다.
이 문서는 새로운 Serving 성능을 추정해서 추가하지 않았다.
