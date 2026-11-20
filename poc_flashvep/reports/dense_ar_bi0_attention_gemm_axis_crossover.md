# BI0 Attention·GEMM 비용과 Queue Crossover

상태: `CLOSED / PARTIAL_ROOT_CAUSE_IDENTIFICATION`. Compute Sweet Spot과 Online Serving Sweet Spot을 구분한다.

## 적격 P Kernel 측정

다음 측정은 Current LongBench200의 Last4 진단이다.
Input Length는 가변이다. Request별 Output Count를 고정한 총4723 Token Replay다.
QPS2.35, Arrival Seed2082, 실제 Batch1/1/1, Credit4와 KV Block128을 사용했다.
이는 Historical Input8000/Output16 실험이 아니다.

| P GPU 측정 | Layer | Token | Layer−Token |
|---|---:|---:|---:|
| Request-attributed Kernel Union | 462.531ms | 449.767ms | +12.764ms |
| 식별한 Attention Kernel | 145.325ms | 130.601ms | +14.724ms |
| 식별한 GEMM Kernel | 216.015ms | 214.504ms | +1.512ms |

각 Axis는 Profile1회와 Clean Control2회를 사용했다.
API Correlation Coverage는100%다. Dropped/Invalid Activity는 없었다.
모든 Observer 비교값의 최대 변화는 Layer3.145%, Token4.274%다.
두 Profile은5% Gate를 통과했다. 독립 Kernel Start3회 CI는 없다.

식별한 GEMM은 전체 GEMM 시간을 뜻하지 않는다.
분류하지 못한 Kernel은 Other에 남겼다. SM/HBM Counter는 `NOT_MEASURED`다.
같은 Profile의 P Host Stage 차이는15.725ms다.
Kernel 차이12.764ms는 그 차이의 약81.2%다.
이 비율은 순수 GPU Service Intervention의 인과 기여율이 아니다.
이 값을 과거 Natural Last4의 P Queue 차이2711ms에 직접 대입하지 않는다.

## 구조적 작업량

Input L, Last-m과 Token Suffix-s의 D GEMM Rows는 각각 Lm과36s다.
GEMM Rows를 맞춰도 Attention Pair와 Transfer Bytes는 같지 않을 수 있다.
Token은 P의 후반 Token Attention을 줄인다.
Layer는 D에서 큰 Token Dimension을 적은 Layer에 적용한다.
전체 P+D의 논리적 Attention Pair는 같다. P/D 배치 위치가 다르다.

정확하게 Rows를 맞추고 L이 큰 경우 다음 근사를 얻는다.
P Attention의 Layer/Token 비율은36/(36−m)에 접근한다.
D Attention의 Token/Layer 비율은2−m/36에 접근한다.
이는 큰 L의 구조적 Pair 근사다. 실측 GPU 시간 비율이 아니다.

| Cut | P Attention Layer/Token 근사 | D Attention Token/Layer 근사 |
|---|---:|---:|
| Last2 | 1.058824 | 1.944444 |
| Last4 | 1.125000 | 1.888889 |
| Last8 | 1.285714 | 1.777778 |
| Last12 | 1.500000 | 1.666667 |

Layer의 D 계산 Shape가 유리해도 P에는 더 많은 Attention이 남는다.
P가 병목이면 이 차이가 Queue에서 누적될 수 있다.
D가 병목이면 반대의 배치 비용이 중요할 수 있다.
어느 쪽이 실제 병목인지는 측정한 Host Service와 Queue로 따로 검사한다.
Attention Pair 비율만으로 Axis Winner를 결정하지 않는다.

전체200개에서 기존 nearest16 규칙의 D Rows 오차는 다음과 같다.
Last2 −0.0384754%, Last4 +0.0115716%, Last8 −0.00737687%, Last12 −0.00135002%다.
이는 Exact Matched Work가 아니다. Padding Query도 별도로 기록한다.
Historical N72의 혼합 Suffix 계획과 혼동하지 않는다.
Rows 오차가 작다고 GPU 시간 오차도 작다고 가정하지 않는다.
인접 길이의 별도 Exact-rows Service Intervention은 이번 예산에서 `NOT_RUN`이다.
Alignment와 Chunk 경계의 효과를 완전히 배제하지 않는다.

## 남은 검사

Input Bucket8192/16384/32768은 원본 Request를 그대로 사용한다.
Bucket 중심값을 정확한 Input Length라고 표시하지 않는다.
Bucket별 Output 분포도 다르다. 따라서 Input Length만 바꾼 순수 Intervention은 아니다.
각 Pair 안에서는 Prompt, Arrival와 Output 계약을 일치시킨다.
Kernel 비용은 측정한 Cell에서만 보고한다.
Host Forward 시간을 다른 Cell의 Exclusive GEMM 시간으로 대체하지 않는다.

## 식별한 원인과 식별하지 못한 원인

적격 Fixed-output Last4 Profile은 P Attention 비용 차이를 확인했다.
Source Query Chunk 수와 Host 준비 횟수도 Axis에 따라 달라질 수 있다.
따라서 추가 P Host 비용을 Attention 하나로 모두 설명하지 않는다.
Natural Last4의 정확한 GPU 인과 기여율은 `NOT_IDENTIFIED`다.
D Suffix의 적격 Kernel별 Breakdown도 `NOT_MEASURED`다.
Input8000/8192/16384/32768 전체의 Kernel Cost Matrix는 실행하지 않았다.
Bucket Online 결과나 구조적 Attention Pair를 그 빈칸의 GPU 시간으로 채우지 않는다.

별도 BI1 Coalescing Flag Intervention은 Model Math를 바꾸지 않았다.
상대 TTFT 비율은 달랐지만 실제 Descriptor 수는 줄지 않았다.
따라서 물리적 Coalescing의 인과 효과는 식별하지 못했다.
Flag 변경의 Timing, 절대 성능 악화와 Intervention 실현 실패를 함께 보고한다.

[적격 Source Compute 비교](dense_ar_bi0_axis_rootcause_20261120/analysis/qualified_matched_source_compute_comparison.json)를 참조한다.
