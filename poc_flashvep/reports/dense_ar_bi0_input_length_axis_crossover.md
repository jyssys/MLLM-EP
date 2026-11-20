# BI0 Input Length와 Partition Axis

상태: `CLOSED`. 증거는 `DIAGNOSTIC`과 `STRUCTURAL_ACCOUNTING`이다.

## 원본 Length Bucket

고정 길이를 만들려고 Prompt를 새로 자르거나 Padding하지 않았다.
기존 LongBench200에서 다음 범위의 원본 Request를 사전 선정했다.

| Bucket 이름 | 실제 Input 범위 |
|---|---:|
| 8192 | 6144–10240 Tokens |
| 16384 | 14336–18432 Tokens |
| 32768 | 28672–36864 Tokens |

각 Bucket은 최대8개다. 선정 Seed는2086이다.
Bucket 이름은 정확한 Input Token 수가 아니다.
이미 동결한 LongBench Context Truncation은 변경하지 않았다.
Output은 Request별 C1 Reference의 생성 길이를 사용하는 Fixed-length 진단이다.
Bucket 사이에는 Output 분포도 다르다.
따라서 Bucket 차이를 Input Length만의 순수한 인과 효과로 해석하지 않는다.

## 공통 통제

BI0, B8192/S1024, KV128/42GiB, Credit4와 Hidden Slots4를 사용한다.
P Prefill/D Suffix/D Decode의 실제 Batch는1/1/1이다.
Last2/4/8/12의 Layer와 Token을 같은 Prompt·Arrival·Output Count로 비교한다.
QPS는2.35다. Arrival Seed는2087이다.
Cell별 독립 Start는1회다. 독립 Start CI는 없다.
B8192 때문에 생기는 Prefill Chunk와 Padding Query를 실제 Query Log에서 확인한다.

## 작업량과 한계

목표 D Rows는 Input Length×Cut이다.
Token의 실제 D Rows는36×Suffix Length다.
기존 nearest16 Prefix 규칙은 Aggregate Rows를 항상 정확하게 맞추지 못한다.
Cell별 실제 Rows와 Padding을 기록한다.
같은 GEMM Rows를 같은 Attention FLOPs 또는 GPU Time이라고 표현하지 않는다.

## 완료한24개 진단 실행

각 Bucket은 실제8개 Request다. 다음 값은 Layer/Token TTFT p95 비율이다.
각 Cell은 독립 Start1회다. 따라서 다음 표에는 성능 CI가 없다.

| 실제 Input Bucket | Last2 | Last4 | Last8 | Last12 |
|---|---:|---:|---:|---:|
| 6144–10240 | 0.938430 | 0.931373 | 0.996427 | 0.957818 |
| 14336–18432 | 1.051740 | 1.064602 | 1.080164 | 1.099030 |
| 28672–36864 | 1.005958 | 1.086560 | 1.117511 | 1.173153 |

짧은 Bucket에서는 Layer가 빠른 Point가 있었다.
긴 Bucket에서는 Token이 빠른 Point가 있었다.
이는 후속 가설의 후보이지 재현 가능한 Crossover의 확정이 아니다.
Task, 원문, Output 길이와 Chunk Count도 Bucket 사이에서 달라진다.
이 진단에는 독립3회와 Native-relative Decode CI가 없다.
따라서 Layer Sweet Spot PASS로 판정하지 않는다.

[사전 Bucket 계약](dense_ar_bi0_axis_rootcause_20261120/configs/frozen_input_bucket_control_v1.json),
[실행 결과](dense_ar_bi0_axis_rootcause_20261120/analysis/input_bucket_online_controls_v1.json)를 참조한다.
다른 Output·Cut의 TTFT를 하나의 Serial/Online 역전 식에 합치지 않는다.
