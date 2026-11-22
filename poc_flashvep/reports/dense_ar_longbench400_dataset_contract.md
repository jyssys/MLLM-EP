# LongBench-Balanced400-v1 데이터셋 계약

증거 수준은 `MEASURED_NATURAL_OUTPUT_REFERENCE`다. Online Serving 결과는 아직 `NOT_RUN`이다.

원문 요청 400개를 동결했다. Input 구간마다 100개를 배정했다.
16개 Cell의 균등 배정은 불가능했다. 부족분은 사전 기준에 따라 같은 Input 구간의 인접 Output Cell로 재배분했다.
이 판정은 선언한 13-task Natural Generation Pool에 한정한다. 모든 공식 Task에 대한 불가능성 증명은 아니다.

| Input Tokens | Output1–8 후보/선정 | Output9–24 후보/선정 | Output25–49 후보/선정 | Output50–99 후보/선정 |
|---|---:|---:|---:|---:|
| 3000–5999 | 215/25 | 91/25 | 62/25 | 28/25 |
| 6000–9999 | 209/25 | 69/25 | 72/25 | 50/25 |
| 10000–19999 | 473/25 | 104/25 | 83/25 | 119/25 |
| 20000–36000 | 54/52 | 29/29 | 3/3 | 16/16 |

| 통계 | Input Tokens | Natural Output Tokens |
|---|---:|---:|
| mean | 13839.600 | 27.420 |
| median | 9965.000 | 16.000 |
| min | 3088.000 | 2.000 |
| max | 32988.000 | 99.000 |
| p10 | 4656.400 | 4.000 |
| p25 | 6004.750 | 5.000 |
| p75 | 19356.500 | 45.000 |
| p90 | 31397.000 | 69.000 |
| p95 | 31446.250 | 84.000 |
| p99 | 32980.010 | 98.010 |

## 생성 및 선정 계약

Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`이다. 선정 Seed는 `2026112101`이다.
Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.
Input 길이는 Qwen3-8B Chat Template 적용 후 Token 수다. Thinking은 비활성화했다.
원문 변경, 새 Truncation, Padding, 요청 복제는 사용하지 않았다.
Natural Output은 BI0 Single-GPU Concurrency1에서 측정했다. Greedy, Natural EOS, 공통 Cap2048을 사용했다.
이 Cap은 LongBench의 Task별 기본 Generation Cap과 다르다.
후보 2165개 중 Output1–99 후보는 1677개였다. 범위 밖 Output은 488개였다.
후보 전체의 Cap Hit는 1개다. 선정 집합의 Cap Hit는 0개다.
선정 집합은 400/400개가 EOS로 종료됐다.
전체 Input은 5535840 Tokens다. 전체 Output은 10968 Tokens다.
고유 Request는 400개다. 고유 Context는 272개다.
같은 Context는 동일 Discovery/Held-out 집합에 배정했다. 집합 사이 Context 중복은 0개다.
Discovery와 Held-out은 각각 200개다. Calibration 집합과 Main400의 Context 중복은 0개다.

## Task 구성

| Task | Requests |
|---|---:|
| narrativeqa | 79 |
| multifieldqa_en | 54 |
| musique | 34 |
| qmsum | 78 |
| hotpotqa | 19 |
| 2wikimqa | 36 |
| qasper | 63 |
| samsum | 10 |
| triviaqa | 23 |
| multifieldqa_zh | 3 |
| dureader | 1 |

## 선택 편향과 한계

후보 전체의 평균 Output은 111.879 Tokens다. 선정 집합의 평균 Output은 27.420 Tokens다.
짧은 Natural Output을 조건으로 선정했다. 따라서 선정 집합은 일반 Production 분포를 대표하지 않는다.
QA Score와 Serving 성능은 선정 기준에 사용하지 않았다. 품질이 낮은 요청도 사후 교체하지 않는다.
같은 Context를 사용하는 서로 다른 Request는 허용했다. Request 자체의 복제는 허용하지 않았다.
최초 7개 Task를 우선했다. 사전에 선언한 공식 Task 확장 순서로 부족한 Cell을 보완했다.
Input 구간별 균형은 달성했다. Input×Output의 16개 Cell 균형은 부분 달성이다.

## 재현 자료

전체 Manifest는 캠페인 Namespace의 `data/longbench_balanced400_v1.json`에 있다.
후보 분포와 부족분 재배분은 `data/balanced400_freeze_manifest.json`에 있다.
후보 전체의 원문 및 Natural Output 참조는 별도로 보존했다.
[공식 Dataset Snapshot](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672)을 사용했다.
Task 품질 검사는 별도 결과로 저장한다. 품질 검사가 동결 집합을 변경하지는 않는다.

## 이후 분석을 위한 범위 명시

첫 문단의 `NOT_RUN`은 Dataset 동결 시점의 상태다.
이 문서의 Output 분포는 C1 Natural Reference다. 이후 Online Start의 실제 Output 수는 별도 Metric에 기록한다.
Natural Reference는 Original Model, BI0, Compile Mode0와 Full-decode CUDA Graph를 사용했다.
이를 C3 Default Compile의 모든 Greedy Sequence와 동일하다고 가정하지 않는다.

16개 Cell 부족 판정은 사전에 선언한 13-task Generation Pool 2165개에 한정한다.
21개 비-E Task의 Input 감사와 13개 Task의 Natural Generation은 다른 범위다.
모든 공식 Task를 자연 생성해 전역적 불가능성을 증명한 것은 아니다.
별도 Input 감사에서 추가 VCSum 원문 181개가 길이 조건을 만족했다.
이를 동결한 Natural Generation Pool에 사후 편입하지 않았다.

위의 Calibration/Main Context 중복 0개는 별도 Qualification1/8/32 집합을 가리킨다.
C0 Cost Model의 전체 Training Trace가 Main400과 겹치지 않는다는 뜻은 아니다.
Cost Model은 Discovery Prompt를 사용한 Capacity Pilot도 포함한다.
별도 Training/Validation 감사는 C0 Validation Prompt와 Context의 중복을 검사한다.
이를 Main400 전체가 모든 분석에서 미관측이었다는 주장으로 확대하지 않는다.

## 동결한 요청 순서와 Round-robin 작업량

요청 순서는 Input Bucket 순서가 아니다.
선정기는 `SHA256("arrival-order-" + seed + prompt_id)` 순서로 400개를 정렬했다.
Seed는 `2026112101`이다. 실제 Manifest가 이 순서와 같은지 다시 검사했다.
이후 Serving 결과를 보고 순서를 변경하지 않았다.

첫째부터 넷째 100-request 구간의 평균 Input은 각각 13730.36, 14912.83, 13708.55, 13006.66 Tokens다.
이를 Stationary Production Traffic의 증거로 해석하지 않는다.
Arrival Seed 변경은 Poisson 도착 간격을 바꾼다. 동결한 Prompt 순서를 다시 섞지는 않는다.

| Round-robin Replica | Requests | Input Tokens 합 | C1 Reference Output Tokens 합 |
|---|---:|---:|---:|
| 0 | 200 | 2832232 | 5298 |
| 1 | 200 | 2703608 | 5670 |

Round-robin은 Request 수를 나눈다. Token 작업량이 정확히 같은 것은 아니다.
Output 열은 C1 Natural Reference다. 실제 Agg Start의 생성 수를 대신하지 않는다.
실제 생성 수, Batch와 Client Metric은 각 Start의 Raw에 별도로 기록한다.

## Discovery와 Held-out의 분포 차이

Context 중복을 막은 200/200 분할은 각 Input 구간의 50/50 분할이 아니다.
동결한 Manifest를 다시 계산한 결과는 다음과 같다.

| 집합 | 3K–6K | 6K–10K | 10K–20K | 20K–36K | Input 평균(Tokens) | C1 Output 평균(Tokens) |
|---|---:|---:|---:|---:|---:|---:|
| Discovery200 | 55 | 56 | 56 | 33 | 11808.345 | 30.010 |
| Heldout200 | 45 | 44 | 44 | 67 | 15870.855 | 24.830 |

Discovery의 전체 Input/Reference Output은 2361669/6002 Tokens다.
Held-out은 각각 3174171/4966 Tokens다.
Held-out에는 긴 Input이 더 많고 C1 Reference Output은 더 짧다.
두 집합을 같은 길이 분포라고 가정하지 않는다.
Main400과 Heldout200의 같은 Offered QPS도 같은 P/D 부하를 뜻하지 않는다.

이 차이는 Context-disjoint 분할의 관측 특성이다.
Serving 결과를 보고 Request를 옮기거나 분할을 다시 균형화하지 않았다.
Latency 차이 전체를 이 분포 차이 하나의 인과 효과로 확정하지 않는다.
각 실제 Start의 Output 길이와 Queue를 별도로 확인해야 한다.
