# LongBench-Balanced400-v2 데이터 계약

## 판정과 범위

실제 원문 400개를 동결했다. Input 구간별 100개를 확보했다.
앞의 세 Input 구간은 Output 구간별 25개를 확보했다.
20K–36K 구간은 긴 Natural Output 후보가 부족했다. 이 부족분은 삭제하지 않았다.
사전에 정한 인접 Output 구간 재배분 규칙을 적용했다.
이 균형 벤치마크는 Production Traffic 분포가 아니다.

기준 Commit은 `5ebed239073b4fa3709361038cbec026d1c32b78`이다.
기존 v1 데이터와 결과를 변경하지 않았다.
새 Evidence Namespace는 `dense_ar_longbench400_v2_20261122`이다.

## Source와 선정

공식 LongBench v1 Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`이다.
공식 Prompt Template Revision은 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`이다.
모델은 Qwen3-8B다. 모델 Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.
Tokenizer와 Chat Template도 이 Revision을 사용한다.
Thinking을 사용하지 않는다. Sampling은 Greedy다. Precision은 Original BF16이다.

기존 13개 Task의 후보 2,165개를 먼저 감사했다.
이후 나머지 8개 공식 Task의 실행 가능한 원문 1,809개를 추가했다.
총 3,974개 후보에서 BI0 Natural EOS 출력을 확인했다.
아래 실제 후보 수는 최종 400개 선정 전에 사용자에게 보고했다.

| Input Tokens | Output1–10 | Output11–50 | Output51–100 | Output101–200 |
|---|---:|---:|---:|---:|
| 3,000–5,999 | 430 | 144 | 44 | 54 |
| 6,000–9,999 | 278 | 126 | 64 | 79 |
| 10,000–19,999 | 759 | 177 | 131 | 86 |
| 20,000–36,000 | 86 | 31 | 17 | 6 |

선정 Seed는 `2026112201`이다.
Task 확대 전 원문을 우선한다. 같은 우선순위에서는 Seed와 Prompt ID의 SHA256으로 순서를 정한다.
각 Cell에서 최대 25개를 선정한다.
부족한 Cell은 같은 Input 구간의 인접 Output 구간으로 재배분한다.
Input 구간 사이에서는 요청을 차용하지 않는다.
원문 복제, Filler, 새 Truncation 또는 EOS 조작을 사용하지 않았다.

## 동결된 4×4 분포

| Input Tokens | Output1–10 | Output11–50 | Output51–100 | Output101–200 | 합계 |
|---|---:|---:|---:|---:|---:|
| 3,000–5,999 | 25 | 25 | 25 | 25 | 100 |
| 6,000–9,999 | 25 | 25 | 25 | 25 | 100 |
| 10,000–19,999 | 25 | 25 | 25 | 25 | 100 |
| 20,000–36,000 | 46 | 31 | 17 | 6 | 100 |
| 합계 | 121 | 106 | 92 | 81 | 400 |

20K–36K의 Output51–100은 목표보다 8개 부족하다.
같은 Input 구간의 Output101–200은 목표보다 19개 부족하다.
이 요청들을 성능 결과에 따라 교체하지 않는다.

## 길이 통계

선정용 Natural Output은 Original Full36 C1의 BI0 Mode0 Prompt 경로를 사용했다.
후속 Decode는 Native Full36 Graph 경로를 사용했다.
EOS를 허용했다. Output Cap은 2,048 Tokens다.
아래 Output 수치는 이 선정용 경로의 실측값이다.

| 통계 | Input Tokens | 선정용 Natural Output Tokens |
|---|---:|---:|
| Count | 400 | 400 |
| 평균 | 13,633.725 | 49.96 |
| 중앙값 | 9,953 | 31 |
| 최소 | 3,061 | 2 |
| 최대 | 32,988 | 194 |
| p10 | 4,505.2 | 4 |
| p25 | 6,013.25 | 7 |
| p75 | 18,953 | 85.25 |
| p90 | 31,128.2 | 123 |
| p95 | 31,444.1 | 150 |
| p99 | 32,978.02 | 182.05 |
| 합계 | 5,453,490 | 19,984 |

선정된 400개 모두 EOS로 종료했다. 선정용 경로의 Cap Hit은 0/400이다.

## Compiled C1 교차 검사

Strong C3의 품질 기준선은 별도의 Original Full36 Default-compiled C1이다.
이 기준선의 계약은 새 C3 품질 결과를 보기 전에 동결했다.
Task Macro 품질 비열등성 Margin은 기존과 같은 2 Percentage Points다.
Mode0 기준 품질 비교도 별도로 보존한다.

같은 400개의 Compiled C1 Output 평균은 53.0325 Tokens다.
중앙값은 31이다. 최소는 2다. 최대는 1,324다.
p10/p25/p75/p90/p95/p99는 각각 4/7/84.75/120.2/147.15/187 Tokens다.
총 Output은 21,213 Tokens다. 2/400개는 200 Tokens를 초과했다.
이 차이는 데이터 선정용 Mode0 길이가 모든 C3 실행에서 보장되지 않음을 보여준다.
이 결과에 맞춰 동결 데이터셋을 다시 선정하지 않았다.
각 C3 실행의 실제 Output 분포와 품질은 별도로 보고한다.

Natural EOS 비교에서 총 Output이 5%보다 많이 다르면 `OUTPUT_WORK_UNMATCHED`로 표시한다.
중요한 조건은 Compiled C1의 요청별 길이를 사용하는 별도 Controlled Replay로 검사한다.
이 Replay는 Natural Generation이 아니다. 이 길이는 Online Selector의 입력이 아니다.

## Task와 중복

| Task | Requests |
|---|---:|
| triviaqa | 34 |
| dureader | 27 |
| narrativeqa | 63 |
| hotpotqa | 12 |
| qmsum | 86 |
| trec | 16 |
| qasper | 36 |
| musique | 13 |
| 2wikimqa | 12 |
| repobench-p | 10 |
| multifieldqa_zh | 22 |
| samsum | 40 |
| multifieldqa_en | 23 |
| lsht | 5 |
| lcc | 1 |

영문 요청은 346개다. 중문 요청은 54개다.
Request ID는 400개 모두 고유하다. Context Hash는 287개다.
113/400개는 다른 요청과 Context를 공유한다. Request 중복은 0/400개다.
기존 v1과 Request ID 157개가 겹친다.

Discovery는 208개다. Held-out은 192개다.
두 집합의 Context Hash 중복은 0개다.
독립 정합성 검사 32개와 Main400의 Context 중복도 0개다.
과거 Compiler Control 8개와 Main400의 Context 중복도 0개다.
모든 Main400 요청은 C1 길이 측정에 사용됐다.
Held-out은 C3 튜닝에서 제외한 집합이다. 모든 과거 실험에서 미관측인 집합은 아니다.
C0의 Fixed-output 측정 노출도 별도로 기록한다.

## Hash와 재현 파일

- Main400 SHA256: `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`
- 후보 Pool SHA256: `3fe46dcd487fca707ae3715478a4ae26df3e7c964626741bbbc8727bb3b4b675`
- 후보 Natural Reference SHA256: `cc9f05cc1eb5907599859155badddc9880879e9edc5581fae6402f891da611db`
- 선정 계약 SHA256: `6be96a35c133a3060770e45df0d153de2f34f06e404aff3e4d55321e124a4152`
- Official Dataset ZIP SHA256: `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`

Manifest는 `dense_ar_longbench400_v2_20261122/data/balanced400_v2_freeze_manifest.json`에 저장했다.
요청 원문과 Token ID는 같은 Namespace의 `data/longbench_balanced400_v2.json`에 저장했다.
재배분의 요청별 기록도 Manifest에 포함한다.
PNG/PDF/CSV는 같은 Namespace의 `figures/v2_dataset_*` 파일에 저장했다.

Evidence는 `MEASURED_NATURAL_REFERENCE`다.
길이 균형과 Context 중복은 Dataset 속성이다. Serving Speedup 또는 Decode-safe Headroom의 근거가 아니다.
