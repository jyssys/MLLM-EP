# LongBench400-v2 후보 감사

증거 수준: `TRACE_ANALYSIS`. 기존 BI0 Natural EOS Reference를 새 Output 구간으로 집계했다.
이 문서를 작성할 때 v2의 400개 요청은 아직 선정하지 않았다.

| Qwen Chat Template 적용 후 Input Tokens | Output1–10 | Output11–50 | Output51–100 | Output101–200 |
|---|---:|---:|---:|---:|
| 3000–5999 | 227 | 144 | 26 | 8 |
| 6000–9999 | 228 | 124 | 49 | 16 |
| 10000–19999 | 493 | 171 | 117 | 54 |
| 20000–36000 | 60 | 26 | 17 | 5 |

기존 후보는 2,165개다. 새 Output 구간 밖의 후보는 400개다.
각 Input 구간에서 100개는 선정할 수 있다.
그러나 각 Cell의 25개 목표는 기존 Pool만으로 달성할 수 없다.

Dataset Revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다.
Reference는 Original Qwen3-8B, BI0, Concurrency1, Greedy, Thinking Disabled를 사용했다.
Prompt는 Mode0 Eager로 실행했다. Decode에는 Native CUDA Graph를 사용했다.
EOS를 허용했다. 공통 Output Cap은 2048이었다.
이 길이는 정답 Text의 길이가 아니다. 실제 모델 생성 길이다.
Compiled C3의 생성 길이는 이 Reference와 다를 수 있다.

## 사전 동결한 후보 확대와 재배분

나머지 공식 Non-E Task 8개의 적격 원문 전체를 조사한다.
Task는 vcsum, lcc, repobench-p, trec, lsht, passage_count, passage_retrieval_en, passage_retrieval_zh다.
Input 길이가 적격한 추가 요청은 1,809개다.
이 요청의 Natural Output은 이 문서 작성 시 측정 중이다.
일부 요청의 결과를 보고 해당 Task의 나머지 측정을 중단하지 않는다.

Task 구성에는 분류, 코드 및 중국어 요청도 포함될 수 있다.
이 구성을 Long-context QA만의 대표 분포로 해석하지 않는다.
선정 집합과 후보 전체의 Task 구성 및 길이 분포 차이를 별도로 보고한다.

각 Input 구간에서 각 Output Cell에 최대 25개를 먼저 배정한다.
부족분은 같은 Input 구간의 인접 Output Cell에서 채운다.
동일 거리에서는 사전에 정한 Task 순서와 Request ID Hash를 사용한다.
적격 원문이 Input 구간당 100개 미만이면 다른 Input 구간에서 조용히 빌리지 않는다.
그 경우 Dataset 부족분을 기록한다.
원문 수정, 새 Truncation, Padding 및 Request 복제는 사용하지 않는다.
Task 품질과 Serving 성능은 선정 기준에 사용하지 않는다.

선정 Seed는 `2026112201`이다.
기계 판독 계약은 새 Namespace의 `configs/dataset_selection_contract_v2.json`에 있다.
후보 SHA256과 기존 Reference SHA256도 그 파일에 기록했다.
기존 v1 Manifest와 결과는 변경하지 않는다.

원문 출처: [공식 LongBench Dataset](https://huggingface.co/datasets/zai-org/LongBench).
