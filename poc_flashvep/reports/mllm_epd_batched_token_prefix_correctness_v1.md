# Generic Token Split — 동시 요청 Prefix 정합성

C31 N1과 동시 제출 N8의 Prefix 상태 검사가 통과했다.
증거 수준은 `CORRECTNESS_ONLY`다. Online 성능과 Decode 보호 결과가 아니다.

## 실행 계약

Qwen2.5-VL-7B, Original BF16, BI0, 원본 Processor, 전체 28개 Layer를 유지했다.
모델/Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
GPU4/5/6은 E/P/D다. P/D B8192/S1024, KV42GiB, Block128이다.
공통 Compiled 설정은 `mllm_epd_compiled_token_correctness_v1.md`와 같다.
원본 이미지 전체를 E에서 실제 Encoding했다. 전체 Sequence의 마지막 20%가 D Suffix다.
각 요청의 생성 상한은 1 Token이다. EOS를 허용했다.
이는 정합성 전용 계약이다. Main100 Natural EOS Cap2048 계약은 변경하지 않았다.

## 동시 요청과 실제 Batch

N1 뒤에 서로 다른 Calibration 요청 8개를 함께 제출했다.
모든 P Prefix가 끝난 뒤 D Continuation 8개를 함께 제출했다.
이 Barrier는 Pending KV의 Lifetime을 검사한다. 제안하는 Serving 정책이 아니다.
기본 vLLM Continuous Scheduler와 원본 Forward는 유지했다.
동시 요청 수 8을 실제 GPU Batch8로 표시하지 않는다.

| Cohort | P 실제 Batch별 Forward 수 | D 실제 Batch별 Forward 수 |
|---|---|---|
| token_BATCH_C31_N1 | {'1': 1} | {'1': 1} |
| token_BATCH_C31_N8 | {'1': 5, '2': 3, '3': 2} | {'1': 2, '2': 2, '3': 1} |

## 요청별 검사

P/D의 Prefix IDs, Prefix MRoPE와 실제 NIXL 수신 KV Hash를 비교했다.
28개 Layer 모두 Byte 단위로 일치했다. Physical Block 번호는 비교하지 않았다.
P Query는 0→k, D Query는 k→L을 빠짐없이 한 번씩 처리했다.
실제 Forward의 Query Position은 요청별 원본 MRoPE Slice와 일치했다.
읽기 계측은 원본 Forward 뒤에 CPU Hash를 계산한다. 모든 Timing은 성능 평가에서 제외한다.

| Calibration | Input | P Prefix | D Suffix | Images | Strict Prefix Layers | Serial Teacher 첫 Token 일치 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 4831 | 3865 | 966 | 1 | 28 | True |
| 1 | 7823 | 6258 | 1565 | 1 | 28 | True |
| 4 | 5032 | 4026 | 1006 | 2 | 28 | True |
| 6 | 2961 | 2369 | 592 | 2 | 28 | True |
| 12 | 5343 | 4274 | 1069 | 3 | 28 | True |
| 17 | 14431 | 11545 | 2886 | 3 | 28 | True |
| 20 | 17334 | 13867 | 3467 | 28 | 28 | True |
| 31 | 27840 | 22272 | 5568 | 45 | 28 | True |

Serial Teacher의 첫 Token과 일치한 요청은 8/8이다.
BI0는 다른 Batch Geometry 사이의 완전 불변성을 보장하지 않는다.
모든 요청 완료 뒤 E Embedding Ownership은 비어 있었다.
미확인 ACK를 무시하거나 사용 중인 Buffer를 조기에 반환하지 않았다.

## 이 검사가 입증하지 않는 항목

전체 Suffix KV의 Cross-batch 동등성과 Task Quality는 이 검사에서 미확정이다.
긴 Natural Output, 후속 Decode Residency, Active Decode 중 삽입은 검사하지 않았다.
독립 N3 C3 성능, Native 대비 ITL/TPOT 보호와 SLO-goodput도 미측정이다.
동시 Prefix Handoff 통과를 전체 Continuous Serving 성공으로 표시하지 않는다.
Raw Source, 명령, GPU Lease는 C31 Namespace에 보존했다.
대응 CSV는 `analysis/batch_token_C31_request_correctness_v0.csv`다.
Figure는 `figures/batch_token_C31_prefix_correctness_v0.png`와 같은 이름의 PDF다.
