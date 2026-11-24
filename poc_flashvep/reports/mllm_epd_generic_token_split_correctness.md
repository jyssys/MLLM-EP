# Generic Token Split — GPU 정합성 Checkpoint

Eager 경로의 고립 정합성 검사는 15/15 통과했다. Stock Compiled C3 정합성 Gate는 아직 통과하지 못했다.
증거 수준은 `CORRECTNESS_ONLY`다. Online Speedup과 Task Quality PASS를 뜻하지 않는다.

## 공통 계산 계약

GPU4/5/6은 각각 E/P/D다. 모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Original BF16, BI0, Full28, Greedy, Natural EOS Cap2048, 원본 이미지와 질문을 유지했다.
P는 원래 전체 입력을 검증한 뒤 Prefix Query만 처리한다. 원래 MRoPE용 이미지 Metadata는 보존한다.
D는 전체 Prompt와 전체 MRoPE를 유지한다. 실제 NIXL 전송으로 Prefix KV를 받는다.
이미지 내부 Split은 같은 원본 E Embedding을 P/D의 필요한 위치에 각각 전달한다.
모델 Forward, Attention, 원래 Token 순서는 변경하지 않았다. P의 임시 Sample은 사용하지 않는다.

## Compiled 경로의 실패와 대조 실험

첫 이미지 내부 Split에서 Prefix 전송은 28개 Layer 모두 Byte 단위로 일치했다.
Prepared Embeddings, Position IDs, 전체 D MRoPE도 일치했다. Greedy Output 16개도 같았다.
하지만 Full Prompt KV는 1/28 Layer만 Strict Match였다. 이 실패 뒤 성능 실험을 시작하지 않았다.
C17R1은 Token Split 없는 Full Prefill을 P와 D에서 각각 실행했다.
같은 P의 반복은 28/28이었다. P 대비 D Full Teacher는 1/28이었다.
따라서 관측한 차이를 Token Split 또는 KV 전송만의 오류로 설명할 수 없다.
첫 Layer의 Attention Q/K/V와 출력은 P/D 사이에서 Strict Match였다.
차이는 그 이후와 다음 Layer KV 사이에 있었다. 정확한 개별 Operator는 아직 식별하지 못했다.
C18은 P/D만 명시적 Eager 경로로 바꿨다. 원래 모델과 원래 E 경로는 유지했다.
이 경우 P 반복, D Full Teacher, Token D 모두 28/28 Layer와 Output 16개가 같았다.
Prefill의 Graph Mode는 Compiled 대조에서도 NONE이었다. Eager 변경은 Compile과 Graph Policy를 함께 비활성화한다.
따라서 Compiler 관련 Numerical Path를 지지한다. 특정 Codegen Operator의 단독 인과 효과로 확정하지 않는다.
고정한 Strict Gate 또는 품질 허용값을 완화하지 않았다. 기존 Compiled 기준선도 약화하지 않았다.

## 실제 Boundary 검사

아래 표는 C15R2의 명시적 Eager 경로다. 각 요청은 P/D Batch1로 순서대로 실행했다.
N8은 서로 다른 요청 8개를 뜻한다. 동시 Batch8 또는 C3 Continuous 검사가 아니다.
Teacher는 같은 Query Chunk 경계의 원본 Full Prefill이다. CPU Tensor 복사는 모든 Timing을 무효화한다.

| Case | L | P Prefix | D Suffix | Images | Natural Output | Strict KV Layers |
|---|---:|---:|---:|---:|---:|---:|
| N1_inside_single_image_TOKEN | 4831 | 2415 | 2416 | 1 | 16 | 28 |
| before_visual_TOKEN | 2961 | 15 | 2946 | 2 | 17 | 28 |
| inside_visual_TOKEN | 2961 | 1360 | 1601 | 2 | 17 | 28 |
| after_visual_TOKEN | 2961 | 2706 | 255 | 2 | 17 | 28 |
| between_images_TOKEN | 2961 | 2708 | 253 | 2 | 17 | 28 |
| text_tail_TOKEN | 2961 | 2953 | 8 | 2 | 17 | 28 |
| long_document_inside_image_TOKEN | 27840 | 13919 | 13921 | 45 | 60 | 28 |
| N8_0_TOKEN | 4831 | 3865 | 966 | 1 | 16 | 28 |
| N8_1_TOKEN | 7823 | 6258 | 1565 | 1 | 20 | 28 |
| N8_4_TOKEN | 5032 | 4026 | 1006 | 2 | 74 | 28 |
| N8_6_TOKEN | 2961 | 2369 | 592 | 2 | 17 | 28 |
| N8_12_TOKEN | 5343 | 4274 | 1069 | 3 | 38 | 28 |
| N8_17_TOKEN | 14431 | 11545 | 2886 | 3 | 9 | 28 |
| N8_20_TOKEN | 17334 | 13867 | 3467 | 28 | 299 | 28 |
| N8_31_TOKEN | 27840 | 22272 | 5568 | 45 | 60 | 28 |

모든 행은 Prompt/MRoPE, 실제 Forward의 Prepared Input, Prefix 전송, 전체 Prompt KV와 Greedy Output을 검사했다.
긴 문서 내부 Split은 이미지 45개와 Input27840 Tokens를 사용했다.
긴 Output299 Tokens도 임의로 자르지 않았다. Teacher와 Token 모두 Natural EOS로 종료했다.

## 남은 Gate

Stock Compiled 경로의 Numerical 정합성 설명과 품질 검증은 미완료다.
Mixed Batch, Full Main100 Token C3, Token N3 성능 비교는 미실행이다.
Absolute/Relative Decode Protection과 추가 Prefill Safe Window도 미검증이다.
Eager 고립 결과를 Compiled C3 성능 또는 전체 Task Quality로 대체하지 않는다.
원시 비교는 `analysis/token_CORRECTNESS_C15R2_*_comparison.json`에 있다.
대응 CSV는 `analysis/token_C15R2_correctness_cases_v0.csv`와 `analysis/token_C17_C18_numerical_controls_v0.csv`다.
모든 Artifact는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace에 속한다.
