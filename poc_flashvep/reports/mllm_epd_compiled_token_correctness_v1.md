# Common Compiled Token Split 정합성 — 독립 재시작 확인

새 공통 Compiled 설정에서 15개 Case가 두 독립 초기화 모두 통과했다.
C30과 C30R1은 각각 15/15다. 증거 수준은 `CORRECTNESS_ONLY`다.
Stock 기본 설정의 미통과 결과는 보존했다. 새 결과가 이전 실패를 삭제하지 않는다.

## 계산 계약

모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
Qwen2.5-VL-7B, Original BF16, BI0, Greedy, Natural EOS Cap2048을 유지했다.
GPU4/5/6은 E/P/D다. P/D B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
P/D는 같은 Compiled 설정을 사용했다. E 경로와 원본 모델 Forward는 변경하지 않았다.
공통 Inductor 설정은 아래와 같다.

```json
{"dynamic_scale_rblock": false, "deterministic": true, "combo_kernels": true, "benchmark_combo_kernel": false}
```

이 설정은 BI1 전환이 아니다. Batch 사이의 완전 불변성을 보장하지 않는다.
기존 Stock Native 기준선도 변경하지 않는다. 이후 비교에는 별도 Common-policy Native가 필요하다.

## 원인과 수정 범위

기존 첫 Layer Attention과 O Projection은 일치했다. 그 뒤 RMSNorm 출력에서 작은 차이를 관측했다.
실제 P/D Reduction Tile은 각각 2048과 4096이었다.
`dynamic_scale_rblock=False`만으로 이 선택을 완전히 고정하지 못했다.
설치된 Torch의 초기 Reduction 후보에도 2048과 4096이 함께 있었다.
CPU에서 실제 Heuristic 호출로 두 후보를 재현했다. Deterministic 설정은 후보를 2048 하나로 제한했다.
C24의 초기 개선은 C28에서 유지되지 않았다. 이를 완전한 수정으로 판정하지 않는다.
첫 C29 초기화는 Deterministic 설정과 Fusion Benchmark의 비호환으로 실패했다.
실제 Torch Scheduler 경로의 CPU 검사로 같은 오류를 재현했다.
수정은 Combo Kernel을 유지하고 해당 Benchmark만 비활성화한다.
설치된 라이브러리, Assertion, Attention 계산을 변경하지 않았다.
C29R1 N1에서 P 반복, D Full Teacher 두 번, Token D가 모두 28/28 KV Strict Match였다.
이는 새로운 공통 Compiler 계약의 효과다. 단일 Operator의 독립 성능 효과를 뜻하지 않는다.

## 두 초기화의 Case 결과

각 Case의 Actual P Prefill / D Suffix / D Decode Batch는 1/1/1이다.
N8 표기는 서로 다른 요청 8개다. 동시 Batch8 검사가 아니다.
Teacher는 같은 Query Chunk 경계의 원본 Full Prefill이다.
각 Case는 Prompt IDs, MRoPE, Prepared Input, 실제 NIXL Prefix, 전체 Prompt KV와 생성 Token을 검사했다.
모든 28개 Layer가 Byte 단위로 일치했다. CPU Tensor 복사 때문에 모든 Timing은 무효다.

| Case | Input | P Prefix | D Suffix | Images | Output | C30 | C30R1 |
|---|---:|---:|---:|---:|---:|---|---|
| N1_inside_single_image_TOKEN | 4831 | 2415 | 2416 | 1 | 16 | PASS | PASS |
| before_visual_TOKEN | 2961 | 15 | 2946 | 2 | 17 | PASS | PASS |
| inside_visual_TOKEN | 2961 | 1360 | 1601 | 2 | 17 | PASS | PASS |
| after_visual_TOKEN | 2961 | 2706 | 255 | 2 | 17 | PASS | PASS |
| between_images_TOKEN | 2961 | 2708 | 253 | 2 | 17 | PASS | PASS |
| text_tail_TOKEN | 2961 | 2953 | 8 | 2 | 17 | PASS | PASS |
| long_document_inside_image_TOKEN | 27840 | 13919 | 13921 | 45 | 60 | PASS | PASS |
| N8_0_TOKEN | 4831 | 3865 | 966 | 1 | 16 | PASS | PASS |
| N8_1_TOKEN | 7823 | 6258 | 1565 | 1 | 20 | PASS | PASS |
| N8_4_TOKEN | 5032 | 4026 | 1006 | 2 | 74 | PASS | PASS |
| N8_6_TOKEN | 2961 | 2369 | 592 | 2 | 17 | PASS | PASS |
| N8_12_TOKEN | 5343 | 4274 | 1069 | 3 | 38 | PASS | PASS |
| N8_17_TOKEN | 14431 | 11545 | 2886 | 3 | 9 | PASS | PASS |
| N8_20_TOKEN | 17334 | 13867 | 3467 | 28 | 275 | PASS | PASS |
| N8_31_TOKEN | 27840 | 22272 | 5568 | 45 | 60 | PASS | PASS |

이미지 내부, 이미지 직전/직후, 이미지 사이, Text Tail, 45-Page 문서 내부 경계를 포함했다.
Calib20은 새 공통 설정에서 Teacher와 Token 모두 Natural Output275 Tokens였다.
이전 Eager 경로의 299 Tokens와는 다른 Numerical 계약이다. 두 길이를 합쳐 평균하지 않는다.

## 남은 Gate

Task Quality는 `PENDING`이다. Strict 고립 정합성은 공식 Task 품질 검사를 대체하지 않는다.
이 Artifact는 동시 Mixed Batch 또는 Main100 Token C3의 통과를 입증하지 않는다.
Decode-safe Headroom, Native 대비 Decode 보호, Token Online Speedup도 이 Artifact에서 미측정이다.
NativeStrong과 공통 설정 Native를 구분해 후속 Runtime 비용을 확인해야 한다.
대응 CSV는 `analysis/compiled_token_C30_two_start_cases_v0.csv`다.
요약과 Source Hash는 `analysis/compiled_token_correctness_checkpoint_v0.json`에 있다.
