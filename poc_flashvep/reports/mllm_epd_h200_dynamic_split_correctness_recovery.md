# H200 Late-bound split 정합성 복구

별도 BI1 Runtime의 staged Mixed-N8 두 사례가 기존 Exact Greedy 및 Full28 State 계약을 통과했다. Interior와 Image-edge 각각 실제 D Suffix 요청이 Native follower와 함께 Decoding했고 Native와 candidate의 원래 입력, E embedding hash, P→D imported KV, full prompt state와 Greedy token sequence가 모두 일치했다. 과거 BI0의 7/8 결과는 기존 판정으로 보존한다. [원자료 SHA와 정합성 CSV](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv)

| BI1 Whole-image, staged N8 | Prefix / Full input | 실제 D Suffix | Native와 공동 Decode Step | 실제 최대 공동 Decode Batch | Exact Greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image 내부 | 24,275 / 27,840 | 3,565 | 49 | 8 | 8/8 |
| Image 경계 | 15,929 / 21,193 | 5,264 | 51 | 8 | 8/8 |

두 사례 모두 paired Full28 prefix와 suffix KV에서 각각 224/224 레이어 구간이 byte-identical이었다. 각 Suffix의 전송 전후 Prefix KV가 일치했고 원래 MRoPE와 각 query 위치를 검증했다. 혼합 배치는 실제 P 종료 후 follower를 풀어 주는 staged 제어다. 대표 C3에서 자연스럽게 형성되는 Batch 분포와 Native 대비 성능은 Active Online 보고서에서 확인한다. [Staged 검증 함수](../mllm_epd_h200_modality_aware_dynamic_split/geometry_control.py)

## Slice-aware transport와 자연 경계

| 명목 Suffix | 실제 Suffix | 실제 Tokens | N1 | Mixed N8 Greedy | Native와 공동 Decode Step / 최대 D batch | 교차 이미지 D clone rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 5% | 7.22% | 1,251 | PASS | 8/8 | 60 / 8 | 경계 |
| 10% | 12.81% | 3,565 | PASS | 8/8 | 49 / 8 | 616→458 |
| 20% | 24.84% | 5,264 | PASS | 8/8 | 51 / 8 | 경계 |
| 30% | 32.03% | 3,861 | PASS | 8/8 | 32 / 6 | 630→37 |

네 자연 경계의 N1과 실제 Mixed N8이 모두 Gate를 통과했다. N8의 네 사례 각각 Native 대비 224/224 D suffix KV 레이어 구간이 byte-identical이었다. Whole-image BI1과 Slice BI1의 두 동일 Cut N8도 최종 Full28 KV, 원본 token ID와 MRoPE를 포함한 tensor 파일이 16/16에서 byte-identical이었다. [비율별 검증 및 실제 D clone](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv) [동일 Cut의 상태 Hash](../mllm_epd_h200_dynamic_correctness_slice_aware/results/same_cut_full_prompt_state.csv)

19/64/128-token Forced Cut은 모두 N1에서 Exact Gate를 통과했다. 첫 N8에서는 각각 8/8 개별 state와 Greedy 검사를 통과했고 candidate의 실제 공동 Decode는 2/4/15 steps였다. Native lead가 follower와 겹친 step은 각 0개여서 세 대조군 모두 최초 staged Mixed Gate는 미달했다. 7,823-token 입력의 19-token Suffix는 D에서 이미지 7,790행 중 마지막 1행만 clone했다. 별도 Rerun에서는 동일 Exact 기준을 유지하고 작은 Calibration follower를 활용해 두 Arm의 Decode가 겹치는지 검증한다. [D clone log에서 재구성한 row 범위와 copy-before-ACK](../mllm_epd_h200_dynamic_correctness_slice_aware/results/ratio_D_clone_rows.csv)

Natural-hold의 예약 거절 제어는 같은 P engine request의 24,275-token 경계에서 P continuation을 실행했다. Full28 KV와 Greedy 기준을 통과했고 중복 query 계산은 0이었다. 실제 Online 품질 및 성능은 별도의 데이터와 측정 Gate를 적용한다. [P continuation 실제 검증](../mllm_epd_h200_dynamic_correctness_slice_aware/results/natural_hold_continuation.csv) [비율별 실행 정의](../mllm_epd_h200_dynamic_correctness_slice_aware/ratio_gate.py)

## P에만 Batch Invariance를 적용한 대조

Slice-aware/Inline, Compiler와 CUDA Graph를 유지한 채 P만 BI1, D는 BI0으로 실행했다. 두 N1에서 동일 입력 query geometry와 원래 tensor를 검증했고 imported prefix는 28/28층 byte-identical이었다. D에서 suffix를 실행하고 완성된 KV는 P teacher 대비 두 사례 모두 0/28층만 byte-identical이었고 Greedy는 양쪽 모두 일치했다.

| P-only BI1, 실제 Mixed N8 | Greedy | 8개 요청의 paired state 통과 | Lead suffix 최대 Layer Relative L2 | Candidate 공동 Decode Step / 최대 Batch |
| --- | ---: | ---: | ---: | ---: |
| Interior 12.81% | 8/8 | 7/8 | 3.9446% | 41 / 8 |
| Image-edge 24.84% | 7/8 | 7/8 | 5.1368% | 46 / 7 |

두 staged N8 모두 당초 Layer Relative L2 ≤2% 기준을 충족하지 못했다. N1의 동일 query row와 exact import 조건에서 D suffix의 최종 KV를 비교한 결과는 P/D 실행 수치 경로를 맞춰야 하는 직접적인 근거다. [실행별 Exact 및 state 결과](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv)

## D-only 고정 FlashAttention split 대조

P/D BI1에서 D의 single-query attention만 고정 split4인 Arm을 검증했다. 두 Slice N1은 D에서 suffix를 끝낸 Full28 KV가 P teacher와 28/28층 byte-identical이었다. 자연 Greedy 길이는 Interior 56→60, Image-edge 58→10으로 달랐다. P에서 그대로 완료한 continuation에서도 최종 P KV와 D import가 각각 28/28층 exact였고 길이는 56→60이었다. 이 결과는 prompt completion 이후의 P/D 수치 경로를 원인에 포함한다. Native와 candidate가 둘 다 동일 D 산술을 사용한 Mixed N8에서는 두 fixture 각각 8/8 Greedy와 개별 state 검사를 통과했다. Image-edge의 Native lead가 follower와 겹치지 않아 staged gate는 Interior 한 건만 통과했다. [D-only split4 Exact 및 Natural 결과](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv) [P continuation 대조](../mllm_epd_h200_dynamic_correctness_slice_aware/results/natural_hold_continuation.csv)

P와 D에 같은 single-query split4를 적용하자 두 N1의 기존 Exact Gate가 통과했다. Interior와 Image-edge의 P teacher와 D Natural Greedy는 각각 60/60, 10/10으로 일치했고 같은 P request continuation도 60/60이었다. Image-edge Mixed N8은 Exact Gate를 통과했다. Interior는 8/8 prompt state 검사와 실제 최대 공동 Batch8을 확보했고 Greedy는 7/8이었다. Native-only follower의 paired Full28 prompt KV가 정확히 같은 상태에서 index 26에 Token 1817→892로 처음 갈라졌고 길이는 60→69였다. 두 D Prepared Batch가 각각 3이고 Graph padding이 각각 4인 시점에서 target row는 1→2로 바뀌었고 동시 요청의 KV context도 달랐다. Fixed split4 결과의 검증 범위는 이 설정에서 통과한 Image-edge fixture다. [P/D 대칭 split4 비교](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv)

## 대칭 Eager와 hold/resume 통제

첫 Eager N1 10/20% 시험에서 prompt KV 28/28, 원래 MRoPE 및 Greedy가 일치했다. 두 fixture에서 7개 embedding 검사만 실패했다. 기존 BatchPrefix observer는 forward 이후 입력 view를 hash했다. Eager worker는 vllm_c의 in-place fused residual kernel을 사용한다. 후속 실행의 observer는 원래 forward 이전 GPU stream에서 입력을 복제했다. 실제 후속 12개의 Prefill chunk에서 모두 original forward 전후 embedding view가 바뀌었고, pre-forward 복제를 사용한 4개 N1의 입력 hash와 최종 KV·Greedy가 동일 기준을 통과했다. CPU 회귀 시험은 이 in-place 변경을 재현한다. [기존 및 수정 observer의 실제 GPU receipt](../mllm_epd_h200_dynamic_correctness_slice_aware/results/eager_embedding_probe.csv)

| Eager Slice, 명목 Suffix | 실제 Suffix | N1 exact KV 및 Greedy | Mixed-N8 paired state | Mixed-N8 Greedy | Candidate 공동 Decode Step / 최대 Batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 7.22% | PASS | 8/8 | 8/8 | 74 / 8 |
| 10% | 12.81% | PASS | 8/8 | 8/8 | 45 / 8 |
| 20% | 24.84% | PASS | 8/8 | 8/8 | 6 / 8 |
| 30% | 32.03% | PASS | 8/8 | 5/8 | 13 / 3 |

30% fixture에는 실제 Split lead와 두 Native-only follower에서 Greedy 차이가 발생했다. 이 비율은 Eager Mixed Gate 미통과로 남는다. [N1, Mixed-N8 및 실제 D transfer](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv)

기존 E+P serial teacher와 Eager 예약 거절 P continuation의 strict Greedy는 60→56으로 다시 미달했다. 별도 진단에서 동일 입력의 unmodified Native EPD와 실제 hold/resume은 모두 56 tokens를 생성했고 전체 token ID가 동일했다. E embedding hash, P query rows [8192, 7891, 8192, 3565], D의 처음 single-row prompt completion과 이후 Decode geometry가 모두 일치했다. P와 D에서 각각 Full28 prompt KV, MRoPE 및 캡처된 logits를 담은 tensor 파일까지 동일했다. 이 경우 hold/resume은 실제 Native EPD의 상태와 출력을 보존했다. 기존 serial teacher Gate 결과도 같은 표에 보존한다. [Serial teacher와 Native EPD 대조](../mllm_epd_h200_dynamic_correctness_slice_aware/results/natural_hold_continuation.csv)
