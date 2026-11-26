# H200 Native Greedy 출력의 수치 재현성

직전 Main100 QPS5 Native A/A에서 같은 서버 Start, 동일 Arrival Trace와 Natural EOS 조건으로 B8은 86/100, Free는 88/100 요청의 전체 Greedy token ID가 일치했다. 두 Start 내 반복은 각각 독립 Start 한 개의 증거다. [직전 Native A/A 원자료 요약](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_native_aa_summary.csv)

Clean client 원자료에서 처음 달라진 token index는 B8의 14건에서 최소 0, 중앙값 45, 최대 507이었다. Free의 12건에서는 각각 3, 41, 378이었다. 7개 요청이 두 조건 모두에서 달랐고, 각 조건에서 길이가 동일해도 토큰 ID가 달라진 요청이 1개씩 있었다. B8에는 첫 token부터 달라진 사례가 있으므로 첫 출력 이전 단계도 원인 후보에 포함된다. 이 Clean pair에는 요청별 실제 P/D Geometry와 E embedding hash가 없다. [원자료 SHA 및 요청별 일치 여부](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_aa_requests.csv)

직전 staged N8의 Image-edge Native-only follower는 P query [2318, 2714]에서 [5032]로, D 최초 차이의 Prepared Batch는 2에서 3으로 변했다. 두 Arm의 E embedding byte와 NIXL prefix가 일치했고 Zero-based index 23에서 Greedy ID 9666→32719가 관측됐다. 당시 P/D Geometry가 함께 바뀌었으므로 각 Stage의 독립 효과는 추가 통제가 필요하다. [직전 staged N8 자료](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_natural_lead_geometry.json)

## P/D Batch Invariance 개입

H200의 기존 vLLM 설치에서 P/D에만 VLLM_BATCH_INVARIANT=1을 적용하는 대칭 Arm을 추가했다. E는 BI0, 모델·BF16·B8192·S1024·KV42GiB·Block128·Decode Cap8은 기존 계약을 유지했다. 실제 P/D worker는 FlashAttention과 컴파일 및 CUDA Graph 경로를 시작했다. 설치된 vLLM은 BI1에서 batch-invariant linear 경로를 선택하고 FlashAttention의 cascade 및 split selection을 제어한다. 이 개입의 단위는 P/D 수치 경로 전체다. [vLLM Batch Invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/)

| 두 staged N8 Arm | 첫 BI0 exact Greedy | BI1 exact Greedy | BI0 / BI1 최종 D suffix KV byte equal |
| --- | ---: | ---: | ---: |
| Interior lead | 7/8 | 8/8 | 116/224 / 224/224 |
| Image-edge lead와 Native-only follower | 7/8 | 8/8 | 8/224 / 224/224 |

BI1에서 Native와 late의 P/D Prepared Geometry는 독립적으로 달랐다. E embedding은 두 Arm의 각 8개 요청에서 원본 byte hash가 일치했다. 실제 NIXL prefix와 최종 D KV의 전 28층이 요청마다 byte-identical이었고 모든 Greedy sequence가 일치했다. 이 관측은 해당 두 사례의 수치 차이를 제어하는 Runtime 방법이 있음을 보인다. [원자료 SHA와 staged CSV](../mllm_epd_h200_dynamic_correctness_slice_aware/results/strict_geometry.csv)

## Native A/A와 동일 token prefix

Slice BI1의 두 독립 정확성 fixture에서 각 Serial Native A/A는 8/8 exact로 paired P/D query geometry까지 일치했다. 각 Mixed Native A/A 역시 8/8 exact였고 8개 요청 모두 전체 P/D geometry 중 적어도 한 항목이 달랐다. 동일 staged arrival에서 실제 Slice와 Native를 비교한 Natural EOS도 두 fixture 각각 8/8이었다. 각 contrast에서 E embedding은 요청별 byte hash를 확인했다. [요청별 P/D geometry 및 A/A 비교](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_aa_geometry.csv)

Teacher forcing은 역사적 최초 차이 Step까지 두 Arm에 공통 token prefix를 넣었고 실제 입력 Token ID와 원래 sampler argmax를 확인했다. Interior는 0–13의 14/14 step, Image-edge Native-only follower는 0–23의 24/24 step에서 전체 vocabulary logit과 최종 hidden state가 byte-identical이었다.

| 역사적 차이 표적 | Zero-based step | BI1 Native / Slice 실제 D batch 및 패딩 | 상위 두 logit | Raw margin | 두 Arm의 Greedy ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interior lead | 13 | 1/1 및 2/2 | 33.75, 33.75 | 0 | 13 |
| Image-edge Native-only follower | 23 | 3/4 및 3/4 | 19.625, 19.5 | 0.125 | 32719 |

Interior의 동일 최고 logit에 해당한 Token ID는 1447과 13이었다. 이는 과거 두 Arm에서 실제로 갈라진 ID와 일치한다. 결과의 범위는 이 P/D 수치 개입과 두 frozen fixture다. 개별 P/D kernel의 독립 효과를 위해 stage-only BI 및 Eager hook 조건을 분리해 둔다. Serving latency는 별도의 clean Natural EOS에서 계측한다. [원자료 SHA 및 각 Step의 logit/geometry](../mllm_epd_h200_dynamic_correctness_slice_aware/results/teacher_divergence.csv) [진단 구현](../mllm_epd_h200_dynamic_correctness_slice_aware/recovery.py)

## Native QPS5 성능 비용

| NativeCommon-B8 Main100 Natural | TTFT p95 (s) | ITL p99 (ms) | TPOT p95 (ms) | 모든 ITL ≤34 ms | SLO-goodput (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 이전 BI0 Clean Before | 12.029 | 13.682 | 5.247 | 94% | 83.504 |
| 이전 BI0 Clean After | 12.516 | 12.431 | 5.187 | 93% | 64.054 |
| 이번 대칭 BI1 Clean | 17.732 | 31.881 | 22.677 | 74% | 20.539 |
| 이번 대칭 Eager Clean | 13.144 | 26.062 | 13.532 | 78% | 40.949 |

각 코호트의 100건이 완료했고 모든 요청이 TPOT 기준을 통과했다. 대칭 BI1과 Eager의 개별 ITL 34 ms 조건 충족률은 각각 74%, 78%로 Primary Absolute Decode SLO 95%에 미달했다. 과거 BI0 값은 캠페인 간 참고점이다. Eager의 별도 batch 관측 코호트에서는 Long-document 요청 한 개가 2,048-token cap에 도달했고 총 출력이 clean 5,379에서 7,244 tokens로 달라졌다. 이 관측은 Eager Natural EOS 부하의 변화를 보여주며 원인 분리에는 추가 대조가 필요하다. [Native QPS5 원자료 요약](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_numeric_cost.csv)

## P-only BI1 대조

P만 BI1로 실행한 경우 두 Serial Native A/A 모두 8/8로 일치했고 두 Mixed Native A/A는 각각 7/8이었다. Image-edge의 순수 Native Mixed A/A에서 Mantis follower의 E는 byte-identical, P의 해당 요청 query rows는 두 Arm 모두 [5032]였다. 실제 첫 차이 index 18에서 D Prepared Batch와 graph row는 2/2에서 1/1로 달라졌고 target 위치도 1에서 0으로 바뀌었다. 그 이전의 P 동시 요청 geometry도 달랐다. [P-only A/A와 실제 geometry](../mllm_epd_h200_dynamic_correctness_slice_aware/results/native_aa_geometry.csv)

P-only BI1의 동일-prefix teacher forcing은 Image-edge follower의 역사적 index 23과 경쟁 Token ID 두 개를 다시 확인했다. 이번 Native와 Slice에서는 각각 32719와 9666 선택이 관측됐으며 원래 로그의 Arm 순서와 반대였다. Raw top-two margin은 각각 0.125와 0, 전체 logit relative L2는 1.0122%였다. Interior는 14개 step에서 전체 logit이 달랐고 역사적 index 13의 argmax는 양쪽 모두 13이었다. 이 결과는 실제 동일 prefix에서 작은 margin과 수치 경로의 변경이 Greedy 선택을 바꿀 수 있음을 보여준다. [P-only teacher-forced logit 및 배치 자료](../mllm_epd_h200_dynamic_correctness_slice_aware/results/teacher_divergence.csv)

## 대칭 Eager의 레이어별 대조

P와 D의 CUDA Graph가 NONE인 Eager에서도 Interior는 공통 실제 token prefix의 index 13에서 13→1447로 갈라졌다. Raw top-two margin은 0→0.25, 실제 D Prepared Batch는 2→1이었다. 역사적 BI0와 동일한 index·경쟁 ID이며 이번 Arm의 선택 순서는 반대였다. Eager Image-edge follower의 역사적 index 23까지 Greedy는 유지됐고 해당 step의 margin은 0.125→0이었다. [Teacher forcing, Raw logits 및 Prepared Geometry](../mllm_epd_h200_dynamic_correctness_slice_aware/results/teacher_divergence.csv)

Interior index 13에서 layer 0의 Q/K/V projection은 전부 exact였다. 측정한 중간 tensor 중 이후 처음 달라진 것은 RoPE를 거친 Q와 K였다. Relative L2는 각각 0.3491%와 0.02268%였다. Layer 0 attention output·projection·MLP output은 exact였고 layer 1 attention output부터 차이를 보였다. 최종 hidden/logits relative L2는 각각 0.8306%와 0.3992%였다. Eager 레이어 관측은 차이가 처음 보인 연산 경계를 제시한다. 해당 step의 RoPE 입력 position과 실행 shape를 함께 고정한 kernel 대조는 후속 검증 범위다. [28개 layer별 Q/K/V, attention 및 MLP 자료](../mllm_epd_h200_dynamic_correctness_slice_aware/results/teacher_layer_divergence.csv)
