# Encoder Batch 수치 차이 — Frozen Fraction 정합성 감사

증거 수준은 `CORRECTNESS_ONLY`, 저장 상태의 `TRACE_ANALYSIS`, 통제된 Numerical `DIAGNOSTIC`이다.
이 결과는 C3 Speedup이나 Task Quality PASS가 아니다. 기존 실패와 허용 오차를 변경하지 않았다.

## C36: 동결한 후보의 정합성 검사

Discovery에서 TTFT 기준으로 선택한 10%와 Goodput 기준 25%를 검사했다.
10%는 N1 및 동시 N8에서 모든 28 Layers KV와 전체 Greedy Output이 Byte-exact였다.
25%는 N1을 통과했다. 동시 N8의 Mantis cw_9에서 Prefix/Suffix KV 허용 오차를 초과했다.
원래 검사는 첫 실패에서 중단했다. 아래 표는 보존된 8개 실제 실행을 CPU에서 모두 다시 비교한 결과다.
GPU 실행을 반복하지 않았다. 원래 FAILED_EXECUTION 기록은 보존했다.

| Calib | Input | Strict KV Layers | Tolerance Layers | Max Prefix L2 | Max Suffix L2 | Output Tokens | Strict Output |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 5343 | 28/28 | 28/28 | 0.000000 | 0.000000 | 38 | True |
| 0 | 4831 | 28/28 | 28/28 | 0.000000 | 0.000000 | 16 | True |
| 31 | 27840 | 28/28 | 28/28 | 0.000000 | 0.000000 | 60 | True |
| 1 | 7823 | 28/28 | 28/28 | 0.000000 | 0.000000 | 20 | True |
| 20 | 17334 | 28/28 | 28/28 | 0.000000 | 0.000000 | 275 | True |
| 4 | 5032 | 28/28 | 28/28 | 0.000000 | 0.000000 | 74 | True |
| 6 | 2961 | 0/28 | 2/28 | 0.305581 | 0.386528 | 17 | True |
| 17 | 14431 | 28/28 | 28/28 | 0.000000 | 0.000000 | 9 | True |

실패 요청의 P→D Prefix Import는 28/28 Layers에서 Byte-exact였다. Token IDs와 MRoPE도 일치했다.
Layer0에서는 Visual2916 Rows만 달랐다. Text/Special45 Rows는 일치했다.
P Query는 0→15→2221이었다. Teacher Query는 0→2221→2961이었다.
10% 검사에서도 0→15의 추가 Chunk가 있었지만 모든 KV가 일치했다.
따라서 추가 15-token Chunk만으로 이 실패를 설명할 수 없다.

## C38R1: Encoder Batch만 변경한 통제 실험

원본 Target은 같은 Mantis cw_9의 실제 이미지 2개다. 원본 Companion은 Calibration17의 이미지 3개다.
Solo는 Target2개를 처리했다. Joint는 동일 Target2개와 Companion3개를 하나의 Encoder Envelope에서 처리했다.
Joint Envelope는 진단용이다. Main100 요청이나 Prompt를 변경하지 않았다.
Original BF16, 원본 Vision/Processor/Weight/Attention과 GPU4 E/GPU5 P를 유지했다.
Target의 실제 Processor Pixel Hash와 Grid를 검사했다. LLM Prompt, MRoPE, Query0→2221→2961을 고정했다.
각 Geometry를 두 번 실행했다. Solo 반복과 Joint 반복 모두 Embedding 및 28/28 KV가 Byte-exact였다.
CPU Tensor Capture가 있으므로 모든 Timing은 성능 평가에서 제외했다.

| Comparison | Target Image | Encoder Images/Call | Embedding Relative L2 | Same Greedy Output |
|---|---:|---|---:|---|
| Solo_B vs Solo_A | 0 | 2→2 | 0.000000 | True |
| Solo_B vs Solo_A | 1 | 2→2 | 0.000000 | True |
| Joint_A vs Solo_A | 0 | 2→5 | 0.104151 | True |
| Joint_A vs Solo_A | 1 | 2→5 | 0.085062 | True |
| Joint_B vs Solo_A | 0 | 2→5 | 0.104151 | True |
| Joint_B vs Solo_A | 1 | 2→5 | 0.085062 | True |

Joint의 Full-prompt KV 최대 Relative L2는 0.322317였다.
Encoder Batch Geometry만 바꿔도 같은 Pixel에서 Embedding과 후속 KV 차이가 발생했다.
이는 통제 실험에서 재현한 인과 효과다. 정확한 Operator별 기여는 측정하지 않았다.
C36 당시 Encoder Geometry는 기록하지 않았다. 따라서 C36 실패의 직접 원인은 아직 확정하지 않았다.
이 효과를 Prefix Transfer 오류 또는 Token Split 자체의 근사 계산으로 해석하지 않는다.
반대로 출력이 같다는 이유로 KV 오차 실패를 PASS로 바꾸지 않는다.
25%는 Primary 후보에서 제외한다. 10%의 통과 범위도 검사한 요청과 Geometry로 한정한다.

## Harness 오류와 수정 범위

초기 C38은 P가 독립 Processor 식별자를 생성해서 원본 E Export를 찾지 못했다.
Strict EC는 Local Vision Fallback 없이 중단했다. 해당 실패를 보존했다.
새 진단 전용 Handoff는 원본 Target Pixel/Token/Grid를 보존하고 실제 E Export 식별자에 연결한다.
Target KV나 Output을 수정하지 않는다. 기존 Main Serving 경로는 변경하지 않았다.
실제 Handoff 경로의 CPU Regression은 수정 전 실패하고 수정 후 통과했다.
전체 CPU Regression은 139개 통과했다. 관련 Red/Green Log를 보존했다.

## 한계와 재현

Task Quality는 PENDING이다. MMLongBench-Doc 공식 외부 답변 추출은 승인 없이 실행하지 않았다.
이 실험으로 전체 BI0 Batch Invariance나 안전한 Active-decode 삽입을 보장하지 않는다.
원본 C36/C38/C38R1 Raw, Config, Source ZIP/SHA256, GPU UUID와 Failure Log를 보존했다.
재계산 명령은 `python -m poc_flashvep.mllm_epd_prefill_heavy_token_split.vision_batch_intervention_report`다.
Derived CSV/JSON은 `analysis/frozen_fraction_C36_encoder_C38_*_v0`에 있다.
Figure는 `figures/encoder_batch_C38_numerical_effect_v0.{png,pdf}`다.
Figure 좌측/우측 수치는 각각 `encoder_images`와 `controlled_KV` CSV에 대응한다.
