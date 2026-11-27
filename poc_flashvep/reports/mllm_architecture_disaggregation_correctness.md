# 원본 모델 계산과 수치 정합성

## CogVLM2

Source 계약은 오직 생성된 language row를 D에 허용하는 것이다. 출력 ID가 이미지 또는 특수 토큰과 숫자로 겹쳐도 expert role은 Language로 append된다. 새 이미지 Prefill을 D로 우회할 수 없다. 두 H200에서 original P가 계산한 Full32 Prompt KV를 D로 실제 전송하고 기존 K/V tensor를 요소별 비교했다. Crop, 해상도, 원래 attention mask 및 BF16 Weight를 유지했다. 기존 xFormers Vision attention 호출에는 모든 Arm에 같은 PyTorch SDPA 대응식을 사용했다.

정합성 대조는 원래 resident, 직접 Language 연산의 resident, 실제 Visual-pruned, Source reachability-pruned 및 동일 직접 연산의 fresh-load A/A다. 원래 Weight coverage는 meta device에서 구성한 Model state dict와 publisher safetensors의 key 집합을 strict match로 확인한다. Image Expert 호출이 D contract를 벗어나면 precondition과 replace된 NoRows 어댑터가 오류를 낸다. Mixed B4의 distinct context와 B1/B4/B8의 prepared shape, 특수 ID를 각각 확인한다. [정확성 실험 코드](../mllm_architecture_first_disaggregation_discovery/hard_expert_decode.py)

최초 smoke의 host-default Geometry에서 같은 연산 arm 사이에도 logit 차이가 나타났다. Deterministic control과 일반 Runtime은 측정 조건을 따로 기록했다. Natural 48-token smoke는 EOS에 도달한 것과 probe 상한을 구분한다. 이 probe에서 동일한 continuation을 얻어도 확장 Serving 품질이나 채점 비열등성을 대신할 수 없다.

Deterministic control의 실제 12개 Prompt는 2,318–2,554 rows였다. 두 GPU 사이의 768/768 K/V Tensor가 같았다. 원래 resident, fast resident, visual-pruned, generic-pruned의 12/12 B1 Teacher logits와 Final Full32 KV가 일치했다. Distinct prompt로 구성한 ragged B4의 16/16 teacher argmax와 특수 Token guard도 일치했다. 이 control은 일반 Runtime의 수치 drift와 성능을 따로 기록하는 데 사용했다. [D phase 결과와 재현 자료](mllm_modality_parameter_state_liveness.md)

## IDEFICS

동일 E pixel and all-prefix 실행과 분리한 E→P 첫 logits를 먼저 맞췄다. 두 GPU에서 각 Arm이 소비한 forced token prefix와 mask는 같다. Cross K projection, K normalization, Text self-KV lifetime과 Query-dependent 계산을 각각 보존한다. 16-image prompt에 단순 compact geometry를 적용한 최초 두 반복의 logit max absolute difference는 0.140625였고 Full32 마지막 self-KV 중 12층만 bitwise 일치했다. 같은 Teacher argmax는 8/8이었다. 별도 same-shape arm으로 payload pruning의 정합성과 reduction shape의 효과를 분리한다. [지연 및 정합성 테이블](mllm_modality_phase_execution_graph.md)

일반 BF16 24개에서 Native A/A 자체가 logits 63/72회 일치했다. A/A 일치 21개 요청의 same-shape visible arms는 각각 63/63회 logits 및 2,016/2,016층 self-KV가 일치했다. Native drift가 있었던 MathVista353와 compact divergence가 있었던 16-image MMLong984에서 결정론적 통제의 48/48 시험이 모두 bitwise 일치했다. 이때 두 Native P+D 측정은 572→909 ms와 579→906 ms로 늘었다. 성능 비교에는 일반 실행을 사용했다. [Native A/A 및 별도 수치 재시험](../mllm_architecture_first_disaggregation_discovery/results/idefics/summary.json)

## ERNIE 지원 범위

설치 vLLM에서 같은 DocVQA 입력으로 하나의 고정 길이 Native와 terminal elision을 비교했다. P를 전체 Weight로 실행하고 81개 Vision-only tensor를 첫 Decode forward 직전에 내렸다. 공유 weight reference 54개와 원래 paged KV를 유지하고 Vision branch 진입 시 실패하도록 했다. 원래 Greedy ID와 12/12가 일치했다. Full logits와 전체 layer-state 및 공식 품질은 후속 검증 범위다. 이 단계에서는 실제 vLLM Native의 한 장비에서 상태를 이어 실행했다. [Greedy ID와 메모리](../mllm_architecture_first_disaggregation_discovery/results/ernie45vl/native_elision.json)
