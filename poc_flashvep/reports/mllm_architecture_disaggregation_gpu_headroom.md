# H200의 Architecture-specific Headroom

두 후보 모두 같은 물리 GPU5/6의 NV18 연결에서 원본 모델 가중치와 실제 Discovery200 입력을 사용했다. CogVLM2는 Decoder Weight를 메모리에 올리지 않는 효과를 직접 측정한다. IDEFICS는 초기 Cross-attention state의 P→D 배치와 이후 원래 계산의 비용을 측정한다. Host enclosure, 실제 할당량과 논리 payload를 구분한다. [환경과 동작 경계](mllm_modality_phase_execution_graph.md) · [실행 Ledger](../mllm_architecture_first_disaggregation_discovery/results/execution_ledger.json)

## CogVLM2-19B

| Decode 구성 | D Weight (GiB) | 실제로 확보한 KV 저장 slot |
|---|---:|---:|
| Resident, direct language route | 27.958 | 120 |
| Scope-guarded visual expert omission | 14.958 | 163 |
| Generic phase reachability | 14.958 | 163 |

같은 64 GiB **weights+KV** 저장 계약에서 Full32 BF16 KV당 2,322 prompt rows와 128 output reserve를 둔 동일 layout이다. 실제 contiguous CUDA buffer에 쓰고 동기화하여 120→163, +35.8%를 측정했다. 양쪽에서 원래 Attention에 필요한 Full32의 Vision-origin KV까지 전달했다. 192개 Vision QKV/O/MLP tensor를 D에서 13.0005 GiB 줄인다. P는 원본 약 36.33 GiB를 유지한다. 동일 Fast D에서는 제거 전부터 그 Weight를 참조하지 않으므로 가중치 로딩 절감과 Token당 커널 단축은 별도 효과다. [가중치 및 KV 수명](mllm_modality_parameter_state_liveness.md) · [측정 데이터](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/smoke.json)

12개 실제 Prompt에서 Full32 전달 payload는 303,824,896–334,757,888 B였다. 일반 실행의 호스트 copy enclosure는 중앙값 40.24 ms, 범위 3.75–498.59 ms로 allocator와 동기화를 포함한다. 같은 조건에서 2,554-row Prompt를 기준으로 실제 KV 저장은 110→149 slot이었다. [일반 실행 Raw](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/profile_ordinary.json)

120과 163은 KV 저장 capacity다. 현재 HF eager 경로는 새 K/V가 생성될 때 기존 tuple과 새 tuple을 함께 유지하며 GQA 확장, 작업공간 및 allocator overhead도 사용한다. 같은 Weight와 KV 저장 Budget의 실제 Serving은 page allocator, Batching 및 admission 조건에 달려 있다. 향상 가능성이 있는 대상은 Decoder Capacity이며, GPU 수 감소나 출력 Throughput의 증가는 별도 검증이 필요하다.

## IDEFICS-9B

이미지당 Raw Perceiver는 163,840 B, 투영된 8층 K/V는 8,388,608 B다. 초기 전송량은 P에서 투영해 보내면 51.2배가 된다. D에서 한번 계산하고 cache하는 비교군은 이후의 D Reprojection을 이미 제거한다. Decode가 반복하는 마지막 Mask의 집합만 P에서 보내면 두 Transport 모두 Image Slot 수만큼 payload를 줄인다. D가 원래 SDPA geometry를 유지해야 하는 조건도 별도 arm으로 검증했다. [실제 전송과 Geometry 대조](mllm_modality_phase_execution_graph.md)

비용 경계는 Suffix Token마다 현재 Hidden State의 Q 및 Cross-attention으로 이어진다. 실험은 같은 Decode GPU에 active K/V를 한 번 전달한 뒤 재사용한다. 외부 서버와 매번 통신하는 구조는 실험 범위에 포함하지 않았다.

일반 BF16의 24개 × 8 arms × 3 repeats × 32 teacher steps에서 P-projected 전체 handoff의 P+D 시간은 D packed projection 대비 중앙 비율 1.0009였다. 동일한 visibility와 원래 D geometry를 맞춘 projected handoff와 raw+D projection의 비율은 전체 0.9988, 10개 multi-image에서 0.9970이었다. 이 규모의 차이는 시스템 후보 채택의 15%를 밑돈다. 실제에서 raw 16→1 slots는 전송 payload 2.5 MiB→0.156 MiB로 줄었다. D의 원래 shape를 유지해야 하는 경우 캐시 padding을 복원한다. [의존성과 수치 대조](mllm_modality_phase_execution_graph.md) · [실측 손익분기 Figure](../mllm_architecture_first_disaggregation_discovery/figures/fig07_idefics_handoff_break_even.png)

## ERNIE-4.5-VL의 Native 독립 검증

고정 Thinking checkpoint에서 Native vLLM BF16/Triton으로 원래 이미지 질문 4개와 text-only 대조 1개를 실행했다. Image prompt 512–4,867 tokens에서 Visual FFN bank는 요청당 27회, 그 뒤 실제 44개 Decode forward에서 0회 호출됐다. 두 bank가 공유하는 FFN 및 동일 self-attention KV를 보존했다.

마지막 DocVQA 동일 요청의 두 번째 실행에서 Image Prefill이 끝난 뒤 첫 D forward 직전에 81개 Visual FFN/router tensor에 대한 GPU Parameter binding을 제거했다. Shared reference 54개는 유지했다. Torch live allocation은 67.013→54.341 GiB로 감소했고, 제거량 13,607,239,680 B는 사전에 확인한 Checkpoint의 role payload와 정확히 같았다. Greedy 생성 12/12 ID가 같았다. 단일 H200의 terminal diagnostic이다. Startup의 D-only loading, 자체 GPU memory pool의 reserved 크기 및 native PD capacity는 별도 경계다. [Native trace](../mllm_architecture_first_disaggregation_discovery/results/ernie45vl/native.json) · [Live elision](../mllm_architecture_first_disaggregation_discovery/results/ernie45vl/native_elision.json)
