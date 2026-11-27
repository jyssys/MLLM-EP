# Architecture-first MLLM Disaggregation · H200

실제 Modality 조건이 가중치 사용과 상태의 수명을 바꾸는 경계를 확인했다. **CogVLM2는 `MEMORY_CAPACITY_ONLY`**, **IDEFICS는 `GENERIC_OPTIMIZATION_ONLY`**, **ERNIE는 `ARCHITECTURE_OPPORTUNITY_CONFIRMED`**다. Cog에서 D Weight 13.0 GiB를 줄이고 같은 64 GiB 저장 예산에서 KV slot을 120→163으로 늘렸으며, 강한 Generic도 같은 결과를 냈다. ERNIE Native의 단일 terminal probe에서는 Visual FFN 12.67 GiB를 실제 내리고 같은 Greedy 12/12 ID를 얻었다. IDEFICS의 24개 paired GPU profile에서 강한 D local cache에 대한 P-projected 전달의 추가 P+D 지연 이득은 확보되지 않았다. [실행 경계](mllm_modality_phase_execution_graph.md) · [강한 대조](mllm_architecture_disaggregation_strong_baseline_ablation.md)

## Architecture Discovery

1. **검토 범위.** 고정 Checkpoint 14개와 논문 기반의 추가 후보를 검토했다. 구조는 hard modality routing, masked adapter, cross-attention, staged visual encoder, parallel vision tower, task-specific generation bank, generic/adaptive MoE, shared decoder로 나뉜다. Conditional depth의 공개 호환 Checkpoint 여부는 별도 표시했다. [후보 Registry](mllm_modality_dependent_architecture_landscape.md)
2. **실제 다른 계산.** CogVLM2는 Vision/Text별 QKV·O·MLP 가중치를 선택한다. ERNIE-VL은 QKV를 공유하고 row type에 따라 서로 다른 FFN expert pool을 선택한다. IDEFICS-1은 별도 Cross-attention Block이 고정 Perceiver State를 읽는다. [동작 그래프](mllm_modality_phase_execution_graph.md)
3. **구조적 차이.** CogVLM2와 ERNIE는 가중치 도달 가능성이 다르다. IDEFICS는 text self-KV와 visual state의 수명 및 접근 집합이 다르다. DeepSeek-OCR2의 asymmetric visual causal-flow mask는 E 내부 의존성이 다르다. 후자의 GPU 최적화는 소스 탐색 범위다.
4. **실행 증거.** CogVLM2 Full32 P→D, IDEFICS Full32와 8개 Cross Block을 실제 NV18 H200 pair에서 실행했다. 독립 Native ERNIE의 실제 이미지 4개가 P의 Visual FFN 27회를 각각 호출했고, 이어진 44개 Text Decode의 Visual FFN 호출은 0회였다. [실행 그래프](mllm_modality_phase_execution_graph.md)
5. **선정 이유.** CogVLM2와 IDEFICS는 서로 다른 구조적 경계를 가진 공개 가중치로 두 GPU Exact PoC를 만들 수 있었다. ERNIE는 공유 QKV와 별도 FFN을 사용하는 추가 독립 계열이다. 접근이 제한된 Mllama, 이미 범위 밖 생성 Weight를 배제하는 BAGEL, 학습된 실제 공개 checkpoint가 확인되지 않은 일부 dynamic depth, 원래 Conditional dispatch가 통계적 선호도에 그치는 일반 MoE를 제외 또는 source control로 남겼다. [후보 선정](mllm_architecture_candidate_selection.md)

## Novelty

6. **기존 EPD와 Parallelism.** EPD와 REDServe는 큰 Stage와 자원 배치를 다루고, ModServe는 Cross-attention 계열도 다룬다. CogVLM2의 차이는 Encoder를 이미 배제한 D에서 Decoder 내부의 image-row 전용 weight까지 구조적으로 제거할 수 있다는 점이다. IDEFICS의 immutable K/V는 기존 encoder–decoder cache의 범용 접근을 강한 대조로 사용했다. 일반 EP의 load balancing 및 TP/PP/CP의 layer/tensor placement와는 다른 축이지만 조합은 가능하다. [문헌 대조](mllm_architecture_prior_work_novelty_audit.md)
7. **범용 원리.** Stage별 도달 불가능한 Weight 제거, immutable projection cache, masked source packing 및 state lifetime optimization은 Conditional AR 모델과 Encoder–Decoder에도 적용된다.
8. **강한 Multimodal 구조.** Image token role은 실제 expert bank의 활성 범위를 크게 바꾸면서, D의 원래 Self-attention에는 P의 시각 Prompt KV 전체가 계속 필요하다. IDEFICS의 last image mask는 생성 중 실제로 읽는 visual source slot을 줄인다.
9. **검증한 경계.** Cog는 Full32 self-KV를 전달하고 D의 visual expert residency를 끈다. IDEFICS는 P/RoPE와 원래 pending query에 필요한 shape를 유지하며 visual source 준비와 D query 반복을 나눈다. [실제 가설과 반증 조건](mllm_architecture_specific_disaggregation_hypotheses.md)
10. **중복 위험.** 다른 task weight를 제거하는 BAGEL vLLM은 기존 구현에서 이미 최적화했다. Cross-state cache와 phase reachability를 강한 일반 방식으로도 적용할 수 있다. 실제 배포 방법의 독립 신규성은 검증한 범용 최적화와의 차이로 판단했다.

## Systems Research

11. **제거한 실제 Weight와 State.** CogVLM2 D에서 192개 Tensor, 13.0005 GiB를 실제 제외했다. 실제 한 온라인 start의 D peak는 Resident 32.025→Visual-pruned 18.259 GiB였다. 동일한 Full32 self-KV는 계속 전달했다. ERNIE 단일 장비에서 Vision-only 81 Tensor를 Prefill 후에 제거하자 live allocation 67.013→54.341 GiB였다. IDEFICS raw latent는 이미지당 160 KiB, 8개 층을 미리 투영하면 8 MiB를 전달한다. [Headroom](mllm_architecture_disaggregation_gpu_headroom.md)
12. **GPU 역할.** Cog는 같은 두 GPU에서 E+mixed P와 Language-only D를 사용했다. IDEFICS는 원본 E/P를 첫 GPU, 실제 Peer copy와 current-query 계산을 둘째 GPU에 배치했다. 둘 다 동일 source의 generic execution을 비교했으며 별도의 Memory GPU를 추가하지 않았다.
13. **실제 전달과 동기화.** Cog는 Prompt당 303.8–334.8 MB를 보내고 Pilot의 three PD median이 2.11–2.21 ms였다. 격리 대조의 Host enclosure는 Allocation과 동기화 영향을 별도로 기록했다. IDEFICS는 token별 Remote Query를 막고 고정 cross-source state를 최초 1회 전달했다. [시각과 payload](mllm_architecture_disaggregation_online_pilot.md)
14. **순이익.** Cog는 같은 64 GiB에서 실제 2,322-row KV 저장 120→163, 2,554-row에서는 110→149였다. IDEFICS의 Profile P+D wall 중앙 비율은 같은 Visibility와 D geometry에서 강한 raw/D projection 대비 P projection의 비율 0.9988이었다. 16-image 사례의 raw peer payload는 2.5 MiB→0.156 MiB로 줄었고 원래 D 형상으로 BF16 결과를 유지할 수 있었다.
15. **강한 Generic.** Cog의 source reachability가 같은 192개 Weight를 제외한다. IDEFICS는 D local projection을 memoize하고 동일 관측 mask로 live state만 전달할 수 있다. 이 강한 비교를 통과한 추가 Modality Scheduling을 검증해야 독립 기여가 생긴다. [대조군](mllm_architecture_disaggregation_strong_baseline_ablation.md)
16. **실제 동시성.** Cog에서 35.5–35.8% 더 많은 Full32 KV 저장 slot을 실제 할당하고 썼다. 실제 온라인 D cap은 4, prepared max는 3–4였다. 동일 GPU 수에서 최대 동시 Active Requests 또는 Server Count 절감은 후속 검증 범위다.
17. **실제 E2E.** Cog 두 GPU, 24개, 0.5 req/s, 독립 시작 각 1회에서 TTFT p95는 DP2 616 ms, PD Resident 799 ms, Visual-pruned 908 ms, Generic 1,016 ms였다. 프로세스 내부 토큰 방출까지의 E2E p95는 23.86/16.63/16.64/16.57 s였다. P+D의 tail gap 완화는 관측했으나 역사적 joint SLO 충족 요청은 모두 2/24였다. Production HTTP와 Paged allocator는 별도의 검증 범위다. [온라인 결과](mllm_architecture_disaggregation_online_pilot.md)
18. **정확성 및 품질.** Cog의 BF16 결정론 통제에서 768/768 초기 K/V, 12/12 teacher logits/최종 Full32 K/V, ragged B4와 special-token control이 일치했다. 일반 Runtime은 최소한의 수치 차이를 보였고 Pilot의 Visual-pruned Natural output은 Resident 및 Generic과 각각 21/24 일치했다. Local diagnostic 점수는 네 arm에서 같으나 낮은 floor와 독립 표본의 제약이 있어 Official quality noninferiority는 판단 보류다. [정합성 계약](mllm_architecture_disaggregation_correctness.md)

## Autonomous Discovery

19. **새로 찾은 후보.** ERNIE-VL은 shared self-attention 위에 별도 Visual FFN pool이 있어서 Cog의 QKV 분리와 독립적으로 검증할 수 있다. BAGEL은 scope-aware loader가 이미 맡은 역할 밖의 bank를 제거하는 source-level 반증이다. OCR2의 visual encoder 내부 asymmetric mask는 별도의 의존성 경계였다.
20. **탐색 전환.** IDEFICS는 강한 D Memoization을 먼저 넣고, 16-image에서 compact geometry가 Full32 KV의 bitwise 결과를 바꾸자, 작은 slice만 이동하고 원래 SDPA Geometry를 유지하는 arm을 추가했다. Cog는 단일 smoke의 수치 변동을 deterministic와 ordinary profile로 나누고 생산형 용량에 가까운 제어와 실제 open-loop를 분리했다.
21. **한계의 원인.** Cog의 가중치 절감은 구조적이다. 시험한 Fast D에는 이미 그 Weight의 계산 비용이 없고, Eager allocator와 Historical SLO는 운영 확대를 막는 실증 범위다. IDEFICS의 projection handoff에는 51.2배의 논리 payload 확대와 동일 query 계산이 남는다. vLLM에서 IDEFICS-1과 CogVLM2는 production adapter 지원 한계가 있다.
22. **가장 유망한 기여.** Cog와 ERNIE에서 input role에 따라 도달 가능한 큰 Decoder bank가 달라지고, 원래 continuation에 필요한 KV를 그대로 두면서 Image Prefill 전용 가중치를 실제로 없앨 수 있었다. Cog는 D-only loading 및 KV 저장 headroom, ERNIE는 한 장비의 terminal live allocation을 검증했다. Serving 논문의 실용적 주장은 확보한 VRAM을 실제 paged KV와 concurrency 또는 더 작은 GPU 예산으로 바꾸고 scope-aware generic을 넘는지에 달려 있다.
23. **추가할 방향과 중단할 방향.** 가중치 도달 범위가 강하게 바뀌며 native fused MoE가 있는 ERNIE-VL의 D-only loading을 우선한다. IDEFICS의 P preprojection은 D cache와 shape-safe liveness를 기준으로 우선순위를 낮춘다. Dense decoder의 반복적인 Visual KV 왕복과 label-only Split은 기존 결과의 범위에서 보존한다.
24. **검증에 필요한 구성.** 같은 NV18 GPU 2대에서 BF16 Native PD 및 role-limited D를 비교하고, 별도 GPU 2대에서 같은 모델 DP2/TP 또는 MoE EP 대조를 반복할 수 있다. 실제 paged KV allocator, 메모리 cap과 긴 동시요청, 서로 다른 image/text arrival, exact decode 및 formal scorer를 갖춘 N3 비교가 필요하다.
25. **판정.** 실제 Modality-dependent Architecture와 memory headroom을 확인했다. 검증한 모델과 설정에서 Generic 대비 독립적인 Serving 우위는 확보되지 않았다. 후속 대상은 Phase-dependent weight loading과 Batch/KV capacity이며 생산 Serving 개선은 별도 검증에 달려 있다.

## Figures와 재현 자료

| Figure | 출력과 근거 |
|---|---|
| 1 · Architecture landscape | [소스와 체크포인트 Registry](../mllm_architecture_first_disaggregation_discovery/figures/fig01_architecture_landscape.png) |
| 2 · Module × Phase | [실행 경로](../mllm_architecture_first_disaggregation_discovery/figures/fig02_module_phase.png) |
| 3 · Weight/KV residency | [CogVLM2 물리 Weight 및 KV Allocation](../mllm_architecture_first_disaggregation_discovery/figures/fig03_cog_parameter_residency.png) |
| 4 · 실제 Execution Dependency | [CogVLM2 및 IDEFICS 실행 DAG](mllm_modality_phase_execution_graph.md) |
| 5 · GPU 비용 | [IDEFICS 실제 visual-state payload, 지연 및 수치 대조](../mllm_architecture_first_disaggregation_discovery/results/idefics/visual_cross_state_handoff.png) · [ERNIE native expert CUDA enclosures](../mllm_architecture_first_disaggregation_discovery/figures/fig05_ernie_native_expert_cost.png) |
| 6 및 8 · Headroom과 강한 대조 | [Cog 실제 KV 저장 및 범용 Reachability 대조](../mllm_architecture_first_disaggregation_discovery/figures/fig06_08_cog_headroom_ablation.png) · [IDEFICS 대조 결과](mllm_architecture_disaggregation_strong_baseline_ablation.md) |
| 7 · Cross-state 손익분기 | [실제 prompt의 같은 GPU 예산에서 본 P/D projection 차이](../mllm_architecture_first_disaggregation_discovery/figures/fig07_idefics_handoff_break_even.png) |
| 9 · 실제 Online pilot | [동일 24개의 두-GPU open-loop](../mllm_architecture_first_disaggregation_discovery/figures/fig09_cog_online.png) |
| 10 · Candidate decisions | [후보 판정](../mllm_architecture_first_disaggregation_discovery/figures/fig10_decisions.png) |

동결 데이터는 기존 Discovery200의 Discovery 분리에서 선택했다. Cog는 실제 공식 지원 범위인 한 장의 intact image를 사용했다. IDEFICS는 측정 envelope로 최대 16장을 지원하는 사례를 선택했다. Dense Qwen2.5-VL 이전 캠페인의 결과, H200 Host의 기존 Runtime과 GPU 외부 Burner 상태를 계승했다. 런처, 증거 수준, 해시, Raw, 실패 및 미실행 항목은 아래 인덱스에 있다. [PoC 계약](mllm_architecture_first_disaggregation_discovery_poc_spec.md) · [재현 명령](../mllm_architecture_first_disaggregation_discovery/README.md) · [실행 Ledger](../mllm_architecture_first_disaggregation_discovery/results/execution_ledger.json)

## 실패, 미실행 항목, GPU 종료 상태

첫 Cog smoke에서 Legacy Triton이 compiler path를 찾지 못했다. 기존 환경의 GCC/G++를 명시하여 재실행했고 이후 Ordinary 및 Deterministic profile을 모두 완료했다. 첫 ERNIE 시작에서는 FlashInfer 샘플러 warmup이 nvcc를 찾지 못했다. 동일 환경의 CUDA Toolkit 경로와 vLLM Torch sampler를 명시하고 원본 BF16 greedy 계약으로 재실행했다. IDEFICS에는 별도의 2개 요청 결정론적 재현을 수행했다. [실행과 실패 Ledger](../mllm_architecture_first_disaggregation_discovery/results/execution_ledger.json)

독립 N3, 새 Validation50 결과, 생산 HTTP/eager-to-paged 이식 및 공식 downstream 품질은 현재 결과의 범위 밖이다. IDEFICS와 ERNIE의 online은 `NOT_RUN`이다. 케이스별 격리 정합성, Whole-model compute 및 두 GPU open-loop를 증거별로 기록했다. 외부 Burner는 실제 사용 장치에 맞춰 복원했으며, CPU/열 간섭의 별도 비교는 이번 측정 범위 밖이다. [상태와 증거 목록](mllm_architecture_disaggregation_correctness.md) · [GPU0–7 Burner 복원](../mllm_architecture_first_disaggregation_discovery/results/burner_restore.json)
