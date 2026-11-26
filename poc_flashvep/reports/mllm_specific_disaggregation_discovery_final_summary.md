# Discovery200 · MLLM-specific Disaggregation 결과

**Track B는 B_INFORMATION_FLOW_ONLY, Track A는 A_NO_GO_COMMUNICATION이다.** Qwen2.5-VL-7B에서 Layer와 Modality별 기능 차이를 확인했고 두 GPU에서 Visual KV의 exact softmax 분할을 실행했다. 검증한 Layout에서 modality 정보로 GPU 또는 Serving의 추가 순이익을 얻을 근거는 부족했다.

| 비교 | 직접 계측 결과 | 범위 |
|---|---|---|
| NativeStrong-Free BI0 | 200/200 완료, TTFT p95 2.040s, ITL p99 6.292ms, TPOT p95 4.910ms, joint SLO 100% | Discovery200 Natural EOS, Open-loop QPS0.75 |
| NativeStrong-Free BI0 보완 | 200/200 완료, TTFT p95 2.360s, ITL p99 10.256ms, TPOT p95 6.280ms, joint SLO 99% | 독립 Open-loop QPS2.5, grouped Discovery에서 별도 부하 탐색 |
| B · 동일 수의 Visual/Text 개입 | 초기층 Visual−Text ΔNLL −3.108 nats, 95% CI [−3.958, −2.355] | 24개 요청과 Attention residual matched 개입 |
| B · 2-GPU Exact Layout | 긴 입력의 일반 분할은 Single GPU 대비 35.5–46.6% 단축. Image boundary의 동일 자원 대비 최고 추가 단축 2.0% | 실제 BF16 Embedding/Weight, Decoder 2층, 5개 입력, 12 timed repeats |
| A · Full28 KV Placement | Ragged B4 Local 12.860ms, Visual 18.328ms, 동일 행 수 Generic Prefix 18.305ms | 원본 BF16, FA3, 2×H200 NV18, fixed one-token replay |

Track B를 마친 후 Track A를 별도 GPU Epoch로 실행했다. 동일한 체크포인트와 200개 고유 요청을 사용했다. 24개 상세 요청은 grouped Discovery150에서 사전 동결했다. [설계 계약](mllm_specific_disaggregation_discovery_poc_spec.md) · [Prior Work](mllm_specific_disaggregation_prior_work_audit.md)

## 공통

1. **실제 구성.** 입력은 평균 5,815.61 tokens, 중앙값 1,527이다. 평균 Visual/Text/Special은 5,715.74/81.20/18.67 tokens이며 Visual은 전체 Prompt 행의 98.28%다. Native 출력은 총 31,835 tokens, 평균 159.18, 중앙값 41, 최대 2,048이며 MathVista 한 건이 cap에 도달했다. [분포와 Figure 1](mllm_discovery200_dataset_profile.md)
2. **Main100과 새 Reasoning Source.** Main100 입력 평균은 11,029.59, 출력 중앙값은 25.5다. MMMU의 출력 중앙값은 2, MathVista는 188, MMReason은 404였다. Official MMMU는 55%였고 MathVista 50%, MMReason 0%는 부분 로컬 Scorer 수치다. Main100 DocVQA의 원문 ANLS는 긴 응답을 짧은 정답과 그대로 비교한다. [Native 품질과 범위](mllm_discovery200_native_baseline.md)
3. **일반 Serving의 범위.** Native EPD를 Prefill-heavy/D-idle 제약 없이 open-loop로 실행했다. 정보 흐름과 KV 실험은 이 부하에서 얻은 실제 Prompt와 Native Token Prefix를 사용했다. 후보의 온라인 Batching 및 Goodput은 미측정이다.

## Track B · Visual Understanding–Reasoning

4. **Layer별 역할.** Post-MRoPE 관측 25,536개에서 Question→Visual 질량은 층 그룹별 약 8–18%, Generated→Visual은 5–12%였고, 해당 causal key prevalence는 약 78%와 77%였다. Visual Query는 같은 이미지에 집중했고 초기층 Text/Question residual 변화가 컸다. 모델에서 실제 분리 가능한 경계는 원본 Vision Encoder/Projection, Attention과 MLP다. 관측한 기능적 역할은 입력과 Layer에 따라 달라졌다. [관측과 도식](mllm_visual_understanding_reasoning_information_flow.md)
5. **Causal Evidence.** 24개에서 1,032개의 개입을 실행했다. 층 1–4의 matched Visual−Text ΔNLL은 Attention −3.108, MLP −2.267, 접근 가능한 KV edge −5.298 nats였다. 같은 수의 row 또는 edge를 바꿔도 영향이 달랐다. 위치 차이와 Text/Visual 중 더 작은 집합으로 matching 범위가 제한된다. [개입 통계](mllm_visual_text_causal_intervention.md)
6. **안전한 Visual 계산 제한.** 후반 여덟 층의 dose는 전체 Prompt row×layer 위치의 15.39%를 바꿨다. Dense 커널은 원래 계산을 수행했다. Natural 출력은 7/24에서 일치했고, Teacher Greedy는 533/554였다. 원문 점수를 유지했지만 양수 기준선은 5건이었다. 실제 15% GPU 절감과 1%p 품질 기준을 만족하는 범위는 미확정이다.
7. **Task에 따른 차이.** Dose에서 MMMU 5건의 HF Natural 출력을 유지했다. 긴 MP-DocVQA/MMLongBench 사례에서는 ΔNLL +0.405/+0.380 nats 및 출력 변화를 관측했다. 원문 점수 floor와 부족한 세부 strata가 품질 일반화를 제한한다.
8. **FastV/HiMAP 비교.** 직접 실측은 matched causal sensitivity와 exact layout까지다. FastV/HiMAP 최적화 Kernel과 동등 품질 Pruning 대조는 후보 채택 기준 미달로 미실행이다. [기존 연구와의 관계](mllm_specific_disaggregation_prior_work_audit.md)
9. **Parallel Layout 차이.** 원본 Vision/Prompt Activations와 동일 Shape의 label 대조에서 modality 자체의 큰 Kernel 효과는 관측되지 않았다. 2,343–8,007행은 일반 두-GPU 분할이 유리했고 115/483행은 Single GPU가 유리했다. Image boundary 대조는 일반 분할보다 최대 2.0% 빨랐고 긴 Prompt에서 불리한 경계도 있었다. [GPU 대조](mllm_modality_parallelism_gpu_headroom.md)
10. **TP/PP/CP 대비 새로운 구조.** 두 Layer의 Prefix Query split에서 원본 Causal Dependency와 peer-copy 비용을 검증했다. 통신을 포함한 이득은 Shape 기반 일반 CP 대조에서도 나타났다. Visual/Reasoning별 자원 배치의 추가 이득은 미입증이다.
11. **Exact와 Approximate.** 일반 Shape 기반 Exact Split의 제한적 Headroom을 확인했다. Approximate의 작업 제거 가능성은 causal mask 수준까지 측정했다. Full28 후보 채택 조건은 충족하지 못했다.
12. **GPU Headroom.** 가장 좋은 Image boundary의 추가 절감은 동일 자원 대비 latency 2.0%, working allocation 0.4%였다. 15% 후보 기준에 미달했다. [두 GPU PoC](mllm_understanding_reasoning_disaggregation_poc.md)

## Track A · Visual Memory–Generation

13. **Visual KV 비중.** Prompt은 98.28%, 생성 이후 논리적 행은 95.68%였다. MMReason은 66.82%에서 29.78%로 감소했다. Native에서 Prompt Block 9,188개, 각 128행, 논리 패딩 12,942행을 확인했다. 독립 QPS2.5의 NIXL Counter는 200개 전송에서 총 67.44GB를 기록했다. [KV 프로파일](mllm_visual_kv_memory_decode_profile.md)
14. **Attention 병목.** 격리한 B1 Full28 Layer에서 Local FA3 Attention은 계측 구간의 8.5–11.0%였다. 실제 Paged Native의 Attention Kernel 비중과 HBM Traffic은 미계측이다.
15. **Exact 계산과 수치.** 1,120개의 원본 Full28 초기 KV tensor에서 실제 P2P 후 일치했다. Visual Teacher Greedy 81/81, Natural 출력 5/5가 Local FA3와 일치했다. BF16 Logits과 일부 Generated KV에는 reduction order 차이가 남았다. [A2](mllm_visual_memory_generation_exact_attention.md)
16. **반복 전송.** B1에서 Layer당 14,448 bytes, Full28 Step당 404,544 bytes를 전달했다. CUDA Event에서 28층 Query Copy 합이 0.75–0.76ms, Result Copy가 0.48–0.66ms였다. 여기에는 발행 간격이 들어가며 두 Stream은 겹쳐 진행했다. 실제 NVLink Data counter로 전송을 확인했다.
17. **손익분기점.** 다섯 Shape의 eager Full28 Visual Step은 Local보다 5.36–5.47ms 길었다. 초기 절감이 양수인 두 Shape에서도 손익분기는 0.14와 0.27 Decode Step이었다. 실제 Native Output Length로 환산한 비용이 모두 늘었다. Cross-device CUDA Graph를 사용한 대조는 반복 replay까지 완료되지 않았다. [손익분기](mllm_visual_memory_remote_attention_break_even.md)
18. **일반 대조.** 동일 행 수 Generic Prefix와 Visual의 B1 차이는 0.24%보다 작았다. B4 Visual은 18.328ms, Generic은 18.305ms였다. Visual discontiguous row는 초기 packing도 필요했다. Stock vLLM 2-GPU GQA DCP는 이 체크포인트의 Head 구성에 호환되지 않아 같은 GPU의 exact Sequence Split을 대조로 실행했다.
19. **메모리와 동시성.** 28,845행 요청의 D route tensor 크기가 1,689.46→114.46 MiB로 줄었다. P에 1,575.00 MiB를 유지해 총 KV Tensor는 같았다. 실제 D Pool의 최대 동시성, OOM 한계 또는 Goodput 우위는 미검증이다.
20. **Prefill 간섭과 자원 비용.** 같은 GPU 수로 계산과 전송을 비교했다. P가 다른 요청을 동시에 Prefill하는 경쟁 시험은 미실행이다. 격리 경로에서 반복비용이 초기 절감보다 큰 조건을 확인했다. [A4/A5](mllm_visual_memory_generation_disaggregation_poc.md)

## 판정

21. **독립 결론.** B는 B_INFORMATION_FLOW_ONLY다. A는 이 NV18 구현과 실측 Shape에서 A_NO_GO_COMMUNICATION이다. Candidate Active C3와 독립 N3 Serving은 NOT_RUN이다.
22. **실제로 확인한 Multimodal 특성.** 같은 이미지 안의 Attention 집중, matched Visual/Text 민감도 차이, 실제 Workload의 Image Span별 KV 비중을 관측했다. 이러한 차이의 독립적인 GPU Layout 이득은 입증되지 않았다.
23. **일반 LLM에도 적용되는 특성.** Causal prefix의 선행 계산, Equal-shape Dense operator, 균형 Sequence Split, 고정 Prompt KV와 성장하는 Generated KV, exact partial softmax는 일반 AR LLM에도 적용된다.
24. **논문 Contribution 근거.** 확보한 것은 model/runtime-scoped 비교와 측정한 비용이다. 긴 입력의 일반 Query Split은 후속 비교 후보이며 독립적인 MLLM-specific Contribution을 고를 근거는 부족하다.
25. **다음 구현 판단.** 이 Dense 모델에서 image boundary 기반 Exact Layout, 실험한 고강도 residual 제거, 매 Layer 왕복하는 Visual Residency의 Serving 확대는 보류한다. 일반 병렬화의 다음 비교에는 Shape-selected Exact Query Split과 생산 Runtime의 최적 대조가 우선이다. MLLM 고유 구조 연구는 Vision 상태에 대한 실제 별도 계산 경계가 있는 아키텍처에서 검증할 수 있다.

## 재현 및 제한

Figure는 Scope별 CSV/JSON에서 재생성할 수 있다. 과거에 사용한 Main100을 포함한다. Internal Validation50은 Image, Context, Question, Problem 키 기준으로 grouped split했으며 Candidate Tier1에 사용하지 않았다. 공식 외부 평가가 필요한 MMLongBench-Doc, MathVista, MMReason은 부분 로컬 점수를 별도 보존했다. 실험 GPU 밖 Burner의 공유자원 효과와 GPU 계측 Observer Gate는 별도 대조하지 않았다. [데이터 계약](mllm_discovery200_dataset_profile.md) · [실행 코드와 재생 명령](../mllm_specific_disaggregation_discovery/README.md)

실험별 Command, GPU UUID, 종료 상태와 실패 증거를 보관했다. 소유한 실험 프로세스 종료 후 GPU0–7의 train_lm.py를 확인했으며 스냅샷 Utilization은 85–92%였다. [실행 Ledger](../mllm_specific_disaggregation_discovery/results/run_ledger.json) · [GPU 복원](../mllm_specific_disaggregation_discovery/results/burner_restore.json)
