# MLLM EPD 선행연구·공식 구현 감사

상태는 `SOURCE_AUDIT`다. 아래 수치는 각 논문의 보고값이다.
현재 Qwen2.5-VL 실험의 성능 결과가 아니다.
원문 PDF·추출 Text·SHA256은 이번 Evidence의 `references/primary_papers/`에 보존했다.
최초성은 확정하지 않는다.

## 실행 구조 구분

| 구조 | 작업 단위 | 이번 연구와의 관계 |
|---|---|---|
| E/P/D 분리 | Encoder·Prefill·Decode Stage | Native Baseline |
| Chunked Prefill | 같은 Engine의 Iteration별 Token Quantum | Strong Baseline에도 적용 |
| Whole-request Deflection | 요청 전체의 Prefill을 다른 Worker에서 실행 | 별도 대조군 |
| Exact Prefix/Suffix | 한 요청의 Prefix KV를 옮기고 나머지 Query 실행 | Generic Token Split 후보 |
| Visual Pruning | 일부 Visual Token을 원래 계산에서 제외 | 이번 범위 아님 |
| Deferred Visual Prefill | 일부 정보를 첫 답변 이후 보완 | 모든 Token을 첫 출력 전에 처리하는 Exact Split과 다름 |
| Context Parallelism | 같은 Layer의 Token 연산과 KV 통신을 GPU 사이에 분산 | 추가 Idle 연구의 미래 후보. 구현 범위 아님 |

## 주요 논문

### EPD Disaggregation, ICML 2025

E/P/D Stage 분리·Encoder 내부 병렬화·자원 할당·Role Switching을 다룬다.
모델은 MiniCPM-V2.6과 InternVL2-8B/26B다.
주 GPU 실험은 A100 8개와 FP16이다.
Synthetic·NextQA·Video-MME를 사용한다.
Appendix E.1은 Online100 요청과 E/P/D Batch1을 명시한다.
따라서 이 결과를 현재 C3 Continuous Batching의 직접 재현으로 취급하지 않는다.
TTFT·TPOT·SLO·메모리를 보고한다.
[논문](https://proceedings.mlr.press/v267/singh25d.html), [공식 코드](https://github.com/vbdi/epdserve).

공개 코드 감사 Revision은 `9b111f710dfb4724072b603d626a1e4dad93ef1f`다.
README는 vLLM0.6.1.post2·Python3.10·CUDA12.1을 요구한다.
현재 설치된 vLLM0.26.0에 그대로 적용할 수 있다고 가정하지 않는다.

### ModServe

요청한 `2502.00937`의 현재 제목은 ModServe다.
Decoder-only와 Cross-attention MLLM의 실행 특성을 구분한다.
Stage 분리·Modality-aware Scheduling·Autoscaling을 다룬다.
Profiling은 A1008개다. Main 평가 규모는 A100128개다.
ShareGPT-4o와 Production Trace 기반 Workload를 사용한다.
모델군은 LLaVA-OneVision·InternVL·NVLM·Llama3.2-Vision을 포함한다.
P99 TTFT/TBT와 처리량·비용을 평가한다.
동일 요청의 Exact Visual Prefix/Suffix 구현을 확인한 근거로 사용하지 않는다.
직접 실행 가능한 공개 System Code는 이번 확인 범위에서 확정하지 못했다.
[원문 v3](https://arxiv.org/abs/2502.00937v3).

### EPD-Serve, Ascend

E/P/D 분리·Feature Prefetch·Grouped KV 전송·Co-location을 다룬다.
Atlas800I A2와 64GB NPU를 사용한다.
모델은 openPangu-7B-VL과 Qwen3-VL-8B다.
VisualWebInstruct와 ShareGPT-4o의 512개 Subset을 평가한다.
TTFT·TPOT·Throughput·SLO를 보고한다.
현재 H100·Qwen2.5-VL·원문 이미지 해상도 계약과 다르다.
공개 구현 URL은 논문의 확인한 범위에서 확정하지 못했다.
[원문](https://arxiv.org/abs/2601.11590).

### HydraInfer

Hybrid EPD 배치와 Stage-level Batching을 다룬다.
모델은 LLaVA-1.5-7B·LLaVA-NeXT-7B·Qwen2-VL-7B다.
Hardware는 NVLink H8008개다. CUDA12.4와 FP16을 사용한다.
TextCaps·TextVQA·VizWiz·MME 등을 평가한다.
자연 생성으로 출력 길이를 얻은 뒤 Fixed-length Replay를 사용한다.
Eager vLLM 대조군과 TTFT·TPOT·SLO·Goodput을 보고한다.
여기서 Hybrid는 Stage 배치다. Layer×Token Hybrid라는 뜻이 아니다.
[논문 v1](https://arxiv.org/abs/2505.12658v1), [공개 코드](https://github.com/dongxianzhe/hydrainfer).

공개 코드 감사 Revision은 `009257f784612dee022c530f8a21024dc22b99a6`다.
현재 Compiled Strong Baseline을 이 논문의 Eager 계약으로 약화하지 않는다.

### EAServe

Encode를 제어 지점으로 사용한다.
Micro-batching·E GPU의 Local Prefill·SM 자원 분할·Configuration Search를 결합한다.
Local Offload 비율과 같은 요청의 Exact Prefix/Suffix Cut을 혼동하지 않는다.
모델은 LLaVA-v1.6-34B·Qwen2.5-VL-32B·Ultravox27B다.
Main Platform은 RTX6000 Ada48GB8개다. A10080GB8개 결과도 보고한다.
영상·이미지는 Synthetic 입력이고, 음성은 LibriSpeech를 사용한다.
대조군은 vLLM0.14.1과 Dynamo0.9.0이다.
Goodput과 Mean/P99 TTFT·TPOT를 평가한다.
직접 실행 가능한 공개 구현은 이번 검색에서 확정하지 못했다.
E GPU Idle 활용 자체를 새 기여로 주장할 수 없다.
[원문](https://arxiv.org/abs/2609.31551).

### FluidPD

FluidToken은 Prefix KV 전달 후 D에서 같은 요청의 Suffix를 실행한다.
FluidRole은 장기 부하 변화에 따라 P/D 역할을 바꾼다.
SGLang0.5.4 기반이며 CUDA Graph를 유지한다.
Llama3.1-8B·Qwen3-14B·Qwen3-30B-A3B를 A10080GB8개에서 평가한다.
Azure Trace 기반 Staged30K·Mixed10K Workload를 사용한다.
주 지표는 TTFT와 TPOT를 함께 만족한 Request Attainment다.
이는 Exact Token Split의 중요한 선행연구다.
해당 논문만으로 MRoPE·이미지 Span 내부 Cut 지원을 확인할 수는 없다.
직접 실행 가능한 공개 Code URL은 확인한 원문에서 확정하지 못했다.
[원문](https://arxiv.org/abs/2610.06917).

### Kairos

부하와 Decode SLO를 보고 요청 전체의 Prefill을 D로 Deflect한다.
Deflected 요청은 D-local Chunked Prefill을 실행한다.
P에서 계산한 Prefix KV에 이어지는 같은 요청의 Suffix Split과 다르다.
vLLM0.18.1·DeepSeek-V2-Lite·Azure A100 환경을 사용한다.
대표 설정은 2P2D다. Production-style Trace와 P95 TTFT/TBT를 평가한다.
새 PoC의 Whole-request Control을 공식 재현으로 명명하지 않는다.
공개 구현 링크는 원문이 제시한 저장소다. 이번 캠페인에서 실행 검증하지 않았다.
[논문](https://arxiv.org/abs/2607.02043), [저장소](https://github.com/sudokara/Kairos).

### DuetServe

동일 GPU 안에서 필요할 때 Prefill·Decode의 SM을 분리한다.
Figure2는 Qwen3-8B·H1002개·Input8000/Output200의 Motivation이다.
TTFT·TBT·전체 Input+Output Throughput을 비교한다.
우리 3GPU·Image-only400 결과는 해당 수치의 재현이 아니다.
원문의 질문과 그래프 구성을 다른 실제 Workload에 적용하는 비교다.
논문 Method 자체의 공개 구현 실행은 이번 Scope에서 검증하지 않았다.
[ICML2026 원문](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf).

### Raincheck

2026-11-24 확인 시 최신 v2는 저자 요청으로 Withdrawn 상태다.
이 사실을 생략하지 않는다.
보존한 v1은 첫 Prefill에서 일부 Visual Token만 사용한다.
나머지는 Decode 중 또는 답변 후에 Materialize한다.
Qwen3-VL-8B·Qwen3.5-9B와 Multi-turn Workload를 다룬다.
미계산 KV의 Attention Visibility를 제어한다.
일부 PD 배치 논의도 포함한다.
모든 Token을 첫 출력 전에 처리하는 Exact C3 EPD Baseline이 아니다.
v1의 공개 구현 완성도는 이번 캠페인에서 실행 검증하지 않았다.
[현재 상태](https://arxiv.org/abs/2610.39562), [역사적 v1](https://arxiv.org/abs/2610.39562v1).

## NVIDIA Dynamo와 설치된 vLLM

[Dynamo 기술 문서](https://developer.nvidia.com/blog/when-to-use-encode-prefill-decode-disaggregation-to-accelerate-multimodal-model-serving/)는 E/PD 분리와 NIXL Embedding 전달을 설명한다.
주 모델은 Qwen3.5-122B-A10B NVFP4다.
GB2004개와 별도 RTX6000D Encoder Tier를 사용하는 구성이 포함된다.
ITL100ms Goodput과 Input/Output 길이별 효과를 보고한다.
추가 Encoder GPU를 사용한 결과를 동일한 3GPU 예산의 개선으로 인용하지 않는다.
전용 저가 E Tier와 동종 H100의 E/P/D 배치는 구분한다.

[vLLM2026-11-11 문서](https://vllm-project.github.io/2026/11/11/encoder-disaggregation-epd.html)는 v0.32 기능을 포함한다.
현재 설치 버전은 vLLM0.26.0+cu129다.
설치된 EC Factory에는 `ECExampleConnector`와 `ECCPUConnector`가 있다.
전자는 Debug File 전달이다. 후자는 Engine-local CPU Offload이며 자동으로 원격 EPD가 되지 않는다.
최신 문서의 NIXL/Mooncake Encoder Connector를 설치 버전이 지원한다고 가정하지 않는다.
현재 NIXL 지원은 별도의 P→D KV Connector에서 확인했다.

## 현재 Runtime에서 확인한 제약

첫 EPD Prototype은 공식 EC Debug Connector를 tmpfs에서 사용한다.
실제 요청 후 E에서 생성한 Embedding을 넘긴다. 사전 계산 Replay가 아니다.
다만 CPU 경유 전달·Raw Media 재전달 비용을 포함한다.
이를 GPU-direct Encoder Transport라고 부르지 않는다.

두 Cache를 모두 끄면 Media UUID가 Renderer-local 순번으로 바뀌었다.
E와 P의 순번이 어긋나 Calibration32가 실패했다.
Processor Cache1GiB와 요청별 고유 UUID를 모든 Arm에 적용했다.
CPU Regression은 실제 설치된 Renderer에서 이 원인과 수정 설정을 확인했다.
실패 로그와 수정 전후 Trace는 보존했다.

## Novelty 판단

Stage 분리·E Idle 활용·D Whole-prefill Deflection·일반 Exact Token Split은 이미 연구된 개념이다.
현재 목표는 Qwen2.5-VL의 실제 Image-only C3에서 그 적용 가능성과 제약을 검증하는 것이다.
이미지 Span 내부 Cut의 정합성·MRoPE·Queue·동시 E/D Slack은 별도 실측이 필요하다.
이 감사만으로 최초성이나 Online 개선을 주장하지 않는다.
아직 Modality-aware Method를 개발하지 않았다.

추가 E/D Idle 연구는 [vLLM Context Parallel 문서](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/)의 Q/KV 통신 요구도 검토한다.
Window가 겹친다는 사실만으로 같은 요청의 병렬 실행이나 Speedup이 성립하지 않는다.
