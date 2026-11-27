# Architecture 후보 선정

CogVLM2-19B와 IDEFICS-9B의 조건부 경계는 공개 Checkpoint에서 각각 독립적으로 검증할 수 있다. 두 PoC는 실제 GPU Candidate 성능 측정 전에 Source와 다운로드 가능성 기준으로 선정했다. ERNIE-VL은 소스 스크리닝에서 발견한 독립 Runtime 대조다. [전체 후보 Registry](mllm_modality_dependent_architecture_landscape.md)

| 선정 | 확인한 구조 | 재사용 가능성과 비용 |
|---|---|---|
| **CogVLM2-19B** | 32개 Decoder 층의 별도 Visual QKV/O/FFN Weight와 Language Weight. 생성 시 Type은 출력 Vocabulary ID와 독립적으로 Language로 append된다. Native E와 visual-prefill K/V는 유지해야 한다. | 공개 publisher checkpoint `f592f291cf528389b2e4776b1e84ecdf6d71fbe3`. BF16 Tensor payload 약 39.0 GB. 동일 NV18 H200 두 장에서 온전한 P와 구조적으로 줄인 D를 실행할 수 있다. 공개 vision/source와 CogVLM academic license를 검토했다. [고정 Checkpoint](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B/tree/f592f291cf528389b2e4776b1e84ecdf6d71fbe3) |
| **IDEFICS-9B** | 32개 Self Decoder 층 중 8개에 별도 Gated Cross-attention Block. Perceiver는 이미지당 64개 latent를 생성하며 생성용 이미지 접근 집합은 마지막 Prompt Mask를 재사용한다. | 공개 publisher checkpoint `7065afb8a5bb03015700b315f6751dc3dd4e16eb`, BF16 safetensors 약 17.86 GB. 설치된 공식 HF IDEFICS-1 모델로 입력과 K/V를 검사할 수 있다. [고정 Checkpoint](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct/tree/7065afb8a5bb03015700b315f6751dc3dd4e16eb) |
| **ERNIE-4.5-VL-28B-A3B-Thinking** | 28개 층의 shared self-attention. 27개 층에서 row role에 따라 64개 Vision FFN expert와 64개 Text FFN expert를 선택한다. 각 pool 안에서 top-6를 사용한다. | Ungated Apache-2.0 publisher checkpoint `e7487b6ff5e726e21bb8bcf9af83b34a82fcdaae`. 29개 safetensors의 파일 합계 59.35 GB. 설치된 vLLM 구현으로 분기와 원본 residency를 확인한다. Image-only CPU overlay 범위는 별도로 보존한다. [고정 Checkpoint](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking/tree/e7487b6ff5e726e21bb8bcf9af83b34a82fcdaae) |

Meta Llama Vision은 공식 Gated 경로에서 필요한 접근을 확인하지 못했다. Florence2는 일반 Encoder–Decoder와 작은 Target 범위 때문에 유력한 새 경계의 대조로 남는다. Granite Vision 및 IDEFICS3의 구현은 fused embedding을 shared decoder에 전달한다.

InternLM-XComposer2-4KHD는 span-conditioned PLoRA를 실제 사용한다. 설치 환경 지원이 낮고 text-only 경로에는 output을 0으로 만드는 one-row adapter 호출이 남아 있어, 기본 경로 자체가 호출을 생략한다고 볼 수 없다. DeepSeek-OCR2는 Vision 내부에 앞쪽 Image Patch와 뒤쪽 causal flow query의 실제 의존성 차이가 있다. 이 검증에서는 OCR 태스크의 지원 범위와 E 내부 pipeline의 일반 대조가 남아 있어 Weight 확보에 앞서 보류했다.

BAGEL은 Image Generation과 Understanding에 별도 MoT Bank를 사용한다. Image→Text의 installed vLLM 경로는 이미 VAE 및 generation-only MoT Weight를 배제한다. 같은 능력을 비교하려면 해당 최적화 경로가 Native다. Uni-MoE-2.0과 일반 MoE의 learned top-k는 실제 계산을 선별한다. 고정적인 Modality-to-Expert 도달 불가능성을 Source에서 확보하지 못한 경우는 이번 구조적 Residency 가설과 분리한다. [Prior Work와 Baseline](mllm_architecture_prior_work_novelty_audit.md)
