# Modality별 Parameter와 State Liveness

CogVLM2는 P에서 실제 이미지 Prompt를 처리할 때 독립 Visual Expert를 사용한다. 같은 모델의 텍스트 생성 D에서는 그 Expert의 가중치가 없어도 모든 Prompt 행의 KV를 읽는 기존 Attention이 유지된다. Hard Routing으로 증명 가능한 D의 파라미터 비도달성과 다른 Modality의 정보가 담긴 KV의 수명을 분리했다. [공식 CogVLM2 논문](https://arxiv.org/abs/2408.16500) · [고정 모델 소스](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B/tree/f592f291cf528389b2e4776b1e84ecdf6d71fbe3)

## 후보 조사

| 모델 | 실제 Operator Routing | 이번 연구의 역할 |
|---|---|---|
| CogVLM2 Llama3 Chat 19B | 32개 Layer마다 Visual 및 Language QKV·O·MLP 분리. 공유 Causal Attention으로 두 역할의 KV를 읽음 | 원본 BF16 체크포인트로 P→D 실측 |
| CogVLM Chat | 이전 계열에서 같은 Semantic Token Type 기반 전문가. 고정 이미지용 모델 | 계보 및 코드 대조 |
| CogAgent Chat | 별도 Visual/Language Expert와 각 Decoder Layer의 Cross Attention. Self KV와 별도로 Cross KV가 살아 있음 | 일반 CogVLM2 Self KV 계약을 그대로 적용할 수 없는 대조 |
| ERNIE-4.5-VL-28B-A3B-PT | 독립 Text/Vision Sparse MoE와 Shared Experts. Native Transformers 및 vLLM의 별도 Text/Vision Router | 다른 계열에서 같은 가설이 적용될 수 있는 소스 근거 |
| InternVL3.5-30B-A3B | 시각 임베딩 후 Qwen3MoE의 학습된 라우팅 | Expert 호출률 관측만으로 Exact Zero Reachability를 보장하기 어려운 대조 |

CogVLM2는 명시적인 Layer별 QKV·O·MLP 분기, 공식 비공개 인증이 필요 없는 Safetensors, P에서 생성되는 완결된 Self KV 계약 때문에 첫 GPU 검증에 선택했다. ERNIE의 소스는 Generation에서 Prompt 다음 Step의 Multimodal MoE Role을 제거한다. Shared Expert와 Full Prompt 상태는 계속 필요하다. ERNIE와 CogAgent의 GPU 실측은 이번 CogVLM2 PoC 범위 밖이다. [CogVLM Chat](https://huggingface.co/zai-org/cogvlm-chat-hf/tree/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c) · [CogAgent 코드](https://huggingface.co/zai-org/cogagent-chat-hf/blob/26eec27a44348fbe0c9fad89348cf6a505f5a5ae/modeling_cogagent.py) · [ERNIE 기술 보고서](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf) · [InternVL 코드](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B/blob/ba1762d830e2aeddb3272b66ba9e0abf7e69f626/modeling_internvl_chat.py)

## CogVLM2의 분리 계약

공식 `get_expert_mask`는 위치 \(i\)와 \(i+1\)의 **Semantic Token Type**이 모두 Vision일 때만 위치 \(i\)를 Visual Expert로 보낸다. 한 Query Row인 D Step에서 마지막 열은 Visual Expert에 들어가지 않으며, 공식 Generation 업데이트는 어떤 Vocabulary ID가 생성됐는지와 무관하게 Language Role을 붙인다. EOS, BOS 및 이미지 Placeholder로 재사용되는 PAD Vocabulary ID도 기존 Generation에서는 Language Role이다. 새 이미지를 수용하는 Prompt Prefill 및 재처리는 P의 경로로 들어간다.

D의 Text QKV는 기존 BF16 Language Expert에서 계산된다. Attention은 현재 Layer의 원본 8개 KV Head와 원래 위치에서 회전된 K를 읽고, 현재 Token의 K/V를 원래 순서대로 Append한다. Output Projection, MLP, Shared Norm, Token Embedding, Logit Head는 그대로 남는다. Attention을 통해 Visual 내용에 접근하는 것과 Visual Expert Weight에 접근하는 것은 독립적이다. [CogVLM2 모델 코드](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B/blob/f592f291cf528389b2e4776b1e84ecdf6d71fbe3/modeling_cogvlm.py) · [검증 코드](../mllm_architecture_first_disaggregation_discovery/hard_expert_decode.py)

| Arm | D에 상주한 모듈 | Decode 경로 |
|---|---|---|
| Resident Original | Language + Visual Expert | 공식 Mask 및 두 Expert 호출. Visual 쪽은 0행 |
| Resident Text Fast | Language + Visual Expert | Language QKV·O·MLP 직접 실행 |
| Visual Pruned | Language Expert 및 공유 모듈 | Resident Text Fast와 같은 연산 |
| Generic Reachability Pruned | 빈 Row Route 분석으로 선택된 Weight | 같은 Direct Language 경로 |

모든 D Arm에서 Vision Encoder는 처음부터 공통으로 제외했다. Visual Pruned는 한 Layer에 Visual QKV Weight/Bias, Visual O Weight, Visual MLP Gate/Up/Down Weight를 건너뛴다. 32개 Layer의 192개 Tensor는 Meta Model에서 제거하고 Safetensors의 해당 키를 GPU로 읽지 않았다. Generic은 원본 Decoder AST에서 비어 있는 Mask Row에만 호출되는 모듈을 확인한다. 현재 고정 소스에서 독립적으로 같은 192개 Weight를 식별했다. 이는 CogVLM2의 Hard Modality Contract가 Exact Phase Specialization을 가능하게 한다는 근거다. 해당 가중치를 찾기 위해 별도 Modality Predictor가 필요한 상황은 확인되지 않았다.

## 체크포인트와 실행 범위

공식 Revision은 `f592f291cf528389b2e4776b1e84ecdf6d71fbe3`다. 8개 Shard Header의 1,221개 BF16 Tensor와 전체 Meta Model Keys가 일치했다. `modeling_cogvlm.py`, Config, Vision, Rotary Utility, Index, License의 SHA-256을 검증한 뒤 코드만 로드한다. 공식 라이선스는 연구 사용을 허용한다. Shard와 변환된 Weight는 결과 Repository에 저장하지 않는다. 설치된 vLLM 0.26의 Native Model Registry에는 CogVLM2가 없어 이 검증은 원본 Model Forward와 CUDA Peer Copy로 수행했다. 공통 Vision Attention의 xFormers 호출에는 동일 수식·Scale·Dtype의 PyTorch SDPA를 연결했다. 첫 Smoke는 Triton Rotary 컴파일에 필요한 Host `CC`를 찾지 못해 중단됐고, 고정 Runtime GCC를 지정한 시작에서 이어서 검증했다. [공식 Index](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B/blob/f592f291cf528389b2e4776b1e84ecdf6d71fbe3/model.safetensors.index.json) · [공식 License](https://huggingface.co/zai-org/cogvlm2-llama3-chat-19B/blob/f592f291cf528389b2e4776b1e84ecdf6d71fbe3/LICENSE)

## 장치 상주량과 64GiB의 동일 저장 예산

| D Arm | 실제 적재 Tensor | Tensor Bytes | CUDA Allocator 증가 | Visual Expert 적재 |
|---|---:|---:|---:|---:|
| Resident Original | 451 | 30,019,563,520 | 30,019,575,808 B | 192 |
| Resident Text Fast | 451 | 30,019,563,520 | 30,019,575,808 B | 192 |
| Visual Pruned | 259 | 16,060,526,592 | 16,060,538,880 B | 0 |
| Generic Reachability Pruned | 259 | 16,060,526,592 | 16,060,538,880 B | 0 |

같은 H200 D에서 실제 Allocate한 Tensor와 CUDA Allocator 증가량이 모두 정확히 **13,959,036,928 B**, 약 **13.000 GiB** 감소했다. Encoder를 공통 제외한 D의 Resident Tensor 대비 **46.50%**다. P의 원본 Weight/Buffer는 모든 Arm에 동일한 **39,006,214,656 B**다. 이 수치는 Layer를 생략하거나 BF16을 다운캐스팅해 얻은 값이 아니다. [Shard Header와 Source Receipt](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/source_receipt.json) · [GPU Smoke Raw](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/smoke.json)

DocVQA 56399의 실제 CogVLM2 Prompt는 2,322행이었다. Processor가 넣은 Image Role 2,306행 중 실제 Visual Expert 호출 대상은 2,305행이었다. Full32의 원본 K/V 64개, 304,349,184 B를 P에서 D로 복사하고 모든 값을 대조했다. Weight와 새 KV Arena의 합에 동일한 64GiB 한도를 적용했다. 128행의 출력 Reserve를 추가하면 한 Slot은 321,126,400 B다. Resident는 **120 Slot**, Visual/Generic Pruned는 **163 Slot**의 GPU Tensor를 실제 할당하고 전부 기록했다. 저장 가능한 Slot 수가 **35.8%** 늘었다. 이 Arena 시험에는 별도의 검증용 Prompt 복사본·CUDA Runtime Allocation·Paged-KV Fragmentation이 들어가지 않는다. 163개 Online 요청을 Admission하거나 완료한 결과로 해석하지 않는다. [동일 예산 GPU 측정](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/smoke.json)

## 최초 Smoke의 수치 변동

Resident Original, Resident Text Fast와 Visual Pruned의 B1 두 Teacher Step에서 Logit과 Full32 최종 KV가 모두 같았다. 동일한 Load Key를 사용한 Generic Pruned에서는 최초 Start의 Resident Original 대비 Logit Relative L2 약 0.33%, 최종 KV 23/32 Layer 일치가 관측됐다. 네 Arm의 짧은 Natural Token Sequence는 일치했다. 실제 BOS/Cog Pad 및 EOS 두 Vocabulary ID를 소비하는 B4 진단에서도 Default Language Role이 사용됐으며, 같은 Batch에 Semantic Vision Role을 넣으면 Forward 전에 거절했다. 이 초기 실행에서 Special-ID 묶음의 Logit은 Original과 다른 일부 직접 Decode Arm 사이에 차이가 있어, 더 긴 검증에서는 같은 Direct Route의 A/A 대조와 Deterministic 설정을 분리했다. [Smoke Raw](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/smoke.json)

## 고정 Deterministic 경로의 정합성

한 이미지 제한 안에서 Discovery의 실제 12개 이미지·텍스트 요청을 네 Source에서 동일 규칙으로 골랐다. 원본 CogVLM2 Processor 기준 Prompt는 2,318–2,554행이다. Full32의 **768/768 K/V Tensor**가 실제 P→D Import 후 일치했다. 한 Prompt의 논리 Payload는 303.825–334.758 MB이며 Copy Wall에는 Python 제출과 동기화가 들어간다. 최초 Device/Copy 준비와 각 요청의 편차를 포함한 값으로 Kernel-exclusive Link Time으로 해석하지 않는다.

Deterministic CUDA 연산과 `CUBLAS_WORKSPACE_CONFIG=:4096:8`를 공통으로 사용했다. Visual Pruned는 같은 Direct Language 연산을 수행하는 Resident Text Fast에 대해 **12/12 Teacher Logit Tensor와 12/12 Full32 최종 KV 상태가 정확히 일치**했다. 8 Teacher Step씩 총 96개 다음 Token을 비교했다. P에서 만든 첫 Token 이후 최대 48개의 Natural 예측을 비교한 결과 12/12 요청, 총 313개 D 예측 Token ID가 P의 원본 경로와 같았다. 9개 요청은 EOS로 끝났고 3개는 확인 구간의 Cap에 도달했다. 같은 계약의 Original 대조에도 Logit 및 KV 일치가 관측됐다.

서로 다른 네 원본 Prompt의 실제 KV를 2,385/2,442/2,365/2,318행에서 Left-pad해 준비한 Ragged B4에 네 번의 실제 Forward를 수행했다. 원래 요청의 Attention Mask와 Position은 유지했다. Visual Pruned의 16개 Teacher Argmax와 Logit Tensor 전체가 Resident Text Fast와 같았다. 별도 4행 Batch에서 명시적인 `128002, 128000, 128001, 128009`를 Language Token으로 소비했을 때도 동일 Logit이 나왔다. 같은 입력에 Vision Semantic Role을 끼우면 D API는 Forward 전에 거절했다. B1/B4/B8의 반복 Timing에는 같은 원본 Prompt의 KV 복제를 사용했다. 독립 네 요청을 동시에 계산한 Ragged B4는 이 복제 Sweep과 분리했다.

| D Layout | 8-step Teacher Logit | 최종 Full32 KV | Natural Token IDs | 실제 Ragged B4·4 Step |
|---|---:|---:|---:|---:|
| Resident Text Fast | 12/12 | 12/12 | 12/12 | 16/16 |
| Visual Pruned | 12/12 | 12/12 | 12/12 | 16/16 |
| Generic Reachability Pruned | 12/12 | 12/12 | 12/12 | 16/16 |
| Resident Text Fast A/A 독립 적재 | 12/12 | 12/12 | 12/12 | 16/16 |

Teacher, KV 및 Ragged의 분모는 결정론적으로 검증한 Resident Original 비교다. 같은 연산의 Resident Text Fast와 다시 비교해도 같은 일치가 유지됐다. Natural은 P 원본 경로와 비교했다. A/A는 같은 GPU Epoch 안에서 새 모델을 적재해 다시 계산한 대조이며 독립 Server Start는 아니다. [Deterministic Full Raw](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/profile_deterministic.json)

12개 중 최대 Prompt인 2,554행에 128행의 Reserve를 더하면 Slot은 351,535,104 B다. 같은 64GiB 예산으로 Resident **110 Slot**, 두 Pruned Arm **149 Slot**을 Allocator에서 실제 확보하고 기록했다. 각각 38,669,385,728 B와 52,378,730,496 B로 Slot 수는 **35.5%** 늘었다. 이 값은 같은 원본 요청의 KV Geometry를 가진 저장량이다. 고정 Ragged B4는 독립 네 요청의 실제 Forward로 확인했고, 다중 반복의 B8은 동일 요청 KV 복제 Geometry로 실행했다.

현재 Eager Decode는 `torch.cat`으로 이전 Full32 KV와 새로운 Full32 KV를 Step 경계에 잠시 함께 보유한다. 64GiB에 110/149개를 동시 실행하려면 이 임시 중복을 처리할 추가 공간이 필요하다. 같은 2,554+128행에서 두 복사본만 계산해도 64GiB의 수치상 상한은 Resident 55개와 Pruned 74개다. GQA 확장과 Workspace를 포함하면 더 낮을 수 있다. 이 수치는 메모리 산술이며 그 배치의 실행 실측은 없다. 온라인 Admission과 Paged KV 동시성은 별도의 실행 근거가 필요하다.

## 일반 Runtime Timing과 Route Guard

Deterministic 정합성을 검증한 Epoch와 별도로 같은 12개 Prompt에서 일반 Runtime을 실행했다. 요청별 중앙값을 다시 중앙값으로 집계했다. 실제 Ragged B4의 값은 네 요청의 고정 Geometry를 세 번 반복한 중앙값이다. 아래는 D Decode Step Wall Time이다. P, 온라인 Queue, SLO는 이 Timing에 들어가지 않는다.

| D Layout | 동일 원본 KV 복제 B1 | 동일 원본 KV 복제 B4 | 실제 독립 요청 Ragged B4 |
|---|---:|---:|---:|
| Resident Text Fast | 27.895ms | 34.680ms | 38.237ms |
| Visual Pruned | 27.700ms | 34.308ms | 35.339ms |
| Generic Reachability Pruned | 27.589ms | 34.524ms | 35.383ms |

B1/B4 복제 Timing에서 두 Pruned Arm과 Resident Text Fast의 차이는 1.1% 이내였다. 실제 Ragged B4에서는 한 Epoch에서 두 Pruned Arm이 모두 소폭 빨랐고, Visual과 Generic 간 차이는 0.044ms였다. 동일 Compute 대조에 대한 Pruning의 재현 가능한 Decode 가속이나 별도 Modality Policy 이점은 확정하지 않았다.

일반 Runtime에서 두 Pruned Arm 모두 **72/72 Teacher Argmax**, **12/12 Natural Token ID Sequence** 및 **16/16 Ragged B4 Argmax와 전체 Logit Tensor**가 Resident와 일치했다. Natural에서는 공통 P 첫 Token 이후 최대 12 예측, 총 100개를 확인했고 그중 5개 요청은 EOS로 끝났다. B1의 Teacher Logit과 Full32 KV Tensor까지 엄격하게 일치한 요청은 Visual Pruned **0/12**, Generic **1/12**였다. Visual에서 Logit Relative L2 최대 **0.894%**, Generic **0.793%**였다. 숫자 차이를 동일 도달성을 가진 설정에서도 관측했다. 일반 Runtime의 Bitwise Reproducibility는 입증 범위에 두지 않았다. Deterministic 확인이 이 Ordinary 수치 차이의 단일 커널 원인을 입증하지는 않는다.

별도 Text-only Guard는 공식 Publisher Prompt Builder와 원본 Model을 실제 실행했다. 19개 Prompt Token에서 Visual Role과 실제 Visual Expert Row가 각각 **0**이었다. 원본 Full32 K/V **64/64 Tensor**가 D Import 후 일치했고, 동일 4 Teacher Step에서 세 Arm의 Logit이 모두 정확히 같았다. 이 Guard는 이미지 Discovery 성능 집계에서 제외했다. [Ordinary Full Raw](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/profile_ordinary.json)

## Evidence Scope

고정한 Semantic Generation 계약에서 **32개 Layer의 Visual Expert Weight 13.000 GiB가 D에서 필요 없다는 점**과 **P가 만든 Visual Prompt KV가 계속 필요하다는 점**을 함께 확인했다. 동일 GPU·연산·예산 대조에서 물리적으로 할당할 수 있는 KV Storage Slot이 약 **35%** 늘었다. 해당 효과는 Hard Modality Graph에서 발생하며 보존 집합은 Generic Reachability로도 동일하게 계산됐다. vLLM Native 처리, 여러 이미지가 들어오는 입력, 영상·음성 생성, Paged KV의 실제 높은 동시성, 독립 N3 Online 우위는 이 격리 실험에서 평가하지 않았다. [실행 명령과 상태](../mllm_architecture_first_disaggregation_discovery/results/cogvlm2/execution.json)
