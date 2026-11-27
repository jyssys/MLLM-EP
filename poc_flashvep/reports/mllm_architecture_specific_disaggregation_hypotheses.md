# Architecture에서 도출한 실행 가설

## H1 · CogVLM2의 Decode Weight Residency

Visual Type row의 QKV/O/MLP는 이미지가 포함된 Prompt에서 실제 모델 계산에 필요하다. 원래 Generation loop는 출력 Token ID와 무관하게 신규 row의 semantic Type을 Language로 확정한다. E/P에 원본 Weight를 유지하고 각 Layer의 전체 Key/Value를 전송한다. Language-only decode contract를 검증하고 D의 visual expert weight를 로드하지 않는다.

대상은 2-device P+D에서 D 가중치, 추가 동시 KV slot, 물리 전송을 포함한 Decode 지연이다. 동일 D fast path와 pointer layout에 전체 decoder weight를 둔 대조, Source reachability를 이용해 똑같이 pruning한 대조를 사용한다. Prompt KV의 Visual 부분은 정상 Attention에 남는다. Full KV가 같은지, 여러 다른 Image Prompt가 동시에 Decode되는지, 보낸 special-token ID가 Visual dispatch를 생성하지 않는지를 확인한다. 비용은 P가 다음 Prompt를 처리할 때까지 KV가 유지되는 점과 P→D KV 복사이며, 두 arm의 수치 경로를 비교한다.

## H2 · IDEFICS의 projected K/V handoff

원본 HF Cross-attention은 이미지 latent를 정적으로 유지하면서 과거 Cross K/V를 재사용하지 않는다. 이미지당 raw Perceiver state는 BF16 64×1,280, 160 KiB다. 8개 Cross Block 전체에서 각 32 heads×128의 K/V는 BF16 합계 8 MiB다. Query와 실제 Attention/FFN은 현재 Language hidden에 의존한다. P에서 원본 K/V와 Key RMSNorm 계산을 끝내고 전달하는 arm을, D가 raw state를 한 번 받아 자기 GPU에서 같은 projection을 cache하는 대조와 비교한다. 가설은 **초기 추가 전송·packing과 D 재계산/일회 projection의 차이에서 순이익이 있는가**이다. 매 Token의 원격 Attention은 후보에 넣지 않는다.

## H3 · IDEFICS의 Decode-only 이미지 State Liveness

초기 Prompt에는 순서대로 여러 이미지가 등장할 수 있다. Processor의 모달 Mask가 Query별 가시 이미지 집합을 만들고 Generation path는 마지막 Prompt의 image mask row를 확장한다. P의 모든 Prompt Query는 원래 Mask로 계산한다. D만 고정된 Visible set의 raw 또는 projected visual state를 갖고, 이미 계산한 Text Self KV는 전부 이어받는다.

Source-level mask invariance와 1/2/4/16개 실제 이미지 사례가 첫 검증이다. State를 인덱싱하는 비용, GPU 전송, Self KV 크기, unmasked/packed attention의 BF16 차이, Mixed request shapes를 측정한다. 동일 Mask를 쓰는 generic liveness/packing이 같은 이득을 내면 modality 전용 정책의 추가 이득은 미확정이다. 새 이미지 입력을 포함한 미래 턴은 이 경계의 생존 범위를 다시 계산해야 한다.

## H4 · 독립적으로 찾은 Encoder 내부 경계

DeepSeek-OCR2는 OCR Decoder 앞의 Visual Encoder에서 Image Patch가 뒤따르는 causal flow query를 참조하지 않는 Mask를 사용한다. Prefix Image State를 독립적인 Encoder 작업으로 준비할 수 있다. 뒤쪽 flow query는 선행 Image Patch를 참조하므로 동일 요청 내에서 두 stage를 동시에 완료할 수 없다. 일반 causal prefix 및 E pipeline이 가장 강한 대조다. 이 실험에서는 공개 Source을 검토했다. Cross-stage Activation과 동시 Batch의 원본 OCR 품질 비용이 남아 GPU 후보보다 우선순위를 낮췄다.

BAGEL은 다른 방식으로 modality 출력에 따라 MoT Expert Bank가 달라지는 Source 근거를 제공한다. Image Understanding 경로의 installed vLLM이 이미 generation-only 가중치를 배제해, 해당 Bank의 전용 제거를 신규 실측 후보로 선정하지 않았다. [독립 Source 후보와 관련 연구](mllm_modality_dependent_architecture_landscape.md) · [Novelty 대조](mllm_architecture_prior_work_novelty_audit.md)

## H5 · Shared QKV를 보존하는 ERNIE-VL의 별도 FFN pool

ERNIE의 VLLM Forward는 식별된 visual placeholder row가 하나 이상 있을 때만 vision FFN expert bank를 실행한다. 일반 Top-k 선택은 해당 pool 내부에서 진행된다. 모든 row가 shared self-attention과 원래 MRoPE를 사용하므로 KV를 생략하거나 다른 장치로 흩을 이유가 없다. 행에 들어올 modality를 보장하는 D에서 vision FFN pool만 배제할 수 있는지가 별도 가설이다. PT source와 Thinking checkpoint를 구분하고 생성용 Thinking을 선택했다.

강한 대조는 같은 installed Native에서 text-only 요청과 image→text 요청의 FFN 호출 추적, 그리고 scope-aware generic reachability다. Image와 Text의 같은 시점 mixed serving에서 visual-prefill row가 D에 입장하는 경로는 이를 기각할 조건이다. Checkpoint의 vision FFN 및 router만 합쳐 13,607,239,680 B라는 값은 weight audit다. 실제 Native가 기록한 GPU residency를 별도로 검증한다. Checkpoint 로더의 Stage별 Weight elision, production fused path, KV capacity 및 장기 출력 SLO는 이 구조 대조와 구분한다. [ERNIE checkpoint와 역할별 payload](mllm_modality_dependent_architecture_landscape.md)
