# Part 1 — 검증된 MLLM MoE EP baseline과 자원 적응형 Virtual EP

문서 버전: 1.0 / 2026-09-24  
상태: **Codex가 구현·실행할 실험 계약. 신규 GPU 결과나 simulator 구현 완료를 의미하지 않는다.**  
모델: `Qwen/Qwen3-VL-30B-A3B-Instruct` / BF16  
현재 허용 GPU: physical `4,5,6,7`  
관련 Part 2: `02_resolution_expert_tradeoff_spec.md`

## 0. 목적과 우선순위

이 문서는 사용자의 요구 ① 올바른 MLLM MoE EP와 공식 보고 성능에 가까운 baseline 검증, ③ GPU 1/2장만 남아도 재사용 가능한 MLLM virtual EP를 담당한다.

이번 Part 1은 메소드 성공 여부와 무관한 연구 인프라 작업이다. 해상도–expert-budget 가설이 나중에 NO-GO여도, 검증된 evaluator/trace/simulator는 남겨야 한다. 기준 모델이 잘못 작동하는데 방법 성능을 평가하거나, simulator 오류를 방법의 NO-GO로 해석하지 않는다.

완료 목표:

1. 정확한 checkpoint·processor·generation·evaluator 계약을 고정한다.
2. 정상적인 이미지 질문을 답변 완료까지 생성하여 공개 점수와 비교한다.
3. 실제 EP4에서 expert dispatch/compute/combine을 확인하고 reference와 비교한다.
4. 모델의 **semantic 실행**, **virtual ownership/replay**, **latency 예측**을 분리한다.
5. 4GPU가 있을 때 calibration anchor를 확보하고, 1/2GPU 전환 smoke를 실제로 수행한다.
6. Part 2에 재사용할 frozen baseline bundle을 만든다.

**금지:** 이 문서의 목표 점수·예시·CLI 계약을 측정 결과로 보고하기. GPU 4장이 있다는 이유만으로 TRUE EP4라고 쓰기. Fixed-32 생성 시간을 정답률 검증으로 사용하기. 동일 모델의 과거 dLLM calibration을 재사용하기.

## 1. 근거와 사전 정정

### 1.1 공개 checkpoint의 기대값 — 실제 tensor로 다시 확인

[S1] 공개 config 기준:

| 항목 | 기대값 |
|---|---:|
| 모델 형식 | Qwen3VLMoeForConditionalGeneration |
| hidden d | 2048 |
| routed expert intermediate m | 768 |
| experts E | 128 |
| expert top-k | 8 |
| decoder layers | 48 |
| vision patch size / spatial merge | 16 / 2 |
| precision | BF16 |

로컬 snapshot 기대값은 기존 사용자 실험의 `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`. 설치된 revision이 다르면 숨기지 말고 diff와 변경 이유를 기록한다. expert W_gate/W_up/W_down의 실제 orientation, fused packing, TP shard 여부를 모두 assert한다. m=512 또는 1024를 Qwen expert dimension으로 사용하지 않는다. routed expert는 존재한다고 확인하되 shared expert 유무는 코드/tensor를 읽어 확정한다.

### 1.2 이미 존재하는 expert-count 조절

- FastMMoE [S6]: visual `K_v=pK`를 사용하고 선택 weight를 재정규화하며 text는 full routing. Token pruning도 별도 결합한다.
- MoDES [S7]: 원래 top-k 안에서 layer/modality-aware importance threshold로 expert를 skip한다. Config의 top-k가 고정이어도 effective executed k는 달라진다.
- MACS [S8]: expert capacity, overflow rerouting/drop, local expansion을 사용한다. 단순 per-token top-k schedule과 같지는 않지만 실행 expert 집합/개수를 불변으로 유지하는 방법도 아니다.
- ReaLB [S9]: 주요 조절 대상은 EP-rank별 execution precision이다. 주요 기여를 dynamic top-k로 설명하지 않는다.

따라서 Part 2는 'vision top-k 최초 조절'이 아니라 resolution × visual-expert budget의 상호작용과 EP critical-path 선택 가치를 검증한다.

### 1.3 현재 repository에 대한 주의

대화의 최신 Spatial-Basis 보고는 TRUE EP4, TP2/DP2 계열 DPEP, DeepEP high-throughput, sparse Triton expert 경로를 기술한다. 원격에서 확인 가능한 과거 `poc_flashvep/reports/batch16_32_quick_poc.md`는 다른 시점의 all-gatherv/reduce-scatterv 경로와 1-token smoke를 기록한다. 둘을 같은 현재 baseline으로 취급하지 않는다. **로컬 실제 소스·process group·runtime call trace가 최종 기준**이다.

## 2. 파일 배치와 변경 원칙

새 root를 repo 내부 `poc_mllm_ep_foundation/`으로 제안한다. 최초 audit에서 실제 위치를 확정한다.

```text
poc_mllm_ep_foundation/
  configs/
  evaluation/
  instrumentation/
  virtual_ep/
  scripts/
  tests/
  artifacts/{manifest,quality,anchors,traces,calibration,sim_validation}/
  reports/
```

기존 dLLM/Spatial-Basis 파일과 보고서를 덮어쓰지 않는다. 기존 runner는 opt-in wrapper/hook만 변경한다. git dirty 상태를 저장하고 unrelated change를 commit하지 않는다. Runtime 전체 업그레이드부터 시작하지 않는다. Native CLI 이름은 설치한 버전의 help/소스에서 확인한 뒤 사용한다.

## 3. GPU 자원 계약 — 4/2/1장을 서로 다른 evidence level로 운용

`--physical-gpus`와 `--target-ep`를 반드시 분리한다.

| Physical resource | 가능한 기본 작업 | 불가능한 주장 |
|---|---|---|
| 4 GPUs | stock reference, TRUE EP4, anchor 수집, virtual validation | 4 replicas를 EP4라고 부르기 |
| 2 GPUs | 모델이 들어가면 TP2 등 semantic 실행, TRUE EP2 가능 시 보조 검증, virtual EP4/8 | actual EP4/8 latency |
| 1 GPU | 적재 가능 시 전체 BF16 semantic 실행, virtual-rank serial replay, kernel calibration | serial rank time 합을 EP4 wall로 사용 |
| 전체 모델 적재 불가 | 기존 trace replay / tensor-level calibration | 새 policy의 quality를 trace만으로 생성 |

현재 기본 `CUDA_VISIBLE_DEVICES=4,5,6,7`. 2/1GPU mode는 사용자가 여전히 허용한 subset만 선택한다. 기본 smoke subset은 `4,5` 및 `4`, 실제 사용 전 허용/점유를 확인한다.

- 모델이 한 장에 들어간다고 보장하지 않는다. 실제 weight+KV+activation+workspace 메모리를 확인한다.
- GPU가 줄었다고 FP8/4bit, 작은 모델, 낮은 해상도로 조용히 바꾸지 않는다.
- BF16 diagnostic CPU staging이 꼭 필요하면 `SEMANTIC_CPU_STAGED_NOT_SERVING`으로 명시하고 wall time을 성능 비교에서 제외한다. 우선은 기존 trace mode를 제공한다.
- GPU 회수 시 active job이 안전하게 checkpoint하고 종료하도록 한다. 이동 가능한 단계 경계에서 4→2→1로 재시작하는 것이지 live elastic EP를 구현하는 것은 아니다.
- request별 atomic JSONL 저장, completed-ID resume, artifact hash 검사, RNG와 policy ID를 보존한다.
- 다른 사용자/GPU의 프로세스를 종료하지 않는다. 이전 burner 요청은 사용자 소유 process임이 확인될 때만 처리하고, timing 동안 경쟁 workload가 없어야 한다. burner 존재/실행을 throughput 증거로 쓰지 않는다.

## 4. Phase 0 — 모델·데이터·sampling contract

다음을 `reports/model_and_eval_contract.md`와 machine-readable manifest에 저장한다.

```text
model_id, local_snapshot, revision, config/weight hashes
processor/tokenizer revision, chat_template hash
transformers/vLLM/torch/CUDA/DeepEP/NCCL versions
GPU model/UUID/HBM/topology/clocks
expert dimensions/activation/weight layout/all routed layer IDs
actual router softmax, normalization, ties, selected weights
image token IDs, special-token IDs, patch/merge/grid mapping
input image hashes, decoded RGB/orientation, image order
resize path, min/max pixels, final width/height, image_grid_thw
actual visual/text/special/padding token counts
DeepStack feature alignment and injection behavior
all generation kwargs, stop/EOS policy, RNG rules
metric implementation, dataset revision/split/IDs
```

### 4.1 이미지가 실제로 사용되는지

16개 이하의 수동 점검용 image-dependent 질문으로 smoke한다. OCR/숫자/색/개수/공간 예시를 포함하고 실제 답변 전체를 저장한다. 이미지 없이도 답할 수 있는 질문을 negative control의 근거로 쓰지 않는다.

다음을 검사한다.

- missing/corrupt image를 무음 무시하지 않는가?
- image placeholder와 encoder feature 수가 일치하는가?
- EXIF 회전/RGB/패딩/이미지 순서가 맞는가?
- preprocessing을 두 번 수행하지 않는가? qwen-vl-utils가 resize했으면 processor의 중복 resize를 방지한다 [S2].
- 같은 질문+잘못된 이미지/blank image control이 의도대로 입력을 바꾸는가? 이 control은 plumbing 검사이며 공식 점수에 섞지 않는다.
- 출력에 prompt가 섞이지 않는가? generated-token slicing, EOS, repeated loops, empty answer를 검사한다.
- decoding k와 expert k를 혼동하지 않는가? **sampling_top_k=20 ≠ expert_top_k=8**.

### 4.2 두 가지 decoding lane

**Lane O — OFFICIAL_REPRODUCTION**

[S2]의 Instruct 재현 설정을 출발점으로 사용한다.

```text
greedy=false / do_sample=true
seed=3407
temperature=0.7
top_p=0.8
sampling_top_k=20
repetition_penalty=1.0
presence_penalty=1.5
max_new_tokens=32768
```

이는 매번 32768개를 생성하라는 뜻이 아니라 EOS까지 생성하되 해당 상한을 사용하는 설정이다. Benchmark-specific 공개 wrapper가 더 구체적인 값을 지정하면 그 source/version을 기록하고 우선순위를 설명한다. 메모리 때문에 상한/해상도를 줄여야 하면 `PROTOCOL_DEVIATION`이며 공식 동일 조건 재현이라고 부르지 않는다.

**Lane D — DETERMINISTIC_PARITY**

Numerical/debugging 및 controlled systems 비교를 위한 greedy lane을 따로 둔다. 동등 입력의 teacher-forced logits, sparse MoE output, next-token argmax를 검사한다. Lane D 점수를 Lane O 공식 reported score와 직접 등치하지 않는다. 같은 seed만으로 서로 다른 runtime의 stochastic generated strings가 bit-exact해야 한다고 요구하지 않는다.

## 5. Phase 1 — 공개 성능의 실제 재현

### 5.1 보고 점수 registry

[S3] Table 3, printed p.16, `Qwen3-VL-30B-A3B / instruct` 열에서 확인한 reference:

| Benchmark | Reported score | 중요 caveat |
|---|---:|---|
| MMStar | 72.1 | 정확한 split/evaluator/prompt 확인 |
| ChartQA test | 86.8 | 공식 relaxed scoring 및 subset aggregation 확인 |
| RealWorldQA | 73.7 | 해당 공개 evaluator 기준 |
| MMMU | 74.2 | MMMU-Pro 60.4와 구별; multi-image 제외 금지 |
| OCRBench | 903 | raw benchmark score이며 903%가 아님 |
| DocVQA test | 95.0 | val 점수를 test score와 직접 비교 금지 |

다른 논문이 자체 평가한 vanilla score를 Qwen 공식 score로 대체하지 않는다. 정확한 공개 image resolution 설정이 불명확하면 unknown을 남기고 알려진 wrapper/processor 설정을 고정한다. 원하는 점수가 나올 때까지 test-set에서 prompt/해상도를 튜닝하지 않는다.

Registry field:

```text
reported_source/version/table/model-column/score/metric
public split/revision/full item count
prompt template hash, image policy, generation policy
judge/parser/normalization details
EXACT_PROTOCOL / PARTIALLY_DOCUMENTED / NONCOMPARABLE
```

### 5.2 추천 기본 benchmark

**MMStar + ChartQA를 두 개의 full public anchor로 우선**한다. 실제 item count는 다운로드한 revision의 split을 읽어 기록한다. available full test split을 임의 32개로 잘라 headline과 비교하지 않는다.

- 초기 plumbing: 각 task 고정 32–64개. smoke일 뿐 reproduction PASS 금지.
- 본 재현: 두 benchmark full public split, stock reference와 TRUE EP4 각각 완료.
- OCR/detail 확장: OCRBench 또는 정답 공개된 DocVQA/TextVQA split. DocVQA val은 paired-local test로 쓰되 reported test95.0 재현이라고 부르지 않는다.
- MMMU는 secondary. 필요 시 원래 multi-image item도 보존한다. Part 2가 단일 이미지라 해서 baseline benchmark를 조용히 필터링하지 않는다.
- 외부 judge가 필요하면 비용/자격증명 허용 없이는 호출하지 않는다. Rule-only 결과는 다른 evaluator라고 명시하고, judge가 필요 없는 anchor를 우선한다.

### 5.3 판정 — 점수 차이와 protocol 차이를 분리

`PUBLIC_REPRODUCTION`과 `EP_REFERENCE_PARITY`는 별도 필드다.

**Public reproduction 운영 기준:** full split·비교 가능한 protocol에서 각 primary task가 보고 score보다 3.0 normalized points 이상 낮지 않은지 확인한다. 이 3pp는 새로 정한 운영 tolerance이지 모델의 이론적 오차 한계가 아니다. 표본/seed 불확실성과 CI를 함께 보고한다. 큰 초과 성능도 leakage/parser/단위 오류를 감사한다.

**EP parity:** 같은 data/protocol의 stock baseline과 EP4를 paired 비교한다. 목표는 score loss가 없고, -1.0pp non-inferiority margin의 paired 95% CI로 배제 가능한지 보는 것이다. CI가 넓으면 `QUALITY_UNRESOLVED`; 차이가 불명확하다는 이유만으로 동등성 PASS 금지. 모든 answer string exact agreement를 요구하지는 않되 deterministic tensor/logit 차이와 stochastic 평가 차이를 분리한다.

공식 prompt/해상도/metric이 불완전하면 `LOCAL_BASELINE_VALIDATED / PUBLIC_REPRO_PARTIAL`로 보고한다. 이러한 결과는 재현 실패와 다르지만 'reported와 동일'이라고 주장할 수 없다. Part 2의 smoke diagnosis는 가능해도 formal method promotion은 baseline 미해결 상태에서 하지 않는다.

### 5.4 실패 시 audit 순서

데이터·정답 연결 → image preprocessing/grid → prompt/template → sampler/penalty → max context/EOS/truncation → evaluator/parser → checkpoint/type(Thinking/FP8 혼동) → EP mapping/numerics 순서로 조사한다.

문제 case와 수정 diff를 저장한 후 baseline을 다시 생성한다. test answer를 이용해 정답을 고쳐 넣지 않는다. `BASELINE_INVALID`는 method NO-GO가 아니다.

## 6. Phase 2 — 진짜 EP4 증명과 reference parity

현재 검증된 runtime이 있으면 유지한다. 사용자 최근 보고의 TP2/DP2/PP1/EP4를 출발점으로 audit하되 SP1 표기와 내부 sequence-sharding의 의미를 코드로 확인한다. vLLM은 일반적으로 EP group을 TP×DP로 구성한다 [S4]. TP2·DP2·EP4를 곱해 16GPU라고 계산하지 않는다.

필수 증거:

1. 4 distinct physical UUID와 실제 process-group membership.
2. 128 experts가 EP4에 어떻게 배치되는지. contiguous라면 32/rank, tensor shape와 expert ID map으로 확인.
3. logical source token ownership. 한 요청이 어느 DP group에서 처리되고 TP 안에서 어떻게 중복/분할되는지.
4. 실제 dispatch/combine backend call, send/recv histogram, nonzero cross-rank payload.
5. missing/duplicate logical token 없음. dummy/padding/TP copies는 실제 token과 분리.
6. selected routed assignments 수가 actual per-token k 합과 일치. expert zero-weight만 만들고 full work를 실행한 경우 별도 표시.
7. all-dense/top8 semantic reference 대비 expert output/logits 검증. Dtype별 tolerance는 stock-repeat/FP32 소규모 reference noise를 먼저 측정하여 고정한다.

네 independent replicas는 semantic throughput 도구일 뿐 EP4가 아니다. TP4 final all-reduce만 있고 token dispatch가 없는 경로도 DPEP A2A와 별도 타입으로 기록한다. `--enable-expert-parallel` flag 존재만으로 증명 완료하지 않는다.

### Backend

Primary는 현재 동작하는 optimized sparse backend(최근 보고는 DeepEP HT + TritonExperts). Optional NCCL/allgather-reducescatter는 실제 installed 지원 범위에서만 추가한다. Backend별 baseline/calibration을 분리한다. Naive backend 대비 gain으로 best available backend보다 빠르다고 결론내리지 않는다.

## 7. Phase 3 — 현재 MLLM의 전체 latency breakdown

이 단계는 Part 2 가설과 독립적으로 완료한다. 이전처럼 method structural gate 실패 때문에 baseline profiler를 생략하지 않는다.

### Workload

- batch1 primary, distinct-image batch4 secondary; DP 분배와 effective active calls를 기록.
- 실제 이미지·질문, native/reference resolution, actual visual tokens의 low/mid/high bins.
- TTFT, natural EOS E2E, fixed-decode 32/128을 분리.
- fixed-decode는 P0가 만든 동일 token sequence를 teacher force하여 시스템 비용을 비교한다. fixed32는 quality evaluation이 아니다.
- 모델 로딩·dataset 다운로드 제외. input-ready CPU tensors→first token/end 경계를 명시. decode-only·LLM-prefill-only를 TTFT라고 부르지 않는다.

Components:

```text
image decode/resize/H2D
vision encoder/projector/DeepStack preparation
language prefill attention
router
counts/offsets/route layout/pack
EP dispatch
routed gate/up + activation + down
EP combine/scatter
TP collectives / host gaps / other
autoregressive text decode and LM head
```

Fused expert time을 2:1 산술로 gate/up·down ms에 나누지 않는다. Replay estimate이면 `ESTIMATED_COMPONENT_SPLIT`. Overlap된 stage sums와 clean E2E의 차이는 residual/overlap 항으로 표시한다.

- clean paired timing과 profiled run 분리; 동일 input/shape/token trace.
- microbench warmup≥10, repeats≥30; full-request paired repeats≥5 기본, long run은 실제 횟수 보고.
- P0–candidate order interleave; clock/thermal/competing processes 기록.
- profiler overhead 목표≤5%, >10%이면 gate용 breakdown으로 쓰지 않고 계측을 수정.
- GPU-local timestamp를 서로 다른 device의 공통 절대 clock으로 간주하지 않는다.

## 8. Trace schema — 이후 1GPU에서도 쓰이는 핵심 자산

Trace는 immutable, versioned, request/image-hash keyed로 보관한다.

```text
model/revision/weight-shape hash, runtime/backend/topology signature
request_id/image_hash/task/split/prompt hash/generation lane
resolution_policy, actual resized shape/image_grid_thw
step_id, prefill/decode, chunk_id, layer_id, invocation_id
logical token_id, sequence pos, visual/text/special/padding flag
source owner, source TP-copy flag, expert owner map
current top8 expert IDs and weights, effective executed k
per-expert actual rows/padded rows, rank histograms
source->destination deduplicated token sets / bytes / metadata
current hidden samples and exact expert-input dtype where required
KV length, actual output token sequence/EOS/finish reason
per-rank component times + clean request timing linkage
```

모든 48 layers는 lightweight route/count/cost metadata를 저장한다. Hidden은 `[0,3,7,15,23,31,39,47]` 등 bounded layers/cases부터 수집하고 trace cap을 manifest에 적는다. 실제 선정되지 않은 layer의 gain은 측정 없이 48배 extrapolate하지 않는다.

Part 2를 대비해 global router top8 raw/normalized weights를 구별해 저장한다. 해상도가 바뀌면 processor/encoder/hidden/route가 모두 달라질 수 있으므로 resolution별 새로운 trace가 필요하다.

## 9. Virtual EP — 세 계층을 분리하여 구현

### V1. Semantic runner: 모델 정답률의 근거

모든 weights와 attention/vision 계산을 실제 실행하여 최종 답변을 만든다. 적재 가능한 1GPU 또는 TP2/다른 검증된 2GPU 경로를 사용한다. Target EP4/8를 'virtual ownership'으로 attach할 수 있지만, 여기의 physical wall은 target EP latency가 아니다.

Virtual P개의 expert owner를 순서대로 처리할 때 입력 row를 expert별로 묶고 실제 weight로 계산한 뒤 원래 token 순서로 합친다. 수학적 라우팅이 같으면 ownership 변경은 계산 분배만 바꾼다. BF16 reduction order 차이는 수치 검사 대상으로 남긴다.

**새 k/해상도 정책의 정답률은 full semantic rollout에서만 얻는다.** Baseline trace에서 expert IDs 몇 개를 지운 것으로 downstream accuracy/routing drift를 추정하지 않는다.

### V2. Routing/layout replay: workload의 근거

실제 trace에서 target P의 source-owner map과 expert-owner map을 지정하여:

- per-expert/per-rank histograms;
- logical assignments와 physical padding;
- unique token-destination sets, source–destination traffic matrix;
- wire replication/dedup/padding/metadata;
- rank max/mean, expert group sizes;
- chunk scheduling과 idle DP dummy work

를 재현한다. Synthetic uniform/random routes는 unittest/control에만 사용한다. 실제 quality/workload에 uniform router simulation을 켜지 않는다.

### V3. Cost simulator: latency 예측의 근거

각 rank의 actual histogram을 real-shape kernel model에 넣고 backend-specific communication/scheduler dependency를 결합한다. 간단한 base model의 예:

```text
T_stage ≈ T_prepare + T_dispatch + max_r(T_expert,r) + T_combine + T_finalize
```

이 식은 자동 truth가 아니다. 설치 backend가 overlap/fused path이면 그 측정 경계에 맞는 event/dependency model로 교정한다. Rank expert max를 쓰면서 통신에서 이미 포함한 wait를 다시 더하지 않는다.

전체 E2E는 실제/검증된 vision, attention, decode, scheduling profile과 합성한다. 다른 resolution의 encoder/attention/decode 시간을 고정해 놓고 EP만 줄여 E2E를 만들지 않는다. Non-EP model이 검증되지 않으면 output은 routed-stage prediction만 제공하고 E2E는 `N/A`다.

### Source ownership / target topology

Target EP4 primary는 actual verified TP/DP/source mapping을 그대로 사용한다. EP8 extrapolation은 예컨대 TP2/DP4처럼 정확한 virtual target topology를 명시하고 **global request batch를 그대로 고정**한다. 자동으로 workload를 2배 늘리거나 TP8/SP8로 바꾸지 않는다. 새로운 topology의 non-MoE time은 검증 전 N/A/interval이다.

## 10. Calibration — shape/collective/profile 오류 방지

### Compute calibration

Key에 최소 다음을 넣는다.

```text
GPU architecture, d=2048, m=768, E_local, weight layout/dtype
kernel/version, sorted count histogram, active experts
rows/expert, padding/tile counts, effective_k/dispatch format
mode(prefill/decode), launch count, graph/eager
```

- 0/1/2/4/8/16/32/64/128… rows와 실제 observed histogram을 포함한다.
- 모든 expert가 같은 n인 synthetic GEMM만으로 uneven grouped GEMM을 예측하지 않는다.
- actual post-policy histograms(Part 2에서는 정책별 re-collected traces)에 적용한다.
- nearest-neighbor 또는 regression을 써도 OOD distance와 held-out residual을 출력한다. 5-nearest median을 deterministic lower/upper bound라고 부르지 않는다.
- unseen shape는 측정 요청 또는 `OUT_OF_DOMAIN`이지 extrapolated point estimate 확정이 아니다.

### Communication calibration

- physical EP4에서 send/recv peer count, skew, payload size, metadata/startup, dedup를 같이 측정.
- DeepEP와 NCCL family를 별도 모델로 둔다.
- two-GPU NVLink bandwidth 한 숫자를 EP4/EP8 all-to-all로 대체하지 않는다.
- peak link bandwidth로 시간을 결정하지 않는다.
- dispatch와 combine의 byte 단위를 실제 backend contract에 맞춰 별도로 센다.

### Validation split

Calibration과 validation은 request/image 단위로 분리한다. 동영상은 제외. 동일 이미지의 다른 layer를 validation으로 넣고 'held-out input generalization'이라고 쓰지 않는다.

최소 초기 anchor: native/reference 24 distinct requests(가능하면 task/token bins 균형), calibration16 + heldout8. Timing economics 검증을 위해 raw row-count 감소/kv4/6 등 representative real shape replay를 포함할 수 있으나 이는 Part 2의 quality pass가 아니다.

필수 metric:

```text
median/P90 APE, ms absolute error, per-stage signed bias
predicted vs actual stage/request latency
prediction interval empirical coverage
ranking of configurations, sign accuracy of material speedups
OOD count and invalid interval count
```

연구 운영 목표: routed-stage median APE≤10%, P90≤20%. 5% 이상의 실제 configuration 차이에 대해 gain sign accuracy≥90%와 Spearman≥0.8을 목표로 한다. 작아 잡음 수준인 차이에 순위 gate를 강제하지 않는다. 통과하지 못하면 `SIM_UNVALIDATED`, 품질 실험은 유지하되 simulator로 tiny gain/NO-GO를 확정하지 않는다.

EP8는 actual EP8가 없으면 **항상 projection**. EP4 correction scalar 하나로 '검증된 EP8'를 만들지 않는다. Raw prediction, EP4 bias-corrected sensitivity, uncertainty를 병기한다.

## 11. 4GPU가 있을 때 먼저 보존할 것

모델 contract와 작은 parity가 통과하면, 긴 full quality run과 병행/전후 자원 사정에 맞추어 EP4 anchor bundle을 우선 확보한다. GPU 확보가 끝날 때까지 official full-quality 결과가 미완료이면 그 상태를 명시하고 1/2GPU로 이어간다.

```text
environment lock / model shape manifest
process-group + expert/source ownership map
clean full-model EP4 timings and component profiles
raw actual routes/counts + scoped H samples
real grouped-expert timing cache
actual four-rank dispatch/combine measurements
calibration/heldout split IDs
reference outputs/evaluator source snapshot
```

역사적 data는 새 signature가 맞을 때만 재사용한다. Backend/GPU/weight shape/precision/kernel/processor 변경 시 해당 cache를 invalidate한다. Method failure 때문에 anchor를 삭제하지 않는다.

## 12. 4→2→1 자원 전환 acceptance test

동일 smoke image IDs와 Lane D teacher-forced outputs를 사용한다.

1. TRUE EP4 reference를 저장한다.
2. 2GPU semantic runner + virtual EP4를 실행한다.
3. 적재 가능 시 1GPU semantic runner + virtual EP4를 실행한다. 불가하면 replay-only smoke.
4. virtual workload의 logical counts/bytes가 동일 target EP4 trace와 일치하는지 비교한다.
5. 새 1GPU microbench가 기존 EP4 anchor의 cost model prediction을 훼손하지 않는지 확인한다.
6. interrupted/resumed 실행의 ID 중복·누락·RNG 오염이 없는지 테스트한다.

메모리 부족은 `SEMANTIC_UNAVAILABLE_MEMORY`이며 모델/방법 실패가 아니다. 1/2GPU local latency를 EP4 truth 열에 넣으면 test fail.

## 13. 구현해야 할 CLI 계약 — 기존 native 명령이라고 가정하지 말 것

Codex는 다음과 동등한 기능을 실제 구현하고 README/report에 검증된 실행 명령을 남긴다.

```text
audit_environment --physical-gpus 4,5,6,7
evaluate_baseline --lane official|deterministic --backend reference|physical_ep
capture_anchor --physical-gpus 4,5,6,7 --manifest <frozen>
virtual_replay --physical-gpus 4 --target-ep 4|8 --target-topology <manifest>
calibrate_cost --backend <actual> --shape-manifest <verified>
validate_sim --holdout-manifest <image-disjoint>
resume_experiment --physical-gpus 4,5 --completed-index <...>
```

모든 command는 dry-run manifest print, resume, error log, incomplete status를 지원한다. Spec 문구를 출력하는 stub를 simulator로 보고하지 않는다.

## 14. Tests

필수 unit/integration tests:

- source/expert owner maps bijective/valid, EP groups correct;
- dtype/shape assertion 및 2048×512/4096×1024 stale calibration 거부;
- logical route counts = sum(k_i), duplicate TP copies 별도;
- same input sorted-top8 subset selection, weight renorm, text/decode top8 유지;
- true variable-k compact work vs masked padded-work 구별;
- per-source token-destination dedup와 combine semantics;
- zero routes, partial chunks, empty DP request, uneven/one-hot rank load;
- wrong image/missing placeholder fail loud;
- matched semantic/replay no-op parity;
- per-policy trace key 분리, baseline trace reuse leakage 금지;
- cache signature invalidation, OOD tagging;
- four serial virtual ranks의 sum wall을 parallel latency로 쓰지 않음;
- confidence interval/wall boundary/energy-unit label consistency;
- resource drop/resume 결과 누락/중복 없음.

## 15. Part 2 승격 및 상태 분리

```text
MODEL_CONTRACT: PASS / BLOCKED
IMAGE_PIPELINE: PASS / BLOCKED
PUBLIC_REPRODUCTION: MATCHED / PARTIAL_PROTOCOL / OUTSIDE_TOLERANCE / NOT_RUN
REFERENCE_VS_EP_QUALITY: PASS / UNRESOLVED / FAIL / NOT_RUN
PHYSICAL_EP4_PROOF: PASS / FAIL / UNAVAILABLE
VIRTUAL_SEMANTIC_PARITY: PASS / FAIL / MEMORY_LIMITED
SIM_EP4_VALIDATION: PASS / FAIL / NOT_RUN
SIM_EP8: UNVALIDATED_EXTRAPOLATION / VALIDATED_WITH_EP8
RESOURCE_FALLBACK: TESTED_4_2_1 / PARTIAL
READY_FOR_PART2: YES / DIAGNOSTIC_ONLY / NO
```

정상 baseline + paired local quality가 확보돼야 method quality 실험을 본격 진행한다. Public protocol 일부 미공개인 경우 Part 2를 진단용으로 실행할 수는 있지만 최종 '공식 성능 유지' 주장은 보류한다. 실제 EP4가 잠시 없을 때 calibration과 semantic evidence가 충분하면 `PROJECTED_ONLY` 연구를 계속한다. Formal EP speedup GO는 physical validation 없이 하지 않는다.

## 16. 필수 보고서와 인계

```text
reports/model_and_eval_contract.md
reports/public_reproduction.md
reports/reference_vs_ep4_quality.md
reports/ep4_path_and_ownership.md
reports/full_model_baseline_breakdown.md
reports/virtual_ep_architecture.md
reports/calibration_shapes_and_backends.md
reports/simulator_holdout_validation.md
reports/resource_fallback_4_2_1.md
reports/baseline_foundation_summary.md
artifacts/frozen_baseline_bundle.json
```

Bundle에 absolute+relative root, dataset IDs, generation/processor signature, official reference score/출처, validated backend, source/expert maps, calibration domain, schema_version, evidence labels, unresolved items를 넣는다. Part 2가 raw paper numbers를 덮어쓰거나 baseline을 재선정하지 못하게 한다.

## 17. Final Required Summary

```text
PHYSICAL_GPU_SET / AVAILABLE_GPU_COUNT:
MODEL_REVISION / ACTUAL_EXPERT_SHAPES:
ACTUAL_TP_DP_EP_PP_AND_SOURCE_MAPPING:
REFERENCE_RUNTIME / EP_BACKEND:
OFFICIAL_GENERATION_LANE:
SAMPLING_TOP_K / EXPERT_TOP_K:
DATASET_PROTOCOL_AND_COUNTS:
MMSTAR_REPORTED_REFERENCE / LOCAL_REFERENCE / ACTUAL_EP4:
CHARTQA_REPORTED_REFERENCE / LOCAL_REFERENCE / ACTUAL_EP4:
OTHER_TASKS_AND_METRICS:
PUBLIC_REPRODUCTION_STATUS:
EP_VS_REFERENCE_PAIRED_DELTA_AND_CI:
BASELINE_TTFT / NATURAL_E2E / FIXED32 / FIXED128:
ROUTED_STAGE_E2E_SHARE / PROFILER_OVERHEAD:
PHYSICAL_EP4_PROOF:
VIRTUAL_EP4_SEMANTIC_AND_WORKLOAD_PARITY:
SIM_EP4_MEDIAN_APE / P90_APE / RANKING_AND_SIGN:
SIM_EP8_EVIDENCE_LABEL:
ONE_GPU_MODE / TWO_GPU_MODE / FALLBACK_TEST:
MODEL_MEMORY_LIMITS:
FROZEN_BASELINE_BUNDLE_PATH:
READY_FOR_PART2:
PRIMARY_BLOCKER / NEXT_ACTION:
NEW_OPTIMIZATION_IMPLEMENTED: NO (Part 1)
```

## 18. 외부 근거 — 실행 시 source/version 고정

- [S1] Qwen official checkpoint config: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/raw/main/config.json
- [S2] Qwen3-VL official repository, Evaluation Reproduction / image processing: https://github.com/QwenLM/Qwen3-VL
- [S3] Qwen3-VL Technical Report, Table 3 (printed p.16), Appendix B: https://arxiv.org/abs/2511.21631
- [S4] vLLM Expert Parallel Deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/ (실행에는 설치 version 문서 확인)
- [S5] Evaluation frameworks: https://github.com/open-compass/VLMEvalKit ; https://github.com/EvolvingLMMs-Lab/lmms-eval
- [S6] FastMMoE, §3.1 / Eq.3–5: https://arxiv.org/html/2511.17885v2
- [S7] MoDES, §5: https://arxiv.org/html/2511.15690v2 ; https://github.com/ModelTC/MoDES
- [S8] MACS, §3 and Appendix A.5: https://arxiv.org/html/2605.05225
- [S9] ReaLB, runtime precision policy: https://arxiv.org/html/2604.19503

공개 문서는 검증한 사실의 출처이고, 이 문서의 tolerance·실험 순서·simulator 설계는 새로 제안한 연구 운영 기준이다.
