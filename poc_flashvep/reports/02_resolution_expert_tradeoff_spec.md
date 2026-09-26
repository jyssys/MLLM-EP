# Part 2 — Visual Resolution × Expert Budget Trade-off under MoE EP

문서 버전: 1.0 / 2026-09-24  
상태: **discovery 계획. 신규성·정확도·가속은 아직 입증되지 않았다.**  
선행 문서: `01_baseline_virtual_ep_spec.md`  
대상: 단일 이미지 / Qwen3-VL-30B-A3B-Instruct / training-free  
Primary hardware: physical GPU `4,5,6,7`, verified TRUE EP4  
Reduced hardware: Part 1의 semantic runner + virtual EP4/8, 명시적인 projection label

## 0. 이 연구 질문은 무엇이고, 무엇이 아닌가?

**질문:** 같은 이미지 질문의 품질을 유지하는 데 필요한 '시각적 sampling density'와 'visual token당 expert 계산량'은 교환 가능한가? 그 최적 배분이 총 FLOPs만으로 고른 결과와 다르게 실제 EP critical path를 개선하는가?

두 조절축:

```text
R: 이미지 preprocessing 해상도/visual-token budget
K_v: visual token당 실제 실행 expert 수
```

첫 실험은 **saliency별 per-token dynamic top-k가 아니다.** 각 이미지의 모든 visual token에 같은 K_v를 적용하여, R과 K_v의 효과를 우선 분리한다. 중요도 모듈·saliency network·복잡한 runtime scheduler부터 만들지 않는다.

이후 saliency가 필요하더라도 '영상이 복잡함', 'router entropy가 높음', '질문에 중요함'을 동일시하지 않는다. 어떤 정보가 예측하는 것이 좋은지 데이터로 확인해야 한다.

### 네 조건

1. MLLM-specific: 같은 원본 이미지의 관측 해상도를 바꾼다. Text-only AR의 generic k 감소만으로 설명되는 결과와 구별.
2. EP workload 감소: 실제 visual assignments/projection FLOPs/traffic를 감소시킨다. 관심은 rank-critical time이며 pair 수만으로 결론내리지 않는다.
3. Text-only와 구별: input representation budget과 expert intensity의 interaction을 실험한다. 낮은 해상도만으로 전부 설명되면 일반 visual compression.
4. EP 재설계 허용: 실제 mixed modality-k dispatch 또는 scheduling을 변경할 수 있으나 비용·quality를 포함하며, 초기에는 fixed config grid와 기존 backend를 우선한다.

**신규성 gate:** joint grid search가 잘 된다는 것만으로 독립 method/논문 GO를 주지 않는다. 강한 독립 tuning control과 EP-cost-blind control을 이겨야 한다.

## 1. 기존 연구와의 경계 — 필수 수정 사항

| 연구 | 확인된 조절 | 본 연구에 대한 의미 |
|---|---|---|
| FastMMoE | visual K_v=pK, selected weight renorm; 별도 token pruning | visual top-k 감소 자체는 이미 training-free로 존재 |
| MoDES | top-k 후보 안의 modality/layer-dependent expert skipping | effective k를 바꾸지 않는 연구라고 주장 금지 |
| MACS | capacity + overflow reroute/drop + local expansion | capacity 기반 실행 expert 집합 변경도 prior에 존재 |
| ReaLB | per-EP-rank precision 조절 | k 조절의 직접 baseline은 아니지만 EP balancing 관련 prior |

이 문서는 위 연구들의 속도·score를 다른 하드웨어/프로토콜에 그대로 적용하지 않는다. FastMMoE의 visual-k component나 MoDES 공식 코드를 포팅한다면 원래 알고리즘·renorm·calibration을 audit한다. 일부 아이디어만 구현하면 `REIMPLEMENTED_COMPONENT_CONTROL`, 원논문 전체 baseline으로 부르지 않는다.

## 2. 시작 조건과 저자원 운영

Part 1의 `frozen_baseline_bundle.json`을 읽고:

- MODEL_CONTRACT, IMAGE_PIPELINE, REFERENCE_VS_EP_QUALITY 확인.
- Official-score reproduction 상태와 protocol 미확인 사항을 그대로 가져온다.
- Verified precision/model/processor/evaluator를 고정.
- SIM_EP4_VALIDATION과 target-topology/calibration domain을 확인.

`BASELINE_INVALID`이면 Part 2 quality/속도 비교를 중단하고 Part 1을 수정한다. Public protocol 일부 미공개지만 local paired baseline 검증은 통과한 경우 `DIAGNOSTIC_ONLY`는 가능하다. Official-score preservation 주장은 불가.

4GPU가 없어도 semantic quality grid와 actual local cost sampling은 가능하다. 각 설정의 target EP4/8 시간은 validated simulator prediction이며 actual로 쓰지 않는다. 4GPU가 돌아오면 같은 shortlisted settings를 재측정한다.

환경 변경이 없으면 Part 1의 전체 benchmark 재실행은 필요 없다. No-op parity와 manifest validation으로 재사용한다.

## 3. 이해를 위한 실험 예시 — 결과가 아니다

| 설정 | Actual visual tokens(예시) | K_v | Visual assignments/layer |
|---|---:|---:|---:|
| 더 자세히 보고 가볍게 계산 | 512 | 4 | 2048 |
| 덜 자세히 보고 더 계산 | 256 | 8 | 2048 |

첫 설정이 OCR에 낫고 두 번째가 일반 장면에 나을 것이라는 것은 **가설**이다. 반대로 나와도 그대로 보고한다. Global visual pair 수가 같아도 vision encoder/attention/KV/EP padding/rank imbalance/latency는 같지 않다.

## 4. Resolution axis의 정확한 계약

`R0`는 Part 1에서 고정한 reference image-processing policy다. 'native'라는 단어를 임의 full pixel image와 혼용하지 않는다.

초기 candidate는 R0 처리 후 예상 visual-token budget에 대한:

```text
R = {1.00, 0.50, 0.25}
```

이 비율은 목표값이지 실제 토큰 수를 강제로 만들었다는 뜻이 아니다.

실행 절차:

1. 같은 original image를 사용한다. 이미지 삭제/선택 crop/외부 OCR 결과 삽입 금지.
2. aspect ratio를 유지하고 설치 processor의 legitimate max-pixel/resize 인터페이스로 목표를 지정한다.
3. patch/merge alignment를 준수하고 actual resized H×W, grid_thw, visual token 수를 저장한다.
4. R0와 candidate가 동일 grid가 되면 `DUPLICATE_CONFIG`로 표시한다. 동일 입력을 서로 다른 resolution 성공 사례로 세지 않는다.
5. min pixels·small image에서는 실제 비율이 달라도 숨기지 않는다. Oversampling으로 원래 없던 detail을 얻었다고 주장하지 않는다.
6. 원본→각 resolution을 직접 만든다. low-res→high-res 재확대로 공정성을 깨지 않는다.
7. qwen-vl-utils 및 processor 이중 resize를 방지한다.
8. 해상도가 바뀌면 visual encoder, DeepStack, MRoPE/grid mapping, KV 길이도 일관되게 다시 생성한다.

초기에는 1.00보다 높은 해상도를 추가하지 않는다. R0가 이미 낮은 cap이거나 protocol 오류라면 Part 1 이슈로 처리한다.

## 5. Expert axis — sampling top-k와 절대 혼동하지 않기

```text
K_v = {8, 6, 4}
K_text = 8
K_generated_decode = 8
```

Primary는 모든 실제 routed decoder layer에서 visual rows에 같은 K_v를 적용한다. 초기 layer를 임의 보호하거나 top-2를 기본으로 넣지 않는다. Uniform grid에서 문제가 보이면 원인을 기록하고, 'first few layers exact' 같은 변경은 별도 registered sensitivity로만 시험한다.

### Router semantics

원래 router logits/softmax는 현재 H에서 그대로 계산한다. Visual row에 대해서만 highest K_v selected experts를 실행하고 선택된 current probabilities로 재정규화한다.

```text
w'_e = p_e / sum_{j in selected K_v} p_j
Y_i = sum_e w'_e Expert_e(H_i)
```

Text, special boundary token, generated decode row는 baseline top8. dtype, activation, weights는 고정한다. Scaling factor 등 실제 runtime 계약이 추가되면 Part 1의 verified behavior를 따른다.

- Config의 `num_experts_per_tok`를 전역 변경하여 text까지 k4로 만들지 않는다.
- KV cache 안의 vision positions가 decode 때 매번 expert에 재입력되는 것으로 세지 않는다.
- zero weights로 만든 expert branch가 실제 dispatch/GEMM에서 실행된다면 logical work만 줄어든 것. 실제 physical work 감소 아님.
- Same-state one-layer 테스트에서는 baseline top8을 rank-order로 prefix truncation할 수 있다. Full rollout의 후속 layer H/routes는 다시 계산해야 한다.
- route tie/order, padding, masked invalid ID의 처리 테스트를 한다. Sentinel expert를 실제로 실행하면 bug.

### 물리적 mixed-k 지원

기존 kernel/API가 ragged per-token k를 지원하지 않으면:

A. semantic reference에서 정확한 k별 output을 얻는다.
B. physical adapter는 visual K_v와 text8을 compact dispatch하는 최소 opt-in 경로를 만든다.
C. 분리 kernel/추가 launches/packing/collective가 발생하면 모두 charge한다.
D. k6를 k8로 pad하는 경우 실제 zero-work skipping인지 측정한다.

지원이 없다는 이유로 강제 비슷한 API를 호출하거나, weak Python-loop baseline을 만들어 큰 speedup을 주장하지 않는다. `PHYSICAL_VARIABLE_K_BLOCKED`와 semantic signal을 분리한다.

## 6. Phase B0 — No-op와 baseline integrity

R=1.00, K_v=8이 Part 1 reference와 같은 processor signature·scores·outputs를 재현하는지 확인한다. 최소 16개 QA smoke + teacher-forced numerics.

- context/token truncation, output-length cap, modality mask 누락 검사.
- Visual/text route counts, source ownership, groups를 기록.
- `48 * (K_v*N_v + 8*N_text)`는 단순 sanity check이고 실제 chunk/padding/불균등 layer accounting은 trace로 센다.
- no-op mismatch를 method-induced accuracy loss로 부르지 않는다.

## 7. 데이터와 splits

새 image-disjoint split을 고정한다. 기존 Spatial-Basis/tuning에 노출된 이미지는 가능하면 heldout에서 제외하며 불가능하면 노출 상태를 기록한다.

기본 규모:

- discovery 96개: OCR/document 32, chart/numeric 32, general/spatial scene32.
- heldout 192개: 유형별 64.
- 실질적 positive signal이 있을 때만 enlarged heldout/full benchmarks.

데이터셋 예시: TextVQA 또는 DocVQA-public-label split; ChartQA; MMStar/RealWorldQA의 실제 scene subset. 정확한 source/revision/metric은 Part 1 registry에서 가져온다. 단일 이미지라는 scope를 표에 명시하며 filtered subset을 원본 full benchmark headline과 비교하지 않는다.

같은 이미지의 여러 질문은 split을 공유한다. 이미지 파일명, dataset ID, 정답을 online selector 입력으로 사용하지 않는다. Task label은 분석용 stratification/오라클에만 사용한다.

**Semantic saliency에 대한 계획:** 첫 3×3 실험에서는 토큰별 중요도를 나누지 않는다. 먼저 '어떤 종류의 요청에서 R과 K의 상대적 가치가 바뀌는가'를 알아낸다. 이 결과 없이 saliency threshold를 꾸미지 않는다.

## 8. Phase B1 — 3×3 full semantic factorial grid

각 discovery 이미지–질문에 9개 (R,K_v) 조합을 모두 실행한다. 전 조합은 같은 공식 evaluation lane을 사용한다. 시간 제한으로 일부 greedy screening을 사용하면 별도 lane으로 표기하고 shortlisted policies는 Lane O로 확인한다.

각 cell의 실제 값:

```text
actual V, text count, image resolution, task score/parsed answer
natural generated length, EOS/finish reason, malformed answer
visual expert pairs, all pairs, physical padded pairs
per-rank/expert counts, active experts, max/mean, remote bytes
vision encode, prefill attention, routed stage, decode, TTFT, E2E
simulated vs measured labels and uncertainty
```

Baseline-correct item만 남기지 않는다. 원래 틀린 문제도 포함하여 both direction regression/improvement를 기록한다.

현재 실험은 정답률을 model full generation으로 얻는다. Baseline hidden trace만 수정해서 전체 accuracy를 예측하는 것은 금지한다. 해상도별 baseline trace와 policy별 recomputed trace를 구별한다.

## 9. Phase B2 — 첫 discovery에서 반드시 답할 질문

### 9.1 '더 보기 vs 더 계산하기'

각 task stratum에서 다음을 비교한다.

- R 감소, K8 유지: 시각 관측량 감소의 손실.
- R0 유지, K 감소: expert intensity 감소의 손실.
- joint 감소: 두 손실/절감의 상호작용.
- 가능한 실제 pair-count가 ±5% 이내인 configuration 쌍: 더 많은 V/낮은 K 대 더 적은 V/높은 K.

Matched pair가 없으면 match 없음으로 둔다. 이론적 V 목표치가 같다는 이유로 actual workload가 동일하다고 쓰지 않는다. Pixel-area grid가 반올림되어 같으면 그 cell을 새 실험으로 세지 않는다.

### 9.2 Interaction

고정 dataset score S에서 진단 interaction을 계산할 수 있다.

```text
I(R,K) = S(R,K) - S(R,8) - S(R0,K) + S(R0,8)
```

동일 image/request를 bootstrap하여 CI를 구한다. I>0만 강제하지 않는다. 어느 축에 더 민감한지, joint pair가 독립 축의 단순 예상보다 나은지/나쁜지 분석한다. 단일 global score가 OCR 손실을 숨기지 않게 한다.

### 9.3 실제 EP 비용을 고려하면 선택이 달라지는가?

아래 두 cost model로 같은 quality-constrained feasible set의 config를 선택한다.

- C_pair/FLOPs: 총 visual/all expert pairs 및 encoder/attention 계산량을 사용하되 rank imbalance를 무시.
- C_EP: Part 1 validated target-topology cost, 실제 rank histograms·dispatch startup·padding·critical path 포함.

중요: 실제 cost comparison에서는 각 candidate의 policy-specific trace를 사용한다. 해상도/k 변경 전의 histogram을 그대로 쓰지 않는다.

비교 output:

```text
PREFERRED_CONFIG_DISAGREEMENT_RATE
ACTUAL_EP4_COST_REGRET_OF_PAIR_ONLY_SELECTION
HELDOUT_QUALITY_DELTA
BENEFIT_VS_BEST_SINGLE_AXIS / VS_INDEPENDENT_TWO_AXIS
```

두 모델이 같은 선택을 하고 속도/quality 이점도 같으면 EP-specificity 미확인. 단순히 MoE를 EP4로 실행한 것은 EP contribution의 증거가 아니다.

## 10. 필수 controls — 약한 baseline 금지

C0: verified R0/K8.

C1: resolution-only, K8; 같은 discovery budget으로 가장 좋은 setting 선택.

C2: expert-only, R0; uniform K8/6/4. 이것을 novel method라고 부르지 않는다.

C3: independent tuning 두 방향. R을 K8에서 선택 후 K를 선택하는 순서, K를 R0에서 선택 후 R을 선택하는 순서를 모두 본다. C3도 두 축을 쓰므로 C1/C2보다 강한 control.

C4: joint grid지만 total-pairs/FLOPs 기반 cost selection. C5와 같은 semantic score 정보·같은 candidate set·같은 tuning budget.

C5: joint grid + EP cost. 이 연구가 EP-aware라면 실제 heldout EP4에서 C4보다 이점이 필요하다.

C6: 가장 가까운 prior. 최소 FastMMoE visual-expert reduction component와 MoDES 접근을 감사하고, 공식 코드가 호환되면 declared budget에서 비교한다. Port/scaffold만 있으면 `NOT_RUN/INCOMPLETE_PORT`; baseline table에 fabricated score를 넣지 않는다. MoDES full reproduction이 어려워도 component controls C1–C5는 먼저 완료 가능하나 SOTA superiority claim은 보류한다.

Text-only diagnostic: 원래 text k8 유지 조건에서는 R축이 존재하지 않아야 한다. Generic K-all-tokens reduction을 따로 시험해 이미지 고유 trade-off와 혼동하지 않는다. Text control이 없으면 'AR에서는 불가능'이라는 과한 주장을 하지 않는다.

## 11. 중요도/online policy는 구조가 확인된 다음 단계

### 11.1 Oracle vs 실제 정책

각 heldout 문제의 정답을 보고 best config를 고르면 `PER_EXAMPLE_ORACLE`다. Task name으로 OCR/scene을 고르는 oracle도 deployment policy가 아니다. Oracle 결과는 headroom까지만.

실제 online resolution은 expensive high-res forward **전에** 선택해야 한다. Candidate별 full inference를 모두 돌리고 가장 좋은 것을 고르면 9-run cost를 모두 지불한 방법이다.

### 11.2 처음 허용할 cheap request-level features

실제 image dimensions/aspect ratio, input token length, image count=1, declared batch composition, query의 deterministic text-intent cue, pixel-only low-cost edge/detail statistic 등을 고려할 수 있다. 이들은 **importance proxy 가설**이며 ground-truth saliency가 아니다. 추가 OCR network/LLM classifier를 새로 불러 공짜라고 쓰지 않는다.

Model training 없이 discovery split에서 소수 threshold/rule/table을 고를 수 있으나 calibration data·tuning count·online cost를 기록한다. Dynamic rule은 fixed best joint policy보다 낫다는 결과가 있을 때만 채택한다. Unseen/불확실 request는 R0/K8 fallback.

### 11.3 토큰별 saliency는 이번 primary 범위 밖

Causal decoder에서 이미지가 질문보다 먼저 배치되면 visual hidden이 뒤쪽 질문에 직접 conditioned되어 있다고 가정하면 안 된다. 마지막 question token의 attention을 모두 얻은 뒤, 이미 실행한 앞쪽 vision-MoE를 같은 pass에서 되돌려 줄이는 것은 불가능하다. Probe/recompute가 필요하면 비용을 모두 세고 별도 방법으로 정의한다.

따라서 'attention saliency를 써서 바로 k를 바꾸자'는 shortcut을 이번 구현에 섞지 않는다. 결과가 좋아도 이번 단계의 결론은 request-level resolution/expert trade-off에 한정한다.

## 12. 실제 EP4 검증과 Virtual EP 사용

### 4GPU가 있을 때

3×3 discovery grid의 대표 real cases는 실제 EP4에서 측정한다. 최소 (R0,8), (R0,4), (R0/2,8), (R0/2,4), (R0/4,8) 및 shortlisted joint candidates를 low/mid/high token bin에서 재생한다. 가능하면 모든 9개를 bounded batch1 timing으로 측정한다.

Backend는 Part 1의 verified sparse baseline과 같다. custom mixed-k adapter가 필요하면 no-op k8 custom-baseline을 포함하여 adapter 자체의 비용/손해를 분리한다.

### 1/2GPU만 있을 때

- full semantic generation으로 quality와 새 routes 확보.
- virtual EP4/8에서 topology별 workload와 calibrated latency prediction.
- 지금 대상 shape가 calibration domain 밖이면 추가 kernel sample 수집 또는 prediction 불확실 상태로 유지.
- actual EP4 gain field는 N/A로 둔다. CPU-staging/serial-rank physical wall을 E2E gain으로 쓰지 않는다.
- 새 config의 predicted gain이 uncertainty band보다 작으면 순위를 확정하지 않는다.
- GPU 복귀 후 frozen shortlisted policy만 actual EP4 측정하여 시뮬레이터를 재검증.

## 13. Latency와 work accounting

총 latency는 다음을 포함한다.

```text
resolution choice / resize / H2D
vision encoder and merger/DeepStack
language prefill attention and other dense operations
router + actual mixed-k layout/dispatch/experts/combine
full autoregressive generation to EOS
```

- N_v 감소는 attention/KV와 decode 비용도 바꿀 수 있다. expert k 감소만으로 KV row count가 감소하지는 않는다.
- 동일 task의 output이 짧아져 빨라진 것인지 fixed teacher-forced 32/128 control을 함께 본다.
- encoder caching은 microbench에는 가능하나 fresh-image E2E에서 빼지 않는다.
- Natural quality generation과 fixed-length latency run의 비용·목적을 구분.
- token/assignment fraction을 E2E fraction으로 환산하지 않는다.
- EP4와 EP8는 같은 global batch/workload, 명시한 topology에서 비교. DP request 수를 몰래 늘려 좋은 throughput을 만들지 않는다.

Headline:

```text
LATENCY_REDUCTION = (T_P0-T_method)/T_P0
SPEEDUP = T_P0/T_method
```

% reduction과 ×speedup을 구별한다. Median of per-request gains와 ratio of aggregate means를 별도로 보고하고 혼용하지 않는다.

## 14. Quality validation

Discovery 후 policy를 최대 2개로 freeze한다. Heldout192에 한 번 평가한다. Heldout 결과를 보고 같은 split에서 threshold를 반복 튜닝하지 않는다.

각 task 공식 score + macro score(정의된 scaling/weight), paired changes, confidence interval, baseline-correct→wrong와 reverse counts, resolution-sensitive 사례, EOS/출력길이를 저장한다.

Quality 목표는 원래 성능 유지다. 운영 목표로 -1 normalized pp non-inferiority margin을 preregister할 수 있지만, 작은 표본으로 이를 입증하지 못하면 `QUALITY_UNRESOLVED`. 원래 source가 VQA soft score이면 binary correct로 억지 변환하지 않는다. Explicit OCR/숫자 case의 손실을 다른 task 향상으로 숨기지 않는다.

공식 full split reproduction 결과와 연구용 filtered heldout score를 같은 열로 합치지 않는다.

## 15. Discovery 판정 — 작은 gate 하나가 연구 전체를 죽이지 않도록

Part 1 인프라는 어떤 결론에서도 유지한다.

### 구조적 의미

- `TRADEOFF_SUPPORTED`: 같은 EP budget에서 두 축의 선택이 quality frontier를 개선하거나, task마다 해상도와 K의 민감도 차이가 재현됨.
- `TRADEOFF_WEAK`: uniform expert reduction 또는 resolution-only가 joint frontier를 거의 설명.
- `EP_SPECIFIC_SUPPORTED`: C5가 공정한 C4/C3보다 actual EP critical path 또는 E2E에서 추가 이점을 보임.
- `EP_SPECIFIC_UNRESOLVED`: simulator만 있거나 uncertainty가 차이보다 큼.
- `GENERIC_COMPRESSION_ONLY`: 좋아졌지만 rank-aware 선택의 추가 이점 없음.

### 운영 목표 (측정 전 고정)

- Primary aspirational target: quality preserved + actual E2E latency reduction ≥15% vs R0/K8.
- ≥5% 실제 개선이지만 15% 미만이면 useful signal/HOLD; 자동으로 main-method GO 아님.
- Closest independent/joint cost-blind control보다 추가 E2E 절감 ≥3pp 또는 same-latency quality +2 normalized points를 discovery materiality 예시로 사용하되 CI·task별 손실을 함께 확인한다. Tiny sample의 이 숫자 충족만으로 확정하지 않는다.
- EP-aware 추가 이점이 없으면 사용자가 요구한 EP 연구 메인 방향으로 GO 금지.
- Simulator-only positive: `HOLD_PENDING_PHYSICAL_EP4`.
- 공식 baseline mismatch: `BLOCKED_BASELINE`, method NO-GO가 아님.
- 실제 variable-k adapter 실패: `BLOCKED_RUNTIME`, semantic hypothesis와 구별.
- Limited grid가 실패하면 **tested resolution/expert family** NO-GO. 모든 MLLM MoE EP의 불가능성으로 확대 금지.

## 16. 실행 순서 / 과구현 방지

```text
load frozen baseline bundle
→ no-op parity
→ 96-image 3×3 semantic discovery
→ policy-specific traces + actual EP4 or virtual cost
→ resolution-only / expert-only / independent / cost-blind controls
→ interaction & matched-budget analysis
→ freeze ≤2 settings/policies
→ heldout192 quality
→ actual EP4 paired TTFT/natural E2E/fixed-decode
→ optional cheap request rule only if static interaction exists
→ simulator cross-check and final scoped verdict
```

GPU 부족이면 appropriate 단계에서 checkpoint하고 semantic/replay로 전환한다. Unbounded CUDA optimization·serving scheduler rewrite·new saliency model training·비디오 확장은 하지 않는다.

## 17. 필수 보고서

```text
reports/baseline_handoff_check.md
reports/prior_topk_audit.md
reports/resolution_and_modality_k_contract.md
reports/factorial_grid_quality_work_cost.md
reports/task_resolution_vs_expert_sensitivity.md
reports/matched_pair_and_latency_frontiers.md
reports/independent_vs_joint_controls.md
reports/ep_specific_increment.md
reports/heldout_quality.md
reports/physical_ep4_or_virtual_projection.md
reports/final_summary.md
artifacts/results.jsonl / frozen_policy.json / summary.json
```

필수 그림: task별 quality heatmap, actual V×K와 EP critical time, accuracy–TTFT 및 accuracy–E2E frontier, C3/C4/C5 comparison, 실제와 simulator prediction. PDF/figure 숫자는 raw JSON에서 일관되게 생성한다.

## 18. 최종 표 형식

| R | actual V | K_v | score(task별) | visual pairs | physical rows | max/mean | TTFT ms | E2E ms | evidence |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| R0 | measured | 8 | measured | measured | measured | measured | measured | measured | ACTUAL/… |

`OBSERVED_REAL`과 `SIMULATED_EP4_CALIBRATED` 결과를 같은 색/열에서 구별 없이 제시하지 않는다.

## 19. Final Required Summary

```text
FOUNDATION_BUNDLE_AND_BASELINE_STATUS:
MODEL_REVISION / GPU_SET / ACTUAL_TOPOLOGY:
PUBLIC_REPRODUCTION_STATUS:
EVALUATION_LANE_AND_METRIC:
SAMPLING_TOP_K / EXPERT_TOP_K_TEXT / EXPERT_TOP_K_VISION:
DISCOVERY_COUNT / HELDOUT_COUNT:
RESOLUTION_GRID_AND_ACTUAL_VISUAL_TOKEN_COUNTS:
PRIMARY_K_GRID:
NO_OP_PARITY:
BEST_RESOLUTION_ONLY:
BEST_EXPERT_ONLY:
BEST_INDEPENDENT_TWO_AXIS:
BEST_JOINT_PAIR_ONLY:
BEST_JOINT_EP_AWARE:
TASK_SENSITIVITY_AND_INTERACTION:
MATCHED_PAIR_COMPARISON:
ACTUAL_EXECUTED_VISION_PAIRS_REDUCTION:
PHYSICAL_PADDED_WORK_REDUCTION:
EP_CRITICAL_PATH_CHANGE:
HELDOUT_QUALITY_DELTA_AND_CI:
TRUE_EP4_TTFT_GAIN / NATURAL_E2E_GAIN / FIXED_DECODE_GAIN:
SIM_EP4_GAIN_AND_UNCERTAINTY / SIM_EP8_LABEL:
EP_SPECIFIC_INCREMENT_VS_STRONGEST_CONTROL:
ONLINE_SELECTOR_CAUSALITY_AND_COST:
PRIOR_BASELINES_COMPLETED_OR_MISSING:
TRADEOFF_STATUS / EP_SPECIFIC_STATUS / QUALITY_STATUS:
VERDICT: GO_BOUNDED / HOLD / GENERIC_ONLY / NO_GO_TESTED_FAMILY / BLOCKED
NOT_TESTED_AND_LIMITATIONS:
NEXT_ACTION:
DO_NOT_IMPLEMENT_UNBOUNDED_PRODUCTION_RUNTIME: true
```

## 20. 근거

- Qwen checkpoint config: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/raw/main/config.json
- Qwen official preprocessing/evaluation: https://github.com/QwenLM/Qwen3-VL
- Qwen technical report Table 3 / Appendix B: https://arxiv.org/abs/2511.21631
- FastMMoE §3.1–3.2: https://arxiv.org/html/2511.17885v2
- MoDES §5: https://arxiv.org/html/2511.15690v2
- MACS §3 / Appendix A.5: https://arxiv.org/html/2605.05225
- ReaLB: https://arxiv.org/html/2604.19503
- vLLM EP topology/backend: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

문서의 grid, sample sizes, novelty gate, simulator 사용 규칙은 이번 연구를 위한 제안이다. OCR에 높은 V/낮은 K가 우월하다는 결과는 아직 없다.
