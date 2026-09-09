# MLLM MoE Transient Branch Compression PoC Spec

이번 작업은 이전의 pairwise critical-rank coalescing을 반복하는 것이 아니다.

새 연구 질문은:

> **Visual tokens need not be redundant as attention entities to be redundant as expert computations.**

즉 Qwen3-VL의 visual token들은 Attention에서는 각각 독립적인 token/position으로 그대로 유지하되,
MoE에 들어가는 순간의 일부 `(token, expert)` branch만 transient하게 압축/공유할 수 있는지 검증한다.

Main idea:

```text
Attention:
모든 vision token 그대로 유지

Router:
원래 top-k 그대로 유지

MoE:
같은 expert로 들어가는 spatially/local하게 유사한 visual branch들을
대표 branch 하나로 압축해서 계산

Reconstruction:
대표 expert update를 원 token branch에 다시 펼침
원래 residual h_i와 router weight g_ie는 유지

Next Attention:
원래 token 수 그대로
```

핵심은 "token merging"이 아니라 **expert computation sharing**이다.

이번 PoC는 약 8–12시간의 deep discovery sprint로 설계한다.
Main hypothesis가 NO-GO여도 captured expert outputs를 이용해
sub-expert-stage compressibility, contribution-weighted sharing,
layer subset, packing/run structure 등의 인접 successor를 스스로 탐색한다.

## 0. WORKING SPEC

다음 spec을 repository에 저장하고 working contract로 사용하라.

```text
poc_flashvep/reports/
mllm_moe_transient_branch_compression_poc_spec.md
```

Branch:

```text
flashvep/mllm-moe-transient-branch-compression-poc
```

Main report:

```text
poc_flashvep/reports/
mllm_moe_transient_branch_compression_poc.md
```

Spec의 gate와 evidence boundary를 임의로 완화하지 마라.

## 1. GPU SAFETY

모든 task-owned GPU 실행은 오직:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

만 사용한다.

GPU 0–3은 절대 사용하지 마라.

8GPU 실험으로 확장하지 마라.
사용자가 새로운 physical GPU set을 별도로 허가하기 전에는 4장만 사용한다.

각 launch 전:

- physical UUID
- logical mapping
- process owner
- free memory

를 확인한다.

다른 사용자의 process를 종료하지 마라.
burn을 실행하거나 재시작하지 마라.

## 2. PREVIOUS NEGATIVE EVIDENCE

이전 negative를 숨기지 마라.

특히:

- critical-rank same-expert coalescing의 파이가 작았음;
- spatial/routing chunk 계열 여러 개가 NO-GO;
- CP→EP fine-grained splitting은 invocation fragmentation tax로 실패;
- hierarchical rebatching은 feasible direct-BCT median ~1.35%, MoE increment ~0.50%로 NO_HIERARCHICAL_HEADROOM.

따라서 이번에는:

```text
critical rank만 줄이기
pair만 찾기
hidden cosine만 보기
작은 MoE invocation 여러 개 만들기
```

를 반복하지 않는다.

## 3. PRIMARY MODEL

Primary:

```text
Qwen3-VL-30B-A3B-Instruct
BF16
4xH100
```

기존 검증된 faithful topology를 다시 확인한 뒤 사용한다.

가능한 예:

```text
TP2 / DP2 / EP4
DeepEP HT
TritonExperts
DBO off
```

TP-only path를 EP라고 부르지 마라.

Kimi는 Qwen strong/promotion gate 이후에만.

## 4. PRIOR ART FIRST

최소 다음을 감사한다.

```text
FastMMoE
MoDES
AnyExperts
MoECa
FastV / SparseVLM
visual token merging / anchor recycling
SERE / expert substitution
SpecMoE if relevant
current vLLM/SGLang/DeepEP
```

특히:

- FastMMoE: routing-aware visual token pruning + expert activation reduction
- MoDES/AnyExperts: token/modality-aware expert reduction
- MoECa: DiT-MoE에서 cross-timestep expert-branch reuse

를 exact collision 관점으로 본다.

우리가 노리는 gap은 단순 "expert-aware merging"이 아니다.

후보 gap:

> same-forward MLLM에서 token identity/attention sequence는 유지하면서
> spatial visual token의 selected expert branch output만 공유하는 것.

Novelty는 측정과 audit 이후에만 판단한다.

## 5. CAPTURE REAL EXPERT BRANCH OUTPUTS

대표 MoE layers에서 actual:

```text
request_id
layer
token
modality
image id
2D patch coordinate
hidden h_i
expert e
router weight g_ie
EP rank
expert output E_e(h_i)
weighted output g_ie E_e(h_i)
combined MoE update
```

를 capture한다.

early/middle/late, 여러 dataset, 여러 resolution을 포함한다.

observer-heavy run과 clean timing을 분리한다.

## 6. MAIN UNTESTED PREMISE — OUTPUT-LEVEL COMPRESSIBILITY

같은 expert e로 들어가는 token branch pair/group에 대해:

```text
input distance:
||h_i - h_j||

output distance:
||E_e(h_i) - E_e(h_j)||
```

를 측정한다.

Vision/Text, layer, expert, spatial distance, route overlap,
router-weight bucket별로 분석한다.

특히:

> distinct inputs가 같은 expert를 통과하면 output은 더 비슷해지는
> "expert-induced contraction" 영역이 있는가?

를 본다.

hidden cosine만으로 결론 내리지 마라.

## 7. GROUPING ORACLES

각 expert별로:

```text
same-expert arbitrary
1D contiguous
true 2D neighbor
2x2 / 2x4 spatial group
route-overlap local group
hidden-sim local group
true output-oracle group
```

을 비교한다.

Output-oracle grouping은 production method가 아니라
compressibility upper bound와 proxy discovery용이다.

Group size:

```text
2 / 3 / 4 / 6 / 8
```

## 8. APPROXIMATION POLICIES

한 group에서 대표 expert evaluation을 1회만 수행한다고 가정하고:

```text
Anchor
Medoid
Centroid
router-weighted centroid
spatial deterministic anchor
output-oracle medoid
```

를 비교한다.

중요:

원 token의 residual `h_i`, position, router weight, token count는 유지한다.

Main semantics:

```text
yhat_i = h_i + sum_e g_ie * approx_branch(i,e)
```

이다.

Token deletion/pruning이 아니다.

## 9. PRIMARY COMPRESSIBILITY GATE

반드시 다음을 계산한다.

```text
vision routed branch assignments
safe compressible assignments
representative branch count
branch row reduction
dispatch row reduction
combine row reduction
per-rank reduction
group size distribution
```

Output-oracle에서도 strict quality-local tolerance에서:

```text
safe vision branch reduction <15%
```

이면 full-branch sharing은 강한 negative다.

```text
>=25%
```

면 promising,

```text
>=35%
```

면 strong geometric signal이다.

한 layer만 잘 나오는 결과로 승격하지 마라.

## 10. QUALITY PROPAGATION

다음 순서로 error를 추적한다.

```text
expert branch output
→ combined MoE output
→ post-residual hidden
→ next-layer router
→ multi-layer rollout
→ final logits
→ benchmark answer
```

측정:

```text
cosine
relative L2
max abs
next-layer top-k agreement
router KL
first-token exact
short greedy exact
benchmark accuracy
```

Suggested screen:

```text
combined MoE rel-L2 <=1%
next-layer route agreement >=99%
benchmark loss <=0.5pp
```

OCR/math/visual-reasoning subset은 별도 보고한다.

## 11. ECONOMIC ORACLE BEFORE PROTOTYPE

Assignment reduction을 speedup으로 착각하지 마라.

실제 measured request trace로:

```text
O1 compute-only perfect oracle
O2 communication-aware oracle
O3 feasible oracle
```

을 만든다.

O3는 최소한:

```text
compression-map
representative creation
gather
scatter
reconstruction
metadata
alignment/padding
```

비용을 포함한다.

Report:

```text
MoE-stage
prefill
TTFT
request E2E
```

Strong promotion:

```text
MoE stage >=15–20%
direct feasible request E2E >=10–12%
```

Feasible E2E <8%면 main method 구현 금지.

## 12. RLE-LIKE ROUTE RUN

사용자가 원하는 직관적 RLE-inspired policy를 반드시 별도 test한다.

한 expert/rank에서 visual branch를 original spatial/token order로 정렬하고:

```text
same image
same expert
spatially adjacent
cheap similarity/safety gate pass
```

이면 run으로 묶는다.

Compare:

```text
1D contiguous RLE
true 2D adjacency
2D connected components
fixed local windows
```

Measure:

```text
run-length histogram
fraction length>=2
fraction length>=4
branch reduction
output-oracle recovery
map-build overhead
```

1D RLE가 2D oracle의 대부분을 회수하면 strong implementation advantage다.

## 13. ONE LARGE INVOCATION ONLY

이전 CP→EP 실패를 반복하지 마라.

Prototype은 반드시:

```text
one compact representative branch list
→ one DeepEP dispatch
→ one expert execution
→ one combine
→ local expansion
```

을 지향한다.

Run마다 dispatch/GEMM을 별도로 호출하지 마라.

## 14. TRIVIAL FIX ATTACK

후보마다:

```text
group size 2 fixed
every-2 spatial share
lowest-router-weight only
early-layer only
middle-layer only
route-overlap threshold only
hidden cosine threshold only
```

를 비교한다.

간단한 static policy가 best oracle gain의 80% 이상을 회수하면
복잡한 adaptive method를 만들지 마라.

Simple method 자체가 strong E2E+quality면 simple paper candidate는 가능하다.

## 15. REQUIRED BASELINES

최소:

```text
Vanilla
Random sharing
Spatial-only
Hidden-sim-only
Routing-sim-only
Whole-token visual merging/pruning
FastMMoE-style if reproducible
MoDES-style if locally reusable
```

를 비교한다.

Main question:

> 같은 quality loss에서 transient branch compression이
> whole-token reduction/expert skipping과 다른 좋은 Pareto point를 만드는가?

## 16. IF MAIN HYPOTHESIS NO-GO — DO NOT IMMEDIATELY STOP

Captured data를 이용해 nearby successor를 자율 탐색한다.

Priority order:

### S1. Sub-expert-stage compressibility

Qwen expert 내부:

```text
gate/up
→ gated intermediate
→ down
```

에서 full output은 공유 불가해도
down-projection input/output만 더 compressible한지 본다.

실제 saved FLOPs/E2E oracle을 계산한다.

### S2. Contribution-weighted branch sharing

`g_ie * E_e(h_i)` 기준으로 보면 low-router-weight branches는
더 쉽게 근사 가능한지 본다.

단 결과가 사실상 expert skipping이면 그렇게 정직하게 분류하고
MoDES/AnyExperts prior art를 공격한다.

### S3. Eligible-layer subset

전체 layer는 안 되어도
특정 8–16개 layer에서만 매우 안전하고 큰 압축이 가능한지 본다.

### S4. Spatial route-run packing without approximation

Same-expert long run이 있지만 outputs는 다르면,
run 구조 자체가 alignment/indexing/packing overhead를 줄이는지 본다.

그 overhead의 direct E2E mass가 <8%면 닫는다.

### S5. Expert-induced contraction/codebook oracle

특정 experts가 broad visual inputs를 narrow output manifold로 map하면
expert-specific low-rank/codebook approximation의 perfect oracle만 계산한다.

Headroom >=10%일 때만 prototype idea로 승격한다.

### S6. Cheap error repair

Anchor reuse가 quality gate 근처라면:

```text
scalar norm correction
group mean residual correction
router-weight-aware scale
```

까지만 본다.

learned auxiliary model은 만들지 마라.

### S7. Opposite result

Experts가 tiny visual differences를 amplify한다면
"why pre-MoE merging is unsafe"라는 system fact로 기록하고,
post-gate/intermediate/down-only child hypothesis가 생기는지 본다.

## 17. AUTONOMOUS TREE

Main candidate가 2–3시간 안에 실패해도
전체 output-compression domain이 죽었다고 단정하지 마라.

최소 20 causal hypotheses를 목표로 한다.

Forced rethink:

```text
#5
#10
#15
#20
```

각 experiment 후:

```text
EXPECTED
OBSERVED
FAILED ASSUMPTION
NEW SYSTEM FACT
DIRECT E2E IMPLICATION
NEXT CHILDREN 0–2
```

저장.

단 다음이 모두 확인되면 조기 종료 가능:

```text
full-output oracle small
sub-expert oracle small
contribution-weighted oracle small
layer subset small
packing overhead small
direct E2E <8%
no novel adjacent phenomenon
```

GPU를 10시간 채우기 위해 의미 없는 실험을 하지 마라.

## 18. SUGGESTED TIME BUDGET

```text
0–1h
source/prior-art/capture validation

1–3h
output compressibility atlas
expert contraction
spatial/run structure

3–4.5h
anchor/medoid/centroid
one-layer/next-layer quality

4.5–6h
multi-layer rollout
benchmark quality
economic oracle

6–7h
RLE proxy / cheap deployable policy
or, if main fails, sub-expert successor search

7–8.5h
best child deep dive
trivial-fix/prior-art attack
request-level oracle

8.5–10+h
only if justified:
minimal compact-EP prototype
clean E2E validation
```

## 19. STRONG GO

FOUND_STRONG_GO requires:

```text
real Qwen VL branch data
output-level causal redundancy
safe branch reduction >=25%, preferably >=35%
deployable proxy recovers substantial oracle
MoE-stage >=15–20%
direct feasible E2E >=10–12%
benchmark loss <=0.5pp
attention token count unchanged
one-large-invocation path
no direct prior-art collision
minimal prototype improves clean E2E
```

Kimi is gated until then.

## 20. FINAL STATUS

Use exactly one:

```text
FOUND_STRONG_GO
FOUND_PROMISING_OUTPUT_ORACLE
FOUND_SUBEXPERT_SUCCESSOR
FOUND_INCREMENTAL_ONLY
NO_OUTPUT_COMPRESSIBILITY
NO_ECONOMIC_HEADROOM
PRIOR_ART_COLLISION
ENVIRONMENT_BLOCKED
```

A NO-GO must state whether it failed at:

```text
geometry
quality
economic mapping
implementation
novelty
```

## 21. FINAL RESPONSE

Spec의 final report format을 따른다.

반드시 다음을 답한다.

1. Vision branch output이 Text보다 실제로 더 compressible한가?
2. Hidden cosine보다 expert-output similarity가 추가 정보를 주는가?
3. Spatial/RLE-like grouping이 output oracle을 얼마나 회수하는가?
4. Whole-token pruning 없이 branch rows를 몇 % 줄일 수 있는가?
5. 다음-layer router와 final benchmark quality는 유지되는가?
6. Assignment reduction이 direct request E2E 몇 %로 번역되는가?
7. FastMMoE/MoDES/MoECa와 exact 차이는 무엇인가?
8. Full-branch hypothesis가 실패했다면 sub-expert-stage child는 살아남는가?
9. Main NO-GO 이후에도 별도의 >=10% direct oracle 후보가 발견됐는가?
10. 실제로 다음 연구를 이어갈 이유가 있는가?

완료 후 commit/push하고:

```text
Branch
Commit
Report
Result root
```

를 명시한다.

## 22. IDLE GPU UTILIZATION OVERRIDE

메시지의 마지막 지시에 따라, task experiment가 GPU 4–7을 사용하지 않을
때와 최종 응답 이후에는 `/home/esjung/vllm-ep/run_utilize.sh`를 물리 GPU
4,5,6,7에만 실행한다. Measurement 전에는 PID/command/owner를 확인한 뒤
task-owned burn만 종료하고, 다른 사용자의 process는 종료하지 않는다.
