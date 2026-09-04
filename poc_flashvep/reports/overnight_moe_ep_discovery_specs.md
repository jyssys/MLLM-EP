# Overnight MoE-EP Phenomenon Discovery Sprint

## Objective

이번 작업의 목표는 새로운 method를 구현하는 것이 아니다.

목표는 4×H100 환경에서 LLM/MLLM MoE inference를 systematic하게 probe하여,

&gt; "현재 literature에서 명확히 다뤄지지 않았거나,

&gt; 새로운 systems research question으로 발전할 수 있는

&gt; 강한 empirical phenomenon"

을 발견하는 것이다.

Positive result를 억지로 만들지 않는다.

각 hypothesis는 falsification-first 방식으로 검사한다.

중요:

- method implementation 금지

- RL 금지

- dynamic TP↔EP implementation 금지

- 새로운 token pruning/rerouting implementation 금지

- 긴 최적화 작업 금지

먼저 phenomenon / oracle headroom / causal relationship만 확인한다.

---

# Environment

GPU:

CUDA_VISIBLE_DEVICES=1,2,3,4

4× H100.

Primary MLLM:

- Qwen3-VL-30B-A3B-Instruct

Optional generic LLM validation:

- Qwen3-30B-A3B

- strong phenomenon에 대해서만 사용

Primary EP configuration:

- TP2 / DP2 / EP4

- DeepEP high-throughput

- 실제 DeepEP runtime activation을 source/log로 verify

필요한 controlled microbenchmark에서는

TP4/DP1 또는 local expert replay 사용 가능.

Precision:

- BF16 primary

기존 validated instrumentation / trace / profiling code를 최대한 재사용한다.

---

# General Measurement Rules

모든 실험에서 가능한 경우 다음을 기록:

- total tokens

- routed assignments

- active experts

- tokens / active expert

- per-expert token histogram

- per-rank token histogram

- rank max/mean

- expert-count CV / HHI

- routing entropy

- distinct destination ranks / token

- Dispatch CUDA time

- Expert CUDA time

- Combine CUDA time

- Full T_MoE

- TTFT / TPOT when relevant

T_MoE:

dispatch-start

→ combine-complete

layer-wise rank-critical CUDA span을 primary로 사용.

단순 Python wall-clock을 primary metric으로 사용하지 않는다.

모든 comparison은:

- same input volume

- warmup

- &gt;=3 measured repetitions

Signal gate:

STRONG:

&gt;=10% reproducible latency effect

PROMISING:

&gt;=5%

WEAK:

3~5%

REJECT:

&lt;3% or inconsistent

Signal이 처음부터 &lt;2%이고 noise 수준이면

해당 hypothesis를 조기 종료한다.

---

# H1. Balance–Fragmentation Paradox

## Hypothesis

MoE에서 "더 균등한 expert load"가 항상 더 빠른 것은 아니다.

같은 total assignments와 같은 per-rank load를 유지해도,

A:

few experts receive large token groups

vs

B:

many experts receive small token groups

일 때 B가 더 느릴 수 있다.

원인 후보:

- many small grouped-GEMMs

- poorer tile efficiency

- launch/scheduling overhead

- reduced per-expert M dimension

즉:

load balance

≠

GPU efficiency

일 수 있다.

## Experiment

Controlled expert replay.

같은:

- total assignments

- rank load

- hidden dimension

을 유지.

각 rank 안에서 active experts 수를 sweep:

1 / 2 / 4 / 8 / 16 experts per rank

예:

rank당 1024 assignments 고정.

case A:

1 expert × 1024

case B:

4 experts × 256

case C:

16 experts × 64

Measure:

- Expert CUDA

- T_MoE if possible

## Important

rank imbalance는 반드시 고정한다.

우리가 보고 싶은 것은

"load balancing"이 아니라

"within-rank fragmentation" 효과다.

## Strong signal

same rank workload인데

fragmentation만으로 expert latency &gt;=10% 차이.

## Research implication

Fragmentation-aware MoE balancing /

GPU-efficiency-aware routing.

Priority: VERY HIGH.

---

# H2. Expert Fragmentation Scaling Law

## Hypothesis

MoE latency는 total token count보다:

- active expert count A

- average tokens per active expert N/A

- expert-size distribution

에 의해 추가적으로 설명된다.

즉:

T_expert != f(N) only

and perhaps

T_expert ≈ f(N, A, fragmentation)

## Experiment

H1보다 넓은 sweep.

N:

128 / 256 / 512 / 1024 / 2048 / 4096 routed assignments per rank

A:

1 / 2 / 4 / 8 / 16 active experts

각 (N,A)에서 latency surface 작성.

Calculate:

fragmentation = active_experts / assignments

or

mean tokens per active expert.

Plot:

x = tokens / active expert

y = cost / assignment

## Strong signal

token count only model 대비

fragmentation feature가 latency prediction error를

meaningfully 감소.

Priority: VERY HIGH.

---

# H3. Router Uncertainty as a Systems-Cost Signal

## Hypothesis

router confidence가 낮은 token은

단순히 semantic uncertainty만 높은 것이 아니라,

- top-k experts가 더 많은 ranks에 퍼지고

- active expert set을 확장하고

- grouped GEMM을 fragment하고

- EP cost를 증가

시킬 수 있다.

즉 router uncertainty가

systems cost predictor일 수 있다.

## Metrics

per token:

- top1-top2 margin

- entropy / normalized entropy

- top-k probability concentration

- number of distinct destination ranks

- contribution to active expert union

## Experiment

Real Qwen3 / Qwen3-VL routing.

high-confidence tokens vs low-confidence tokens를

같은 token count로 sampling.

가능하면 operator replay로

matched high-entropy / low-entropy batches 구성.

## Strong signal

router entropy/margin이 token count control 후에도

T_MoE-related metric과 strong relationship.

## Prior-art caution

MACS도 entropy를 사용하지만 semantic importance/capacity 목적이다.

단순히 MACS를 재현하지 않는다.

우리가 찾는 것은

"routing uncertainty → GPU execution geometry/cost".

Priority: HIGH.

---

# H4. Route-Shape Transition Penalty

## Hypothesis

같은 workload B라도

B → B

steady-state execution과

A → B

routing shape가 갑자기 변한 직후 execution latency가

다를 수 있다.

원인 후보:

- CUDA graph / dynamic-shape behavior

- allocator / workspace state

- DeepEP metadata preparation

- kernel/autotune/cache state

즉 serving latency는 현재 workload뿐 아니라

previous workload shape에도 영향을 받을 수 있다.

## Experiment

Create two very different shapes:

A:

small / concentrated route

B:

large / fragmented route

Compare current B latency under:

B,B,B,B

vs

A,B,A,B

Measure only B iterations.

Also test:

small → large

large → small

balanced → skewed

skewed → balanced

## Strong signal

same current B input인데 history에 따라 &gt;=5% latency difference.

Priority: VERY HIGH.

---

# H5. Temporal Expert Warmth / Cache Locality

## Hypothesis

모든 expert weights가 HBM에 있어도,

바로 직전 invocation에서 사용된 expert set과

현재 expert set이 유사하면 current expert execution이

더 빠를 수 있다.

Possible causes:

- L2/cache residency

- metadata/cache reuse

- persistent kernel state

- memory-access locality

## Experiment

Target batch B 고정.

Precede B with:

1. route-similar priming batch

2. route-disjoint priming batch

3. idle/cache-flush-like control if feasible

Then measure B only.

Repeat randomized ordering.

## Strong signal

current B latency differs &gt;=3~5% reproducibly.

## Research implication

temporal routing locality could become scheduling signal

without changing model outputs.

Priority: VERY HIGH.

---

# H6. Per-Token EP Rank-Fanout Tax

## Hypothesis

같은 total routed assignments라도

한 token의 top-k experts가

몇 개의 destination EP ranks에 걸쳐있는지가

Dispatch/Combine latency에 영향을 준다.

Example:

Token A top-8:

2 destination ranks

Token B top-8:

4 destination ranks

B may be more communication-expensive.

## Measurement

For every token compute:

fanout(t) =

# unique destination EP ranks among top-k experts

Analyze natural distribution.

Then build controlled DeepEP replay with:

- same assignment count

- similar rank totals

- different average per-token fanout

Measure:

- Dispatch

- Combine

- T_MoE

## Strong signal

fanout alone produces &gt;=5% communication difference.

Priority: HIGH.

---

# H7. Token-Order Sensitivity with Identical Routes

## Hypothesis

Even when routing assignments are exactly identical,

token ordering before MoE dispatch may affect latency.

Compare:

- original token order

- random order

- rank-grouped order

- expert-grouped order

- routing-signature-grouped order

All produce mathematically identical MoE result after inverse permutation.

## Important

Expert assignments and values must remain identical.

Only physical ordering changes.

## Measure

- prepare/permutation

- dispatch

- expert

- combine

- T_MoE

## Strong signal

semantics-preserving ordering alone changes T_MoE &gt;=5%.

## Research implication

zero-quality-loss token-layout optimization.

Priority: HIGH.

---

# H8. DP Partition Sensitivity

## Hypothesis

Under TP2/DP2/EP4,

the same global set of requests can have different MoE latency

depending on how requests are partitioned between DP replicas.

Global routing histogram can remain approximately identical,

while sender→destination traffic matrices differ.

## Experiment

Take small pool of 4–8 real requests.

Enumerate or sample different assignments:

DP0: {requests ...}

DP1: {requests ...}

Keep total workload equal.

Measure T_MoE.

## Required analysis

Does latency variability correlate with:

- sender/destination matrix entropy

- per-DP routed volume

- rank imbalance

- expert fragmentation?

## Strong signal

same global batch,

DP partition alone causes &gt;=5–10% change.

## Prior-art caution

Semantic Parallelism addresses model-data co-scheduling.

If positive, compare carefully before claiming novelty.

Priority: MEDIUM-HIGH.

---

# H9. Physical Rank Mapping Sensitivity

## Hypothesis

Logical TP/DP/EP rank placement onto physical GPUs

may affect latency even on a nominally symmetric 4-GPU node.

## Experiment

For CUDA_VISIBLE_DEVICES=4,5,6,7,

permute logical rank mapping if runtime permits.

Examples:

4,5,6,7

4,6,5,7

7,6,5,4

etc.

Use identical workload.

Collect topology/P2P information if available.

## Strong signal

mapping changes T_MoE &gt;=5%.

Priority: MEDIUM.

---

# H10. Layer-Specific Cost Regimes

## Hypothesis

All 48 MoE layers have identical expert architecture,

but their real routing distributions can place them in

different GPU efficiency regimes.

Some layers may systematically have:

- more active experts

- smaller tokens/expert

- higher fanout

- higher fragmentation

- worse cost per assignment

## Experiment

For real requests,

measure every layer.

Compute:

cost_per_assignment(layer)

and correlate with:

- active experts

- fragmentation

- HHI

- fanout

- rank imbalance

Check whether expensive layers are persistent across requests.

## Strong signal

same token volume에서

specific layer groups consistently &gt;=10% worse

and explainable by route geometry.

## Research implication

layer-aware execution strategy.

Priority: HIGH.

---

# H11. Prefill vs Decode Equal-Work Phase Gap

## Hypothesis

Prefill and decode MoE execution may differ

even after matching routed assignment count.

If equal routing workload produces different latency,

phase-specific runtime behavior exists.

## Experiment

Capture real:

- prefill route

- decode route

Construct matched replay by:

- assignment volume

- active expert count if possible

Compare cost per assignment.

## Required

Do not conclude from normal prefill-vs-decode latency alone.

Control token volume.

## Strong signal

matched workloads differ &gt;=10%.

## Prior-art caution

decode expert locality is already actively studied.

This is characterization first, not a proposed method.

Priority: MEDIUM.

---

# H12. Multi-Image Composition Penalty

## Hypothesis

In MLLM,

the same total number of vision tokens may cost differently depending on

whether they originate from:

- one image

- multiple images

Multi-image inputs may increase routing diversity /

expert fragmentation.

## Experiment

Construct approximately matched vision-token budgets:

A:

single large/high-resolution image

B:

2 images

C:

4 images

Use real images.

Control:

- total vision tokens

- text tokens

Measure:

- active experts

- fragmentation

- rank fanout

- T_MoE

## Strong signal

at matched vision-token volume,

image count changes normalized T_MoE &gt;=5%.

Priority: HIGH (MLLM-specific).

---

# H13. Visual Semantic Complexity → Systems Cost

## Hypothesis

At equal vision-token count,

visual content type may change expert routing geometry and MoE latency.

Compare:

- simple/low-entropy image

- natural photo

- chart

- document/OCR

- dense diagram

- high-text screenshot

## Control

Match image token count as closely as possible.

## Measure

- expert set size

- routing entropy

- rank fanout

- fragmentation

- cost per vision assignment

- T_MoE

## Strong signal

visual category produces &gt;=5% normalized cost difference

at matched token volume.

## Research implication

MLLM serving cost may be content-dependent,

not just token-count-dependent.

Priority: VERY HIGH (MLLM-specific).

---

# H14. Spatial Geometry / Aspect-Ratio Cost

## Hypothesis

At approximately equal vision-token count,

spatial token geometry may alter routing locality and therefore

expert fragmentation / execution efficiency.

Compare images with:

- square

- wide

- tall

- tiled/mosaic

Control total visual tokens.

Analyze:

- adjacent routing similarity

- active expert set

- tokens/expert

- T_MoE

## Strong signal

geometry alone gives &gt;=5% normalized latency difference.

Priority: MEDIUM.

---

# H15. MoE Latency Sufficient-Statistics / Residual Mining

This is the final meta-experiment.

## Question

Can T_MoE be predicted almost entirely from a small set of variables?

Candidate features:

1. total routed assignments N

2. active experts A

3. tokens / active expert

4. max rank load

5. rank CV

6. expert HHI

7. average per-token rank fanout

8. router entropy

9. modality fraction

10. workload phase

Fit only simple interpretable models:

- linear regression

- polynomial terms if needed

- shallow tree only as diagnostic

Do NOT build a complicated ML predictor.

## Goal

Find:

T_MoE ≈ f(...)

Then inspect high residual cases:

predicted fast but actually slow

predicted slow but actually fast

Those residual cases are candidates for NEW hypotheses.

## Important

This experiment should integrate results from H1-H14.

## Strong outcome A

3–4 simple variables explain &gt;95% of variance.

Then those variables reveal the true systems control plane.

## Strong outcome B

large systematic residual clusters remain.

Then automatically characterize those clusters and propose

up to 3 follow-up hypotheses.

Priority: VERY HIGH / FINAL.

---

# Execution Priority

Do NOT simply run H1→H15 blindly.

Priority Tier A:

H1

H2

H4

H5

H6

H10

H13

H15

Then Tier B:

H3

H7

H8

H11

H12

Then if time remains:

H9

H14

If a hypothesis requires &gt;45 minutes of debugging before any data is collected,

mark:

BLOCKED

and move on.

Do not consume the night fixing one experiment.

---

# Time Budget

Total target &lt;= 12 hours.

Suggested:

Environment / smoke test:

30 min

Tier A:

~6 hours

Tier B:

~3.5 hours

Tier C + residual follow-ups:

~1.5 hours

Final analysis/report:

30–45 min

The exact schedule may be adjusted autonomously.

---

# Artifact Policy

Do not save huge hidden-state dumps unless required.

Prefer:

CSV / JSON / NPZ summaries.

Each hypothesis directory:

results/overnight_discovery/&lt;Hxx_name&gt;/

containing:

- config.json

- metrics.csv

- summary.json

- plots/

- logs/

Final report:

poc_flashvep/reports/

overnight_moe_ep_discovery_[report.md](http://report.md)

Also create:

poc_flashvep/reports/

overnight_moe_ep_discovery_scoreboard.csv

Columns:

hypothesis

status

max_effect_pct

robust_effect_pct

main_metric

causal_control_pass

novelty_risk

runtime_minutes

recommend_followup

summary

---

# Final Ranking

At the end rank all hypotheses by:

ResearchScore =

EffectStrength

× Robustness

× CausalClarity

× NoveltyPotential

× Implementability

Use qualitative 1–5 scores.

Do NOT rank merely by largest latency number.

A 7% clean causal phenomenon may be more valuable than

a noisy 20% result.

---

# Final Answer Format

OVERALL DISCOVERY STATUS:

...

TOP 5 FINDINGS:

Rank 1:

Hypothesis:

Status:

Effect:

Why interesting:

Likely mechanism:

Prior-art risk:

Recommended next PoC:

...

SURPRISING NEGATIVE RESULTS:

...

GENERIC LLM-MOE FINDINGS:

...

MLLM-SPECIFIC FINDINGS:

...

BEST RESEARCH DIRECTION:

...

SECOND BEST:

...

DO_NOT_PURSE:

...

NEW HYPOTHESES GENERATED FROM RESIDUALS:

...

Total runtime:

GPU:

Branch:

Commit:

Push:

Report:

Scoreboard:

Results root: