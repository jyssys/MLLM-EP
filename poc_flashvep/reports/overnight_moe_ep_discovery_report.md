# Overnight MoE-EP phenomenon discovery (bounded sprint)

## Executive decision

This sprint did **not** find a new robust optimization opportunity.  The most
useful positive result is a falsification: at fixed global work and fixed EP
rank load, spreading the same Qwen3-VL assignments over more active experts did
not slow the real DeepEP/TritonExperts path.  The A=16 condition was faster than
the fair A=2 unique-ID control by 50.1% at M=512 and 33.6% at M=1024.  This is
the opposite of the balance--fragmentation hypothesis and makes a
fragmentation-aware execution policy unattractive without a new mechanism.

History-conditioned replay produced one interesting but unstable signal: a
similar-route prime had a 27.8% higher target-B median than the steady control,
but its five target cases had 72.7% CV and the same low-latency tail appeared in
all conditions.  It is a measurement follow-up candidate, not a finding.

**Overall discovery status: NO STRONG NEW PHENOMENON FOUND.**

No scheduler, routing, placement, dropping, merging, RL, or kernel method was
implemented.

## Environment and provenance

| Item | Value |
|---|---|
| Repository | `jyssys/MLLM-EP` |
| Branch | `flashvep/overnight-moe-ep-discovery` |
| Base commit | `dac63be` |
| Model | Qwen3-VL-30B-A3B-Instruct, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| GPUs | physical 1,2,3,4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`) |
| Topology | TP2 / DP2 / EP4 / PP1 |
| Backend | DeepEP high-throughput; TritonExperts; BF16; eager; DBO off; prefix cache off |
| Placement | linear, expert `e // 32` |
| New measured replay | persistent vLLM worker, 2 warmups + 5 repetitions for H1; 1 warmup + 3 repetitions per H4/H5 condition |

The worker logs show ranks 0–3 bound to the four visible devices and
`DeepEPHTPrepareAndFinalize`/`TritonExperts`.  An inherited trigger metadata
field still says `physical_gpus=[4,5,6,7]`; this is stale provenance from the
driver and is explicitly not used as physical-device evidence.  `nvidia-smi`
showed the new workers on physical 1–4 and no new process was launched on
physical 0 or 5–7.

New-run wall time was approximately 6.5 minutes (two H1 runs and four history
conditions), or **about 0.43 GPU-hours** at four GPUs.  Offline analysis used
preserved artifacts and consumed no GPU time.

## Tier-A results

### H1 — Balance–fragmentation paradox (new controlled replay)

Cases used **synthetic, balanced route IDs** on the real Qwen3-VL BF16
DeepEP/TritonExperts operator path; they are a causal operator diagnostic, not
a natural model-routing trace.  They had the same M, total assignments, and
per-rank assignments.  A=1 is a
diagnostic only because it must duplicate top-k expert IDs; A=2 is the fair
unique-ID baseline.

| M | A=2 expert critical ms | A=16 expert critical ms | A=2→A=16 expert change | A=2→A=16 critical-wall change | Verdict |
|---:|---:|---:|---:|---:|---|
| 512 | 0.9948 | 0.4964 | −50.09% | −56.99% | NO-GO |
| 1024 | 0.9622 | 0.6393 | −33.56% | −44.66% | NO-GO |

The route construction, token count, placement, precision, backend, and EP4
topology were unchanged.  Raw rank rows and correctness checks are in the two H01 result
directories.  The result falsifies “more fragmented active experts is
intrinsically slower” for these real-kernel replay shapes; it does not imply
that every future shape benefits from fragmentation.

### H2 — Fragmentation scaling law (offline, leave-request-out)

The nine-request real Qwen3-VL replay table (M=32…512, layers 4/24/44,
text/vision) was used without changing routes.  A linear model using total
assignments alone had median held-out RMSE **0.1567 ms**; adding active experts,
mean local expert M, rank CV, and per-token rank fanout gave **0.1571 ms**
(−0.27% RMSE reduction).  Thus the saved data provides no incremental
fragmentation signal above measurement/kernel-regime variation.  Gate: **NO-GO**;
the requested 4096-assignment surface was not rerun because existing evidence
already falsified the Tier-A effect and the overnight budget is bounded.

### H4 — Route-shape transition penalty (same current B route)

The target was always `vision.deep_field.npz`, layer 24, first 256 tokens.  The
prime route immediately before B was changed; the target route itself and all
operator settings were fixed.

| Prime condition | B critical-wall median | vs steady |
|---|---:|---:|
| steady | 2.0302 ms | reference |
| alternating, 32-token text A | 2.1200 ms | +4.42% |
| disjoint, 256-token text D | 2.1142 ms | +4.14% |

Both effects are below the preregistered 5% follow-up threshold.  Gate:
**NO-GO**.  A low-latency final case appeared across conditions, so the
case-level CV is not evidence of route transition.

### H5 — Temporal expert warmth/cache locality

The same B route was primed with a similar vision route or a disjoint text route.

| Prime condition | B critical-wall median | CV |
|---|---:|---:|
| steady | 2.0302 ms | 4.06% |
| similar vision | 2.5953 ms | 72.72% |
| disjoint text | 2.1142 ms | 38.39% |

The similar-prime median is a potentially large signal, but one 7.78-ms case
and one recurring 0.89-ms tail make the effect non-robust at n=5.  The target
case count is too small to distinguish cache warmth from replay state, kernel
selection, or teardown-adjacent timing.  Gate: **HOLD**, not GO.  The cheapest
decisive follow-up would be a single persistent worker with randomized
condition order, 20+ target-B repetitions, and direct cache/kernel-selection
markers.

### H6 — Per-token EP rank fanout tax

Across 95,577 preserved route tokens, unique destination-rank fanout had mean
3.60, median 4, and p90 4 (EP4).  The preserved paired table correlates rank
runtime with assignment volume at Pearson **0.977**, but contains no independent
fanout-controlled DeepEP replay.  Gate: **NO-GO for a causal claim**; a new
controlled replay would be required before treating fanout as a cost signal.

### H10 — Layer-specific cost regimes

Layers 4/24/44 were compared at identical M.  At M=512, critical-wall medians
spanned 0.9469–1.4693 ms (45.3% spread).  The difference is real in the replay
table, but does not form a stable modality-specific or shape-specific rule and
is strongly entangled with kernel regime and M.  Gate: **HOLD** (diagnostic,
not a new policy).

### H13 — Visual semantic complexity

At equal M, vision categories `natural`, `chart_document`, and `fine_grained`
showed a 30.2% max-to-median spread at M=512.  The category curves are not
consistent across M/layer and the existing real-image tile study found no
stable ≥5% normalized mechanism.  Gate: **HOLD**; no semantic-complexity
controller is justified.

### H15 — Sufficient statistics and residual mining

The simple full-table model using assignments, active experts, mean local M,
rank CV, and fanout produced R² **0.326** and RMSE **0.1998 ms** on the saved
operator table; 37/90 rows had absolute residual ≥10%.  Residuals cluster around
M/layer/kernel-regime changes rather than a clean modality or routing-history
cluster.  This is a **HOLD** diagnostic: the current table is too heterogeneous
for a strong universal model, but it does not reveal a validated new mechanism.

## Tier-B/C artifact mining

No broad GPU reruns were justified after the Tier-A falsifications.  The
following prior, real-route artifacts were linked into the current result root
with provenance files:

| Hypothesis | Result | Verdict |
|---|---|---|
| H3 router uncertainty | Router scores/margins were not preserved in the route artifact | BLOCKED |
| H7 identical routes/order | sequential 2×2 1.0105×, spatial 1.0031×, generic 1.0002× | NO-GO |
| H8 DP partition | median 1.0139×, best 1.0231× | NO-GO |
| H9 physical rank mapping | no controlled remapping in this sprint | BLOCKED |
| H11 prefill/decode equal work | static tails <4%, no repeatable dynamic tail | HOLD/NO-GO |
| H12 multi-image composition | TP/EP relative gain tracked volume; no modality crossover | HOLD |
| H14 spatial geometry | prior spatial/order effects <5% | NO-GO |

These are artifact-backed controls, not new claims.  The existing Qwen3-VL
granularity study also showed text and vision total ms/token curves nearly
overlapping, both preferring M=512; this independently argues against a new
modality execution policy.

## Candidate scoreboard and Top-5 findings

The machine-readable table is
`poc_flashvep/reports/overnight_moe_ep_discovery_scoreboard.csv`.  Scores are
1–5 for effect strength, robustness, causal clarity, novelty potential, and
implementability.

### 1. Fixed-work fragmentation is not a monotone penalty (H1)

- **Effect:** A2→A16 critical wall −57.0% (M512), −44.7% (M1024).
- **Causal control:** exact same M, assignments, rank load, placement, backend;
  A2 removes the A1 duplicate-ID confound.
- **Mechanism:** measured kernel behavior favors these dense local shapes; the
  expected fragmentation cost is absent/reversed.
- **Why surprising:** it directly contradicts the intuitive active-expert
  fragmentation story and prior proxy-only expectations.
- **Scope:** MoE/EP-specific operator evidence, not MLLM-specific (the replay
  route is real Qwen3-VL but the control holds token/routing structure).
- **Novelty risk:** could be a shape/kernel-regime artifact; do not claim a
  general law.
- **Cheapest decisive next experiment:** repeat the same fixed-work matrix on
  one generic MoE checkpoint with a pre-registered route seed.

### 2. Temporal warmth is a candidate, not yet a phenomenon (H5)

- **Effect:** similar-prime target median +27.8% vs steady.
- **Control:** current target B route is identical; only previous route changes.
- **Mechanism:** possible cache/kernel-selection state, but 72.7% CV and common
  low tail prevent attribution.
- **Scope:** potentially MoE/EP-specific; MLLM-specificity unknown.
- **Novelty risk:** high risk of replay-order confounding.
- **Cheapest decisive next experiment:** randomized 20–30 repeats in one worker
  with explicit prime/target timing and no final-case special position.

### 3. Volume/kernel regime dominates the saved latency curves (H2/H15)

- **Effect:** adding shape features reduced held-out RMSE by −0.27%; full-table
  simple fit R² only .326 because M/layer regimes are heterogeneous.
- **Control:** leave-request-out validation and exact preserved routes.
- **Mechanism:** fixed launch/kernel regimes explain more than a universal
  fragmentation statistic; residuals are regime-shaped.
- **Scope:** generic MoE execution diagnostic; not a novel method.
- **Novelty risk:** crowded cost-model territory.
- **Cheapest decisive next experiment:** pre-register M/layer bins and remeasure
  each bin with 20 paired repetitions before fitting any predictor.

### 4. Layer/category differences exist but are not a control plane (H10/H13)

- **Effect:** 45.3% layer spread at M512 and 30.2% vision-category spread.
- **Control:** same M, matched layers/categories, existing real routes.
- **Mechanism:** likely kernel regime and route-size interactions.
- **Scope:** workload-dependent MoE/MLLM diagnostic; no robust EP intervention.
- **Novelty risk:** easily rediscovered as per-layer/kernel tuning.
- **Cheapest decisive next experiment:** one layer-normalized shape matrix with
  independent remeasurement, not a scheduler.

### 5. Fanout/order/partition effects are small in measured real traces (H6/H7/H8/H14)

- **Effect:** fanout is usually already 4 ranks; order/spatial/partition gains
  are ≤2.31% in the best saved oracle.
- **Control:** identical routes/order and paired partition controls.
- **Mechanism:** assignment volume and kernel launch dominate these bounded
  cases.
- **Scope:** MoE/EP-specific but negative.
- **Novelty risk:** crowded communication scheduling/placement literature.
- **Cheapest decisive next experiment:** do not pursue unless a new backend or
  scale changes the effect by >5%.

## Research ranking

### BEST RESEARCH DIRECTION

Do **not** implement a method yet.  The only worthwhile next bounded study is a
proper H5 temporal-state validation: randomize prime/target order in one
persistent worker, collect ≥20 target repetitions, and instrument cache/kernel
selection state.  It is cheap and directly tests the only >10% residual signal
seen this sprint.  If it disappears, close the temporal-locality direction.

### SECOND-BEST RESEARCH DIRECTION

A pre-registered, layer-normalized kernel-regime model (H15) could explain why
the same assignment volume changes cost at M/layer boundaries.  This is a
diagnostic cost model, not an optimization method, and should only continue if
independent remeasurement produces systematic residual clusters.

### INTERESTING BUT PRIOR-ART-CROWDED

Generic fragmentation-aware batching, route-order scheduling, expert placement,
capacity/rerouting, chunked-prefill control, and TP/EP switching are already
well represented by the prior work audited in earlier project reports.  The
current negative results do not add a distinctive causal angle to them.

### DO NOT PURSUE

Do not implement dynamic communication scheduling, expert placement, token
pruning/merging, persistent/fused kernels, RL controllers, or modality-aware
granularity based on the current evidence.  H1 is reversed, H4/H6/H7/H8/H14
are below threshold, and H13/H15 are not robust causal mechanisms.

## Questions Q1–Q4

1. **Dense model analogue:** H1/H6/H7/H8 concern distributed expert routing
   and do not have a meaningful dense analogue; H10/H13 could occur in dense
   kernels and therefore are not MLLM-specific.
2. **MoE without EP:** fragmentation, fanout, and destination-rank effects are
   undefined or materially changed without EP; H1 is EP-relevant, but this
   sprint does not prove an EP-only universal law.
3. **Routing × distributed execution required:** H1/H6/H7/H8 require expert
   assignments and EP destination execution, so Q3 is YES for their definitions;
   the effects are negative or too small for a method.
4. **Text-only MoE control:** prior text/vision matched replays showed nearly
   overlapping ms/token curves and no robust vision-specific execution gap;
   current evidence therefore does not support an MLLM-only phenomenon.

## New residual hypotheses

1. **H16 – replay-state/kernel-selection tail:** the common final low-latency
   case across H4/H5 conditions suggests persistent-worker state or measurement
   position, not route history.  Test with randomized order and explicit cache
   markers.
2. **H17 – layer × M launch-regime interaction:** the H10 spread and H15 residuals
   suggest a piecewise kernel regime at M=512 in some layers.  Test independent
   per-layer repetitions before proposing any runtime control.
3. **H18 – metadata/driver provenance mismatch:** inherited `physical_gpus` fields
   can silently disagree with actual device binding.  Add an automated
   `nvidia-smi`/PID-to-visible-device proof to future experiments.

## Artifacts

- Spec: `poc_flashvep/reports/overnight_moe_ep_discovery_specs.md`
- Scoreboard: `poc_flashvep/reports/overnight_moe_ep_discovery_scoreboard.csv`
- Result root: `poc_flashvep/deepep_revalidation/results/overnight_moe_ep_discovery_20260905/`
- New analysis code: `poc_flashvep/overnight_moe_ep_discovery/`
- H1 plots/metrics: `H01_balance_fragmentation/` and `H01_balance_fragmentation_M1024/`
- H4/H5 target timing and history plot: `history_target_case_timings.csv`,
  `history_condition_summary.csv`, `history_condition_latency.png`,
  `history_summary.json`
- Offline figures and tables: `H06_rank_fanout/`, `H10_layer_regimes/`,
  `H13_visual_semantic_complexity/`, `H15_residual_mining/`

All raw prior artifacts remain in place; this sprint only adds new result
directories and provenance links.
