# Deep PoC Specification
# EP-Aware Dynamic Block Diffusion for LLaDA2.0-Flash
# Block Size / AR-ness × Sparse-MoE Expert-Parallel Execution Regime

## 0. Mission

This is a **deep, persistence-oriented PoC**, not a quick kill test.

Primary research question:

> Does diffusion block size (and therefore AR-ness) systematically change the physical execution regime of sparse-MoE expert-parallel inference, and can a quality-constrained dynamic block-size policy exploit this to beat every fixed-block baseline?

The target is not merely to show that block size affects latency or quality.

The desired evidence chain is:

```text
block size / AR-ness changes
        ↓
physical EP workload geometry changes
        ↓
different bottlenecks / different system-optimal block sizes emerge
        ↓
the optimal safe block size varies across requests or block states
        ↓
a dynamic block schedule has meaningful quality-constrained E2E headroom
        ↓
observable EP state can predict that choice
```

The final method, only if the data supports it, is:

> **EP-Aware Dynamic Block Diffusion**

The system chooses the next diffusion block size using distributed sparse-MoE execution state, with a quality guard.

Do **not** implement a production controller before establishing a strong dynamic oracle.

---

# 1. Experimental Budget

Available GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

Use **only physical GPUs 4,5,6,7**.

Do not use GPUs 0,1,2,3.

The experiment may use close to the full available ~10-hour wall-clock budget on all four GPUs if scientifically useful.

Do not terminate the study after one failed setting.

A failed setting should trigger diagnosis and controlled retries as defined in the Failure-Recovery section.

However, do not fabricate positive results or continue a direction after its upper bound is conclusively small.

---

# 2. Primary and Secondary Substrates

## 2.1 Primary model

```text
inclusionAI/LLaDA2.0-flash
```

Primary final claims must be validated on the 100B model.

Expected high-level properties:
- ~100B total parameters
- 32 transformer layers
- sparse MoE
- block-diffusion decoding
- official/default inference around `block_length=32`, `steps=32`

Before experiments, record the **actual downloaded revision and actual config**.

## 2.2 Primary distributed path

Prefer the already validated production-like large-model path:

```text
dense TP4
+
routed EP4
+
DeepEP dispatch
+
owner-rank fused routed-expert execution
+
reverse combine
```

Revalidate on GPUs 4,5,6,7.

Record:
- TP / EP / DP / SP
- expert ownership
- shared-expert placement
- dense-layer placement
- communication backend
- fused expert backend

Do not assume the topology is unchanged simply because four GPUs are used.

## 2.3 Secondary/debug models

If needed for decoder correctness or rapid block-schedule debugging, the following may be used as **secondary diagnostic substrates only**:

```text
LLaDA2.0-mini
LLaDA-MoE
```

Rules:
- do not cherry-pick a small model as the headline result;
- any promoted method must return to LLaDA2.0-Flash;
- small-model results may be used to distinguish model behavior from runtime bugs.

---

# 3. Common Quality Workloads

Primary tasks:

```text
GSM8K
HumanEval
```

Use identical bounded request sets across block-size settings.

Prefer:
- 32 requests/task for static sweeps;
- a smaller but fixed representative subset for expensive counterfactual schedule enumeration;
- full 32-request revalidation for promoted policies.

If signal is strong and time permits, add one secondary task such as:

```text
MGSM
MBPP
```

for robustness.

Do not select a dataset solely because it produces a favorable system result.

Report all attempted tasks.

---

# 4. Output-Length Strategy

A block-size study is meaningless if generation is too short to contain multiple blocks.

Use at least two generation regimes if feasible:

## Diagnostic regime

```text
gen_length ≈ 128
```

Purpose:
- fast static sweep;
- exact/near-exact dynamic schedule enumeration;
- block-by-block analysis.

## Validation regime

```text
gen_length ≈ 256 or 512
```

Purpose:
- realistic multi-block behavior;
- validate best static and dynamic candidates;
- expose long-horizon quality drift.

Use task-appropriate lengths when necessary.

Do not compare settings with accidentally different effective output budgets.

---

# 5. Core Terminology

## 5.1 Block size B

`B` = number of token positions jointly refined inside one diffusion block.

Interpretation:

```text
small B -> more AR-like / stronger AR-ness
large B -> more diffusion-like / weaker AR-ness
```

The study should initially consider:

```text
B ∈ {8, 16, 32, 64, 128}
```

subject to runtime correctness / memory.

If B=8 or B=128 is invalid under a specific runtime path, diagnose why before removing it.

At minimum try to establish valid points around:

```text
16, 32, 64
```

because they enable tractable dynamic-schedule enumeration.

## 5.2 Steps

In the reference LLaDA2 implementation, `steps` is the per-block refinement budget and interacts directly with `block_length`.

Therefore `B` and `steps` must not be conflated.

## 5.3 Fixed block vs dynamic block

Fixed:

```text
32 -> 32 -> 32 -> 32
```

Dynamic:

```text
16 -> 64 -> 32 -> 16 -> ...
```

A dynamic policy may choose the **next block size at a block boundary**.

Do not initially change block size in the middle of an already-open block.

Block-boundary adaptation is the primary PoC scope because its semantics are easier to verify.

---

# 6. Formal Objective

The desired controller solves approximately:

```text
minimize   request E2E latency / BCT
```

subject to:

```text
quality >= baseline quality - epsilon
```

where the baseline is the strongest valid fixed-block policy, not automatically B=32.

Also report the full Pareto frontier rather than a single scalar objective.

The desired outcome is not necessarily an accuracy-latency tradeoff.

A successful dynamic policy may produce a **Pareto improvement**:

```text
same quality + lower latency
or
higher quality + same/lower latency
```

---

# 7. Stage 0 — Environment and Baseline Truth

Before any block sweep:

1. confirm GPUs 4,5,6,7 only;
2. record hardware/software versions;
3. reproduce the current best static EP4 baseline at the official/reference block setting;
4. verify outputs / NFE against prior known-good execution;
5. confirm no CPU offload;
6. confirm real routed EP4 traffic.

Baseline should record:
- BCT
- throughput
- NFE
- final output
- benchmark quality
- peak HBM
- attention time
- router/prep
- dispatch
- expert
- combine
- remote assignments / bytes
- rank fanout
- rank-load CV
- rows/expert
- active experts

Run clean timing with multiple restarts.

---

# 8. Stage 1 — Variable-Block Decoder Correctness

This stage is mandatory and should not be rushed.

Before performance conclusions, verify that different block sizes and arbitrary block schedules obey correct block-diffusion semantics.

## 8.1 Static B correctness

Run:

```text
B = 8,16,32,64,128
```

or the maximal valid subset.

For each B confirm:
- block partition;
- attention mask;
- prefix visibility;
- within-block bidirectional visibility;
- no future-block visibility;
- position IDs;
- termination;
- transfer-token schedule;
- NFE accounting.

## 8.2 Dynamic block schedule support

Implement or validate a schedule interface such as:

```text
block_schedule = [16, 32, 64, 16]
```

where block boundaries may vary but each block:
- sees all completed prefix blocks;
- sees itself bidirectionally;
- does not see future blocks.

The schedule should sum to or safely cover the generation budget.

## 8.3 Reference check

If dInfer dynamic-block execution is hard to trust:
1. implement/validate the semantics in the official/HuggingFace reference decoder;
2. reproduce outputs for fixed B=32;
3. verify several static alternative-B cases;
4. only then port the same block schedule semantics to the production EP runtime.

Do not interpret runtime bugs as block-size quality failure.

---

# 9. Stage 2 — Decoder-Effort Controls

This is one of the most important parts of the study.

Block size and steps interact strongly.

Run at least two controlled regimes.

## Regime A — Serving-native / fixed-step

Keep the currently validated decoding configuration fixed as much as possible, e.g.:

```text
steps = 32
threshold = baseline threshold
temperature = 0
```

and vary B.

Purpose:

> What happens if deployment changes only block size under the existing decoder configuration?

This is the real serving tradeoff.

## Regime B — Matched refinement granularity

Use a schedule approximately preserving the default `steps / B` relationship, with a key control such as:

```text
steps(B) = B
```

when feasible.

At the default B=32 this matches `steps=32`.

This makes the nominal minimum transfer quota approximately one token per refinement step and prevents the comparison from being explained only by different block-refinement budgets.

Purpose:

> What physical EP behavior comes specifically from B / work shape rather than just fewer/more denoising iterations?

## Optional Regime C — Quality-calibrated decoder

For each B, allow a small decoder calibration search over:
- steps;
- threshold;

to find the fastest configuration that preserves the target quality.

Do not conduct an enormous grid blindly.

Use a small diagnostic subset to identify promising settings, then validate.

---

# 10. Stage 3 — Static Block-Size Sweep

For each valid B and decoder regime, collect both semantic and systems metrics.

## 10.1 Semantic / decoding metrics

- GSM8K exact answer
- HumanEval pass@1
- final token sequence
- output length
- NFE total
- NFE/block
- accepted tokens / iteration
- decision-live ratio
- confidence distributions
- number of blocks
- early stop behavior

## 10.2 EP physical metrics

Per request, block, iteration, and representative layer:

```text
physical rows M
token-expert assignments
remote assignments
remote fraction
remote hidden bytes
dispatch bytes
combine bytes
active experts
unique experts
rows/expert mean/p50/p90
tiny-expert fraction
destination-rank fanout
rank-load CV
critical-rank load
dispatch time
expert time
combine time
attention time
router/prep time
whole block time
GPU utilization
HBM
```

## 10.3 Liveness / redundancy metrics

Measure:
- decision-live rows;
- physical rows;
- dead/stable physical work fraction;
- fresh-work fraction;
- potential Epoch-like removable fraction.

The goal is to see whether larger B structurally creates more live/dead mismatch.

---

# 11. Stage 4 — Runtime-Granularity Fairness Controls

Previous work showed `mini_batch_size` can dominate performance.

Therefore block-size comparisons must include fairness controls.

## 11.1 Fixed-runtime-knob view

Hold mini-batch / serving knobs fixed.

Purpose:
- measure what block size alone changes under a fixed runtime configuration.

## 11.2 Best-per-B view

For each B, perform a small calibration over feasible mini-batch sizes.

Example:

```text
mini ∈ {4, 8, 16, 32}
```

adjusted for submitted batch and HBM.

Find:
- best BCT;
- best throughput;
- memory limit.

This prevents a false claim caused by using a poor mini size for one B.

## 11.3 Matched-M control

Where feasible, choose mini sizes such that:

```text
B × mini_batch_size ≈ constant physical rows M
```

Example target M:

```text
B=16 -> mini32
B=32 -> mini16
B=64 -> mini8
B=128 -> mini4
```

or another feasible constant.

Purpose:

> Does B change routing / liveness / EP geometry even when the aggregate physical row count is held similar?

This is critical for separating block semantics from mere batch-shape effects.

---

# 12. Stage 5 — AR-ness × EP Regime Map

Build a central regime table:

| B | AR-ness | Quality | NFE | BCT | Attn % | MoE % | Dispatch % | Expert % | Combine % | Remote bytes | Fanout | Rows/expert | Dead-work % |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Required plots:

1. B vs BCT
2. B vs throughput
3. B vs quality
4. B vs NFE
5. B vs remote bytes
6. B vs token-expert pairs
7. B vs rows/expert
8. B vs active experts
9. B vs rank fanout
10. B vs rank-load CV
11. B vs dispatch/expert/combine share
12. B vs decision-live/physical ratio
13. B vs Epoch-like redundant work
14. B vs HBM

Do not assume linearity.

Look for:
- U-shaped curves;
- knees;
- crossover points;
- change-points;
- regime transitions.

Use:
- Spearman;
- piecewise trends;
- matched-M analysis;
- not only Pearson.

---

# 13. Stage 6 — Strong Observation Search

We want to discover whether block size changes the *kind* of bottleneck.

Possible examples:

```text
small B:
startup dominated
tiny expert GEMMs
low utilization

medium B:
best amortization
balanced execution

large B:
remote traffic / fanout / liveness waste dominated
```

The strongest desired observation is:

> **AR-ness controls the physical bottleneck regime of dLLM-MoE EP.**

Promote only if:
- reproducible across requests/tasks;
- not explained solely by mini-batch size;
- not explained solely by steps/NFE;
- visible in actual clean request timing.

---

# 14. Stage 7 — Fixed-B Pareto Frontier

For every fixed B, construct its best quality-latency operating point.

Compare:
- official/default B=32;
- best fixed B under baseline decoder;
- best fixed B after allowed decoder calibration.

This establishes the strongest static frontier.

Dynamic block sizing must beat this frontier.

Do not compare dynamic B only against default B=32 if another fixed B is faster and equally accurate.

---

# 15. Stage 8 — Request-Level Heterogeneity

Before block-level dynamics, test the simpler question:

> Do different requests prefer different fixed block sizes?

For every request record:

```text
fastest B
fastest quality-safe B
best quality B
```

Create distributions.

If all requests choose the same B, dynamic control has less motivation.

If preferences differ systematically, identify predictors:

```text
prompt length
task
confidence
NFE
EP remote fraction
rows/expert
fanout
liveness
```

This is a lower-bound form of dynamic opportunity.

---

# 16. Stage 9 — Counterfactual Dynamic Block Oracle

This is the decisive stage.

Do **not** build the live controller first.

At each block boundary, allow B to be selected from a candidate set.

Primary tractable set:

```text
B ∈ {16, 32, 64, 128}
```

for a diagnostic `gen_length=128` if all are valid.

This set has a manageable number of valid block compositions compared with including B=8.

If B=8 proves important, add it later using beam search rather than blindly exhaustive enumeration.

## 16.1 True trajectory requirement

A schedule such as:

```text
16 -> 64 -> 32 -> 16
```

must be actually rolled out or faithfully emulated.

Do not create a fake oracle by summing independent per-block latencies from different trajectories.

Changing an early B can change:
- generated prefix;
- future hidden states;
- NFE;
- later quality;
- later costs.

The oracle must respect this.

---

# 17. Stage 10 — Oracle Levels

Construct several levels.

## O0 — Latency-only schedule oracle

Future knowledge is allowed.

Find the fastest valid schedule regardless of quality.

Purpose:

> absolute maximum systems headroom from dynamic B.

## O1 — Dataset-score-constrained oracle

Find the fastest schedule satisfying:

```text
dataset bounded score >= best-static baseline - epsilon
```

Use clear epsilon values, including:
- zero score drop;
- optionally <=1 percentage point or nearest meaningful bounded-set tolerance.

## O2 — Per-request preservation oracle

Stricter:

- if baseline GSM8K request is correct, dynamic schedule must keep it correct;
- if baseline HumanEval sample passes, dynamic schedule must keep it passing.

Baseline failures may improve.

This prevents one request's degradation from being hidden by another request's improvement.

## O3 — Strong semantic guard

Optional strict control:
- bounded score preserved;
- NFE increase limited;
- no severe output corruption;
- sequence divergence reported.

Do not require exact sequence if benchmark quality is preserved, but always report exact-sequence rate.

---

# 18. Stage 11 — Search Strategy for Dynamic Schedules

## Short-generation exact/near-exact search

For `gen_length=128` and candidate set `{16,32,64,128}`, enumerate all valid schedules if computationally feasible.

This is strongly preferred because it gives a real counterfactual oracle.

Use a fixed diagnostic subset from both tasks.

## Larger-generation search

For 256/512-token validation, exhaustive search is too expensive.

Use:
- beam search;
- branch-and-bound;
- oracle-informed candidate schedules derived from the short regime.

Always report the search approximation.

Do not call a beam-search result a mathematically exact oracle.

---

# 19. Stage 12 — Oracle Economics

For each task report:

```text
best fixed B latency
latency-only dynamic oracle
quality-safe dynamic oracle
strict per-request oracle
```

and gains:

```text
mean
median
worst-task
per-request distribution
```

Gate on the quality-safe request-level E2E gain.

Suggested gates:

```text
<5%      NO-GO
5–8%     CHARACTERIZATION
8–12%    HOLD
12–20%   STRONG
>20%     VERY STRONG
```

Given the project's history of 1–5% oracles, prefer >=12% before investing in a complex production controller.

---

# 20. Stage 13 — What Predicts the Oracle Choice?

Only if the dynamic oracle is meaningful.

At each block boundary capture **observable, future-free** features from the just-completed block / current serving state.

## EP/system features

```text
dispatch / expert / combine ratio
remote fraction
remote bytes
rank fanout
rank-load CV
rows/expert
tiny-expert fraction
active experts
GPU utilization
DeepEP startup/payload ratio
current mini-batch / ready-pool size
```

## dLLM semantic features

```text
decision-live ratio
acceptance rate
confidence mean/p50/p90
confidence margin
NFE used in current block
```

Predict target:

```text
next quality-safe oracle B
```

Compare simple predictors first:
- threshold rules;
- lookup table;
- shallow tree;
- linear / piecewise cost model.

Do not use a learned policy before simple rules.

---

# 21. Stage 14 — EP-only vs Semantic-only vs Joint Trigger

This is essential for novelty.

Compare:

## Policy A — Semantic-only

Uses:
- confidence;
- acceptance;
- semantic/refinement signals.

This approximates the kind of signal used by adaptive-block decoding prior work.

## Policy B — EP/system-only

Uses:
- remote bytes;
- rows/expert;
- fanout;
- dispatch/expert ratio;
- utilization;
- liveness cost.

## Policy C — Joint

System chooses a preferred B, with a semantic quality guard.

Example concept:

```text
EP state -> preferred B_system
semantic guard -> allowed B set
final B = fastest allowed system choice
```

Desired finding:

> EP/system state provides predictive value beyond semantic confidence alone.

If semantic-only matches the oracle and EP features add nothing, the EP-aware novelty is weak.

---

# 22. Stage 15 — Live Dynamic Prototype

Implement only if:

```text
quality-safe dynamic oracle >= 8%
```

Prefer >=12%.

The first live controller should be simple.

Example structure:

```text
if startup_dominated and quality_guard_allows_growth:
    increase B
elif remote/liveness_cost_high:
    decrease B
else:
    keep B
```

Do not hard-code this before the data supports it.

Compare:
- best fixed B;
- semantic-only adaptive B;
- EP-aware dynamic B;
- joint EP+semantic B.

Measure:
- clean BCT;
- throughput;
- quality;
- NFE;
- HBM;
- controller overhead;
- block-switch overhead.

---

# 23. Failure-Recovery Protocol

This section is mandatory.

Do not stop at the first negative result.

## Failure F1 — Alternative B crashes or produces malformed output

Do not conclude "variable B unsupported".

Check:
1. block attention mask construction;
2. total-length padding;
3. prompt/block alignment;
4. transfer-token schedule;
5. KV/prefix assumptions;
6. dInfer hard-coded B=32 assumptions;
7. position IDs;
8. EOS handling.

Reproduce the same B in the reference model.

Patch runtime only after confirming semantics.

---

## Failure F2 — Quality collapses for B != 32

Before concluding the model only supports B=32:

1. verify output against reference decoder;
2. test `steps=B`;
3. test a small threshold sweep;
4. verify temperature=0;
5. increase output length if truncation is causing failure;
6. inspect NFE / accepted-token schedule;
7. test neighboring B values;
8. use LLaDA2.0-mini as a decoder diagnostic;
9. return to Flash for final validation.

If only a subset of B is quality-valid, continue with that subset.

---

## Failure F3 — Latency results are noisy / inconsistent

Do not average blindly.

Use:
- randomized order;
- 5+ restarts for key points;
- warmup;
- clean node if possible;
- CV / p25-p75;
- paired comparisons.

Check:
- clocks / thermals;
- background load;
- DeepEP initialization;
- graph/cache warmup.

---

## Failure F4 — One B wins everything

Do not immediately kill dynamic B.

Check:
1. fixed-step regime;
2. matched-steps regime;
3. best-per-B mini calibration;
4. matched-M control;
5. longer generation;
6. request-level heterogeneity;
7. task heterogeneity.

If the same B still dominates across all controls and quality, then dynamic B is genuinely unnecessary.

---

## Failure F5 — EP metrics show no correlation with B

Check for nonlinear regimes.

Use:
- change-point plots;
- matched-M;
- phase-conditioned analysis;
- per-block rather than request aggregate metrics.

If physical geometry truly does not change, terminate the EP-aware hypothesis.

---

## Failure F6 — Dynamic oracle is small

Before killing:
1. confirm schedule search did not use independent-block fake costs;
2. expand the candidate B set if a boundary winner is missing;
3. inspect request-specific schedules;
4. try longer generation;
5. check whether quality constraints are overly strict due to bounded-score noise;
6. report both zero-drop and small-epsilon Pareto points.

If the true full-trajectory oracle remains <5%, stop.

---

## Failure F7 — Dynamic oracle is large but no simple trigger works

Do not immediately discard.

Check:
- which hidden future variables drive oracle choice;
- whether observable EP metrics lag by one block but remain predictive;
- whether task-level/request-level static selection captures most gain;
- whether a small lookup table suffices.

If only future information can predict the winner, report oracle-only characterization.

---

# 24. Anti-Cherry-Picking Rules

The user permits searching for favorable settings/models, but scientific rigor requires:

1. Log every attempted B / steps / threshold / dataset / model.
2. Separate:
   - discovery configuration;
   - validation configuration.
3. Do not omit negative tasks.
4. Any headline method must validate on LLaDA2.0-Flash.
5. If a method works only on mini or one dataset, label it accordingly.
6. Report best and worst-task gains.
7. Avoid selecting a configuration based on test-set accuracy without a held-out policy validation split.

---

# 25. Suggested 10-Hour Campaign Allocation

Use this as a flexible guide, not a rigid stopping clock.

## Phase 1 — ~0.5–1 h
Environment, baseline, variable-B correctness.

## Phase 2 — ~2 h
Static B sweep + decoder-effort controls.

## Phase 3 — ~1–1.5 h
mini calibration + matched-M + deep EP tracing.

## Phase 4 — ~2.5–3 h
Counterfactual dynamic schedule enumeration / oracle on diagnostic subset.

## Phase 5 — ~1 h
Robustness:
- longer generation;
- secondary task/model if useful;
- repeat key points.

## Phase 6 — remaining time
If oracle passes:
- simple live trigger prototype.

If oracle fails:
- use remaining time to falsify alternative explanations and strengthen the negative conclusion rather than forcing a method.

---

# 26. Required Statistical / Causal Controls

For key comparisons:

- multiple restarts;
- paired request sets;
- randomized configuration order;
- confidence intervals / bootstrap when easy;
- Spearman for monotonic trends;
- matched-M controls;
- per-request distributions.

Do not infer causality from a single aggregate correlation.

---

# 27. Required Figures

At minimum:

1. Block size vs quality
2. Block size vs BCT
3. Block size vs throughput
4. Block size vs NFE
5. Block size vs remote bytes
6. Block size vs token-expert pairs
7. Block size vs active experts
8. Block size vs rows/expert
9. Block size vs fanout
10. Block size vs rank-load CV
11. Block size vs dispatch/expert/combine fraction
12. Block size vs liveness waste
13. Block size vs HBM
14. Matched-M EP comparison
15. Per-B best mini size
16. Fixed-B quality-latency Pareto
17. Request-wise winning B distribution
18. Latency-only dynamic oracle vs best fixed
19. Quality-safe dynamic oracle vs best fixed
20. Dynamic schedule examples
21. Oracle B vs observable EP features
22. Semantic-only vs EP-only vs joint policy
23. Long-generation validation

---

# 28. Required Reports

Create:

```text
reports/00_environment_and_model_truth.md
reports/01_variable_block_decoder_correctness.md
reports/02_decoder_effort_controls.md
reports/03_static_block_sweep.md
reports/04_ep_regime_map.md
reports/05_minibatch_and_matchedM_controls.md
reports/06_quality_latency_frontier.md
reports/07_request_heterogeneity.md
reports/08_dynamic_schedule_oracle.md
reports/09_oracle_predictor_analysis.md
reports/10_live_dynamic_block_poc.md
reports/11_robustness_and_reproduction.md
reports/12_prior_art_and_novelty.md
reports/final_decision.md
```

Also create machine-readable tables:

```text
STATIC_BLOCK_SWEEP.csv
BLOCK_EP_METRICS.csv
BLOCK_QUALITY_METRICS.csv
DYNAMIC_SCHEDULE_ORACLE.csv
POLICY_COMPARISON.csv
ATTEMPT_LOG.csv
```

---

# 29. Prior-Art / Novelty Audit

Compare deeply with:

- AdaBlock-dLLM
- Adaptive Block Diffusion / variable-block diffusion work
- Block Diffusion
- LLaDA2.0 decoding
- Epoch
- dLLM serving systems
- semantic/confidence adaptive decoding
- distributed MoE runtime systems

Novelty cannot be:

> dynamically change block size.

The required distinction is:

> **block size is treated as a distributed sparse-MoE execution-control knob, and the trigger uses physical EP cost / liveness / communication / expert-shape signals rather than only semantic confidence.**

The strongest story would be:

```text
AR-ness changes EP bottleneck regime
+
different requests/blocks have different quality-safe system-optimal B
+
EP state predicts the choice
+
dynamic B beats every fixed-B quality-latency frontier
```

---

# 30. Promotion Gates

## Observation gate

Promising if:
- clear block-size-dependent EP regime transition exists;
- survives matched-M and steps controls.

## Dynamic-oracle gate

```text
<5%      NO-GO
5–8%     CHARACTERIZATION
8–12%    HOLD
12–20%   STRONG
>20%     VERY STRONG
```

Prefer continued engineering only if quality-safe oracle >=12%, unless the method is exceptionally simple.

## Live-method gate

Strong candidate if:
- beats best fixed-B baseline;
- preserves bounded quality within declared epsilon;
- reproduces across both GSM8K and HumanEval;
- gain survives controller overhead;
- EP/system features add value over semantic-only trigger.

---

# 31. Final Decision Labels

Choose one:

```text
SUBSTRATE-FAIL
NO-DYNAMIC-NEED
CHARACTERIZATION-SIGNAL
HOLD-CANDIDATE
STRONG-CANDIDATE
VERY-STRONG-CANDIDATE
```

Definitions:

### SUBSTRATE-FAIL
Variable-block semantics cannot be validated on the primary substrate.

### NO-DYNAMIC-NEED
One fixed B dominates all quality-safe settings across controls.

### CHARACTERIZATION-SIGNAL
Block size changes EP regime, but quality-safe dynamic oracle <8%.

### HOLD-CANDIDATE
Quality-safe dynamic oracle >=8%.

### STRONG-CANDIDATE
>=12% credible oracle plus a plausible EP-aware trigger.

### VERY-STRONG-CANDIDATE
Live policy produces >=10–15% request E2E or meaningful throughput improvement over the best fixed-B frontier with bounded quality and reproducibility.

---

# 32. Final Questions

The final report must answer:

1. Does block size materially change the EP physical workload?
2. Does smaller B create startup/tiny-GEMM domination?
3. Does larger B create communication/fanout/liveness-waste domination?
4. Is there a block-size-driven EP regime transition?
5. Are results explained by steps/NFE rather than B itself?
6. Do results survive matched-M control?
7. What is the best fixed B for each task/request?
8. Is the official B=32 actually Pareto-optimal?
9. Do different requests prefer different quality-safe B?
10. What is the maximum latency-only dynamic oracle?
11. What is the zero-quality-drop dynamic oracle?
12. What is the small-epsilon quality-constrained oracle?
13. Does the oracle require future information?
14. Which observable EP features predict the next best B?
15. Do EP features add predictive value beyond semantic confidence?
16. Can a simple live policy recover meaningful oracle gain?
17. Does the gain reproduce at longer generation lengths?
18. Does it reproduce on both tasks?
19. Is the final method clearly distinct from AdaBlock?
20. Is the opportunity large enough for a paper-level systems method?

---

# 33. Final Instruction

Do not seek a quick YES/NO.

This is a deep regime-discovery study.

When a setting fails:
- diagnose whether the cause is decoder semantics, runtime implementation, quality calibration, steps coupling, mini-batch confounding, or true absence of the phenomenon;
- repair and rerun when scientifically justified.

At the same time, remain strict about economics:

> a method is only worth promoting if block-size adaptation creates substantial **request-level quality-constrained E2E headroom** over the strongest fixed-block baseline.

The desired discovery is not simply:

```text
B changes latency.
```

It is:

```text
B / AR-ness changes the physical EP regime,
the system-optimal safe B varies over execution,
and EP-aware dynamic block sizing captures a large Pareto improvement.
```
