# dLLM MoE-EP Communication Amplification PoC
## Residual EP Bytes per Accepted Token after Optimistic Dead/Decoded-Token Elimination

**Target repository:** `/home/esjung/MLLM-EP-github`  
**Recommended server path:**  
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_moe_ep_comm_amplification_spec.md`

**Available GPUs:** `2, 3, 6, 7` only  
**Primary hardware mode:** TRUE physical EP4 if supported by the selected runtime  
**Primary research target:** low-request / batch-1 serving  
**PoC philosophy:** characterize the physical communication pathology first; do not implement a new decoding algorithm, scheduler, or migration scheme before the residual headroom is proven.

---

# 1. Research Question

This PoC tests whether **diffusion language model (dLLM) decoding creates a structural Expert-Parallel communication amplification that remains large even after already-decoded/dead positions are optimistically removed from EP traffic**.

The central question is:

> How many bytes of physical Expert-Parallel communication are paid for each finally accepted output token in a MoE dLLM, and how much of that amplification remains after removing the part already targeted by decoded/dead-token skipping systems such as Epoch?

The desired evidence chain is:

```text
Iterative denoising
    ->
many live token positions repeatedly traverse MoE
    ->
only a subset becomes newly accepted each iteration
    ->
repeated EP dispatch/combine traffic
    ->
high EP bytes per accepted token
    ->
large residual amplification even after optimistic dead-token skipping
```

The PoC should determine whether this is a sufficiently large and distinct systems problem to justify a new method.

---

# 2. Motivation and Prior Boundary

A standard AR MoE decoder approximately performs:

```text
one new decode token
    ->
router
    ->
EP dispatch
    ->
expert execution
    ->
EP combine
    ->
one new output token
```

A block-wise dLLM instead repeatedly refines a set of positions:

```text
many live positions
    ->
router
    ->
EP dispatch
    ->
expert execution
    ->
EP combine
    ->
only some positions are newly accepted
    ->
remaining positions are processed again later
```

This creates a mismatch between:

```text
EP work performed
```

and:

```text
new output tokens accepted
```

Two important prior works already cover adjacent parts of this problem:

- **TEAM (ICML 2026)** identifies that many experts are activated at each denoising step while only a small subset of tokens is accepted, and reduces expert activation using temporal/spatial consistency.
- **Epoch (arXiv 2609.09748, Sep. 2026)** removes substantial redundant MoE/EP work for already-decoded/dead positions and compacts the fresh token-expert worklist through the EP path.

Therefore, this PoC must **not** claim novelty from the observation that decoded positions are redundantly routed.

The key residual question is:

> After giving prior systems the strongest possible benefit by assuming that all already-accepted/dead positions can be removed from subsequent EP traffic, do live-but-not-yet-accepted positions still create a large communication amplification?

If the answer is no, close the direction.

---

# 3. Scope and Non-Goals

This PoC is a characterization/oracle study.

Do **not** spend substantial effort on:

- a new dLLM decoding algorithm;
- confidence-threshold tuning for speed;
- expert pruning;
- learned acceptance prediction;
- speculative denoising;
- real expert migration;
- EPLB redesign;
- a new CUDA kernel;
- a production dLLM serving scheduler;
- multi-node execution;
- quantization;
- full reproduction of Epoch.

The experiment should first establish whether the communication problem survives a strong optimistic baseline.

---

# 4. Hardware

Use only:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

No other GPU may be used.

Before any model launch, record:

```bash
nvidia-smi
nvidia-smi topo -m
```

Also record:

- physical GPU UUIDs;
- logical-to-physical rank mapping;
- inter-GPU topology;
- P2P/NVLink availability;
- free memory;
- any competing processes owned by the current user.

Do not kill or alter other users' processes.

---

# 5. Model and Runtime Selection

## 5.1 Required Property

The decisive experiment requires:

```text
MoE dLLM
+
real multi-GPU Expert Parallel execution
+
observable routing / denoising acceptance state
```

A single-GPU MoE trace projected offline onto four fake ranks is not sufficient for the headline result.

Offline projection may be used only for debugging or secondary structural checks.

---

## 5.2 Preferred Model Candidates

Use the first practical option that provides a correct physical EP path on GPUs 2,3,6,7.

Recommended order:

### Option A — Existing working MoE dLLM EP stack already available locally

Search the repository and server first.

Reuse any existing:

- LLaDA-MoE;
- LLaDA2;
- SDAR MoE;
- dInfer;
- TEAM-derived;
- Epoch-derived;
- vLLM/SGLang MoE-dLLM

environment before installing or rewriting a new stack.

### Option B — SDAR-30B-A3B

SDAR has an open 30B-A3B MoE dLLM family derived from Qwen3-30B-A3B and public inference code. This is attractive because it gives a large modern MoE dLLM and an architecture scale comparable to the existing Qwen3-30B-A3B EP experiments.

### Option C — LLaDA-MoE / LLaDA2 family through dInfer

dInfer publicly supports LLaDA-MoE and LLaDA2 families and uses vLLM/SGLang distributed backends.

A smaller model is acceptable for the first PoC if it is the only model with a trustworthy TRUE EP4 execution path.

---

## 5.3 Do Not Over-Invest in Runtime Bring-Up

If no trustworthy physical EP4 path can be made operational with bounded engineering effort, stop and report:

```text
RUNTIME_BLOCKED
```

Do not spend days rewriting a serving framework merely to reach Stage 1.

The purpose of this PoC is to test the research signal, not to build a new inference engine.

---

# 6. Primary Workload

Primary target:

```text
batch size = 1
```

because the intended research regime is low-request serving.

Use at least two workload categories if practical:

```text
reasoning: GSM8K or equivalent
coding: HumanEval / MBPP or equivalent
```

A cheap first pass may use:

```text
16-32 prompts per workload
```

with deterministic or fixed-seed decoding.

If runtime cost is high, use a smaller smoke cohort first and expand only if the result is near a GO/NO-GO boundary.

Do not select examples based on favorable routing or acceptance behavior.

---

# 7. Decoding Configuration

Use the selected model's official/recommended dLLM decoding configuration as the primary setting.

Record all of:

```text
block length
number of denoising steps or maximum steps
acceptance/confidence threshold
remasking policy
generation length
temperature
top-p
top-k
seed
```

Do not tune these parameters to artificially increase communication amplification.

If the runtime supports a cheap block-size sweep, evaluate at least two settings such as:

```text
small/default block
larger block
```

Examples may include 4/16/32/64 depending on the model family.

The exact supported values should follow the model/runtime.

---

# 8. Acceptance Semantics

For each denoising iteration `i`, explicitly identify:

```text
positions_live_before_i
positions_newly_accepted_i
positions_already_accepted_before_i
positions_still_live_after_i
```

Define:

```text
L_i = number of positions live before iteration i
A_i = number of positions newly accepted during iteration i
```

The acceptance yield is:

```text
acceptance_yield_i = A_i / L_i
```

Record zero-progress iterations if they occur:

```text
A_i = 0
```

Do not infer acceptance only from output length if the decoder exposes exact position-level state.

Use the runtime's real masking/remasking/commit state.

---

# 9. Required Router / EP Instrumentation

For each:

```text
request
block
denoising iteration
layer
token position
```

record at minimum:

```text
live/dead/accepted state
top-k expert IDs
routing weights, if cheaply available
source EP rank
destination EP rank
```

Also record physical or logical EP traffic information.

Preferred order:

1. measured dispatch/combine byte counts from the real backend, if available;
2. measured token counts plus exact backend payload shape;
3. validated logical-byte reconstruction.

Reuse existing physical EP instrumentation from previous project PoCs where possible.

In particular, if the repository already contains validated destination-deduplicated remote-byte accounting from prior EP work, reuse it rather than creating a conflicting definition.

---

# 10. Physical Traffic Accounting

Report both of the following whenever possible.

## 10.1 Remote Expert Paths

Count token-expert routing edges whose destination expert is on another EP rank.

This is useful structurally but is **not equivalent to actual communication bytes**.

---

## 10.2 Destination-Deduplicated Hidden-Row Bytes

If the backend sends one hidden row once per destination rank even when several experts on that rank are selected, account for this deduplication.

For each token/layer/source rank:

```text
number of unique remote destination ranks
    x hidden-state bytes
```

is a more realistic logical dispatch-byte metric than:

```text
number of remote expert assignments
    x hidden-state bytes
```

Use the backend's real behavior if known.

Apply the same accounting convention consistently to dispatch and combine.

---

# 11. Core Metric: EP Bytes per Accepted Token

For each denoising iteration:

```text
EP_bytes_i
newly_accepted_tokens_i = A_i
```

Define:

```text
EP_bytes_per_new_accept_i
    = EP_bytes_i / A_i
```

If `A_i = 0`, do not silently divide by zero.

Record:

```text
ZERO_ACCEPT_ITERATION
```

and separately report the full EP bytes paid during that zero-progress iteration.

At the request/block level:

```text
Total_EP_bytes_per_final_token
    =
    sum_i EP_bytes_i
    /
    final_generated_tokens
```

This is the primary communication-efficiency metric.

---

# 12. Three Required Communication Views

The decisive analysis must distinguish three traffic views.

## View A — Actual / Vanilla Runtime Traffic

This is the communication traffic actually executed by the selected dLLM runtime.

Call it:

```text
B_actual
```

This may include:

- positions already accepted in prior iterations;
- still-live positions;
- newly accepted positions;
- any refresh-required positions required by the runtime.

This is a measured fact whenever backend instrumentation is available.

---

## View B — Optimistic Live-Only Traffic

Construct a structural accounting baseline that gives decoded/dead-token skipping the strongest possible benefit.

For iteration `i`, count only token positions that were **not accepted before iteration i**.

In other words:

```text
already accepted before i
    -> zero EP traffic in this optimistic accounting

still live before i
    -> retain their observed routing traffic
```

Call the resulting total:

```text
B_live_only
```

This is intentionally optimistic.

It is **not** automatically a valid executable implementation because some accepted/refresh-required state may still be needed for exact semantics.

Its purpose is to answer:

> Even if already-accepted positions were perfectly removed from EP traffic, how much repeated communication remains because live positions require multiple denoising iterations?

If the residual is large even under this optimistic bound, that is strong evidence.

---

## View C — Acceptance-Route One-Shot Reference

Construct an internal one-pass reference using the same dLLM's own observed routing.

For each final output position:

1. find the iteration in which that position is first accepted;
2. take that position's routing at that acceptance iteration;
3. count its EP traffic exactly once.

Sum this over all final output positions.

Call it:

```text
B_one_shot
```

This is not an executable model.

It is a structural lower/reference point answering:

> What would EP traffic look like if every output position paid for its observed acceptance-time route only once?

This avoids comparing different model architectures.

---

# 13. Primary Amplification Factors

## 13.1 Raw Amplification

```text
Raw_Amp
    = B_actual / B_one_shot
```

This includes all runtime repetition.

## 13.2 Residual Live-Only Amplification

```text
Residual_Amp
    = B_live_only / B_one_shot
```

This is the decisive metric.

It estimates how much communication amplification remains **after optimistically deleting all prior-accepted/dead-position EP traffic**.

Interpretation:

```text
Residual_Amp = 1.0x
    -> dead-token removal nearly explains the whole problem

Residual_Amp = 2.0x
    -> even after optimistic skipping, each final token still pays roughly twice the one-shot EP traffic

Residual_Amp = 4.0x
    -> large repeated communication remains on live-but-not-yet-accepted positions
```

Do not call all residual bytes "waste."

Live token computation may be semantically necessary because those states can affect other positions.

Use terms such as:

```text
residual communication amplification
acceptance-inefficient traffic
repeated live-token EP traffic
```

rather than claiming it is safely removable.

---

# 14. Iteration-Level Diagnostic Metrics

For every denoising iteration report:

```text
live positions before iteration
newly accepted positions
acceptance yield
cumulative accepted positions
routed token count
expert assignments
unique experts
remote expert paths
unique remote destination ranks
dispatch bytes
combine bytes
EP bytes / newly accepted token
max-rank routed load
rank max/mean
rank CV
iteration latency
dispatch latency
expert latency
combine latency
```

Also report:

```text
fraction of current EP bytes associated with positions
that remain unaccepted after the current iteration
```

Call this:

```text
nonaccepted_live_traffic_fraction
```

This is diagnostic only.

Do not claim these bytes can be skipped without changing semantics.

---

# 15. Secondary Diagnostic: EP Rank-Sparsity Collapse

Because many token positions are routed in parallel, token-level sparsity may become dense at the EP-rank level.

Measure per iteration:

```text
active destination ranks
fraction of possible source->destination rank edges used
remote-rank fanout per token
remote-rank fanout per block
```

For EP4, report how often each iteration effectively touches:

```text
1 / 2 / 3 / all 4 ranks
```

This is not the primary hypothesis, but it may explain why dLLM communication is expensive.

---

# 16. Secondary Diagnostic: Temporal Rank Imbalance

For each iteration record:

```text
hottest EP rank
max-rank token load
max/mean rank load
rank CV
```

Across iterations calculate:

```text
hottest-rank switch rate
variance of rank imbalance
correlation with acceptance yield
correlation with live-token count
```

This may reveal a second dLLM-specific EP pathology.

Do not promote it to the main contribution unless the signal is clearly stronger than the communication-amplification result.

---

# 17. Measured Timing

If the runtime allows trustworthy timing, record:

```text
router
dispatch
expert execution
combine
attention / non-MoE
total denoising iteration
request E2E
```

Use CUDA events where appropriate.

Do not use heavily instrumented wall-clock numbers as headline latency without validating instrumentation overhead.

Report:

```text
dispatch + combine share of total denoising latency
MoE path share of total denoising latency
```

This is important because a large byte amplification is less useful as a research target if physical communication is only a negligible fraction of latency.

---

# 18. Optional AR Contextual Baseline

A matched AR reference may be collected using the already-working:

```text
Qwen3-30B-A3B TRUE EP4
```

on GPUs 2,3,6,7.

Use:

```text
batch size = 1
same or similar generated output length
same physical EP4 topology
```

Report:

```text
remote EP bytes / generated token
expert assignments / generated token
active destination ranks / generated token
dispatch+combine latency / generated token
```

However:

**Do not use cross-model AR-vs-dLLM bytes as the decisive gate**, because hidden size, routing behavior, expert topology, and implementation may differ.

The decisive baseline is the same-model internal:

```text
B_live_only / B_one_shot
```

amplification.

The AR run is contextual evidence only.

---

# 19. Small Robustness Sweep

If the primary run succeeds, perform a bounded sweep over one or two factors.

Preferred:

```text
block size
```

because block size directly changes the number of simultaneously refined positions.

Example, only if supported:

```text
default/small
medium
large
```

Optionally evaluate:

```text
batch 1 primary
batch 4 secondary
```

Do not turn this PoC into a large benchmark grid.

The purpose is to verify that the result is not a single-configuration artifact.

---

# 20. Quality / Correctness Sanity

Because this PoC should not modify the model semantics, output quality should match the uninstrumented reference path.

For a small subset:

- compare generated token IDs or deterministic output when possible;
- confirm the acceptance schedule is unchanged by tracing;
- verify no router hook changes expert selection;
- verify measured traffic counters match routed tensor shapes.

Do not interpret performance data if instrumentation changes decoding behavior.

---

# 21. Required Statistical Treatment

Use request-level or problem-level independent units.

Do not treat:

```text
layers
iterations
token positions
```

as independent samples.

For headline metrics report:

```text
mean
median
p25
p75
95% bootstrap CI
```

for:

```text
Raw_Amp
Residual_Amp
Total_EP_bytes_per_final_token
dispatch+combine latency share
acceptance yield
```

Bootstrap over requests/problems.

---

# 22. GO / NO-GO Gates

These thresholds are heuristic discovery gates.

## Gate 0 — Physical EP Validity

PASS only if:

- experts are physically sharded across GPUs 2,3,6,7;
- real EP communication occurs;
- routing-to-rank ownership is verified;
- traffic accounting passes synthetic sanity checks.

If not:

```text
RUNTIME_BLOCKED
```

rather than a research conclusion.

---

## Gate 1 — Residual Live-Only Amplification

Primary gate:

```text
Residual_Amp
    = B_live_only / B_one_shot
```

Interpretation:

```text
< 1.5x
    -> NO-GO

1.5x - 2.0x
    -> CONDITIONAL

>= 2.0x
    -> meaningful GO signal

>= 3.0x
    -> strong structural signal
```

Use request-level bootstrap CIs.

A strong result should not depend on one or two outlier prompts.

---

## Gate 2 — Physical Communication Relevance

A communication-specific research direction is weak if communication is not materially exposed.

Preferred GO signal:

```text
dispatch + combine >= ~15% of measured denoising E2E
```

or a comparably strong demonstrated communication bottleneck.

Interpretation:

```text
< 10% consistently
    -> communication-specific direction likely NO-GO

10-15%
    -> CONDITIONAL

>= 15%
    -> useful signal
```

If timing instrumentation is not trustworthy, mark this gate unresolved rather than fabricating a latency conclusion.

---

## Gate 3 — Robustness

The residual amplification should appear in:

```text
at least two workloads
```

or:

```text
at least two decoding/block configurations
```

If the effect exists only in one unusual setting, mark CONDITIONAL.

---

## Gate 4 — Prior-Removal Test

The direction is worth continuing only if:

```text
Raw_Amp is high
AND
Residual_Amp remains materially high
```

A result such as:

```text
Raw_Amp = 4.5x
Residual_Amp = 1.2x
```

means prior-style decoded/dead-token elimination explains almost everything.

Verdict:

```text
NO-GO
```

A result such as:

```text
Raw_Amp = 6.0x
Residual_Amp = 3.2x
```

means a large unexplained residual remains.

Verdict:

```text
GO
```

---

# 23. Important Interpretation Rule

Do not make the following invalid claim:

```text
"live-but-unaccepted token traffic is useless and can simply be skipped"
```

That is not established.

Those token states may be required to preserve exact denoising semantics and influence other positions.

The PoC only establishes whether there is a **large residual systems cost associated with repeatedly processing live positions that make limited acceptance progress per iteration**.

If GO, the next research question becomes:

> Which part of this live-token EP work can be avoided, approximated, deferred, compressed, merged, or scheduled differently without harming dLLM semantics or output quality?

That question belongs to the next phase, not this PoC.

---

# 24. Recommended Execution Stages

## Stage 0 — Runtime Bring-Up

- inspect repository/server for existing dLLM assets;
- identify the cheapest valid physical EP4 stack;
- run one tiny prompt;
- verify denoising iterations and acceptance state;
- verify all four GPUs participate in EP.

Stop if runtime bring-up becomes a major implementation project.

## Stage 1 — Instrumentation Sanity

For one or two prompts:

- capture acceptance positions per iteration;
- capture expert IDs;
- verify expert ownership;
- calculate logical remote paths/bytes;
- compare with backend counters where available;
- validate no tracing-induced output change.

## Stage 2 — Primary Batch-1 Cohort

Run the frozen prompt cohort.

Collect:

```text
B_actual
B_live_only
B_one_shot
Raw_Amp
Residual_Amp
per-iteration acceptance/traffic/timing
```

Do not inspect favorable prompts and then change the cohort.

## Stage 3 — Small Robustness Sweep

Run one bounded sweep over:

```text
block size
```

or a second workload.

## Stage 4 — Optional AR Context

Reuse the existing Qwen3-30B-A3B TRUE EP4 path only after the decisive same-model dLLM analysis is complete.

---

# 25. Required Outputs

Recommended directory:

```text
poc_dllm_ep_comm/
├── scripts/
│   ├── inspect_runtime.py
│   ├── run_dllm_ep4.py
│   ├── trace_acceptance.py
│   ├── trace_ep.py
│   ├── analyze_amplification.py
│   ├── analyze_rank_diagnostics.py
│   └── run_ar_context.py
│
├── results/
│   ├── environment.json
│   ├── raw/
│   ├── iteration_metrics.csv
│   ├── request_metrics.csv
│   ├── amplification_summary.json
│   ├── rank_diagnostics.csv
│   ├── timing_summary.json
│   └── validity.json
│
├── plots/
│   ├── live_accept_bytes_by_iteration.png
│   ├── bytes_per_accept_by_iteration.png
│   ├── raw_vs_live_only_vs_one_shot.png
│   ├── residual_amplification_distribution.png
│   ├── rank_density_by_iteration.png
│   └── rank_imbalance_by_iteration.png
│
└── report.md
```

Do not overwrite unrelated PoCs.

---

# 26. Required Plots

## Plot 1 — Live Tokens, Accepted Tokens, EP Bytes vs Iteration

X-axis:

```text
denoising iteration
```

Show:

```text
live tokens
newly accepted tokens
EP bytes
```

This should make the core pathology visually obvious or falsify it.

## Plot 2 — EP Bytes per Newly Accepted Token

For each iteration:

```text
EP bytes / newly accepted token
```

Mark zero-accept iterations explicitly.

## Plot 3 — Raw vs Live-Only vs One-Shot

Per request:

```text
B_actual
B_live_only
B_one_shot
```

Normalize to:

```text
B_one_shot = 1.0x
```

This is the most important figure.

Desired example of a strong signal:

```text
One-shot      1.0x
Live-only     3.0x
Actual        5.5x
```

## Plot 4 — Residual Amplification Distribution

Show request-level:

```text
Residual_Amp = B_live_only / B_one_shot
```

with median and CI.

## Plot 5 — Secondary EP Diagnostics

Show:

```text
active destination ranks
rank CV
hottest rank
```

over denoising iterations.

---

# 27. Required Final Report Questions

The final `report.md` must directly answer:

1. Which MoE dLLM and runtime were used?
2. Was TRUE physical EP4 verified on GPUs 2,3,6,7?
3. What is the acceptance pattern across denoising iterations?
4. What is measured EP communication per final output token?
5. What is `Raw_Amp`?
6. What is `Residual_Amp` after optimistic prior-accepted/dead-token elimination?
7. How much of total amplification is explained by already-accepted positions?
8. How much residual communication is associated with repeatedly processed live positions?
9. Is the residual robust across prompts/workloads/configurations?
10. Is dispatch+combine a material measured latency component?
11. Does EP-rank communication become dense across a block?
12. Is there meaningful temporal rank imbalance?
13. What is the final verdict: GO / CONDITIONAL / NO-GO / RUNTIME_BLOCKED?
14. If GO, what is the **single smallest next experiment** needed before designing a method?

The report must clearly distinguish:

```text
measured physical traffic
validated logical-byte accounting
optimistic structural baseline
non-executable lower/reference bound
measured latency
modeled/speculative interpretation
```

---

# 28. Decision Examples

## Strong GO

```text
Raw_Amp        = 5.8x
Residual_Amp   = 3.1x
dispatch+combine share = 22%
effect robust across reasoning and coding
```

Interpretation:

> Existing decoded/dead-token elimination does not explain the majority of dLLM EP communication amplification. Repeated live-token processing remains a large physical EP cost.

## NO-GO because Epoch-like skipping explains the problem

```text
Raw_Amp        = 4.2x
Residual_Amp   = 1.2x
```

Interpretation:

> Most amplification is attributable to already-decoded/dead positions. There is little residual problem beyond the prior-art target.

## NO-GO because communication is not critical

```text
Residual_Amp   = 2.5x
dispatch+combine share = 5%
```

Interpretation:

> A structural byte amplification exists, but physical communication is not a sufficiently important latency target in the tested low-request regime.

---

# 29. Evidence Integrity

Before finalizing:

- save exact model revision/hash;
- save runtime/library versions;
- save decoding parameters;
- save GPU topology;
- save prompt IDs;
- preserve raw traces;
- ensure no future acceptance state is used in measured runtime decisions;
- label all oracle/offline accounting as nondeployable;
- run consistency tests for token counts, expert ownership, and byte arithmetic.

Do not silently repair or overwrite failed runs.

---

# 30. Git / Repository Hygiene

The existing repository may contain unrelated dirty or staged work.

Requirements:

- preserve all pre-existing dirty/staged changes;
- create new PoC files in a separate directory;
- do not reset or clean unrelated files;
- commit only files created/modified for this PoC;
- push the completed PoC branch after validation;
- report branch name and final commit SHA.

Suggested branch name:

```text
flashvep/dllm-ep-comm-amplification-poc
```

---

# 31. Final Success Criterion

The direction should continue only if the data supports:

```text
dLLM iterative acceptance
    ->
large physical EP traffic per accepted token
    ->
large residual even after optimistic dead-token removal
    ->
communication is materially exposed in latency
```

The most important number is:

```text
Residual_Amp
    = optimistic live-only EP bytes
      /
      acceptance-route one-shot EP bytes
```

If this residual is not large, do not invent a method. Close the direction.
