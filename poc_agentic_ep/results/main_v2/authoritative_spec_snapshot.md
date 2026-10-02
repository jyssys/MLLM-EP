# Agentic MoE-EP PoC Specification
## Cross-Turn Routing Locality and Proactive Expert-Parallel Preparation

**Target repository:** `/home/esjung/MLLM-EP-github`  
**Recommended server path:**  
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/agentic_moe_ep_poc_spec.md`

**Available GPUs:** `2, 3, 6, 7`  
**Primary model:** Qwen3-30B-A3B  
**Primary execution mode:** TRUE EP4  
**PoC philosophy:** characterization and oracle analysis first; do not implement a full scheduling or expert-migration method before the signal is validated.

---

# 1. Research Goal

This PoC evaluates whether **agentic inference creates a new and exploitable optimization opportunity for Mixture-of-Experts Expert Parallelism (MoE EP)**.

The central hypothesis is:

> Consecutive model invocations within the same agent trajectory may exhibit predictable expert-routing locality. If this locality exists, the tool-execution interval between two model invocations may provide an off-critical-path window in which the system can proactively prepare expert placement, replication, or related EP state for the next model turn.

This experiment should answer three questions in order:

1. **Cross-turn locality:**  
   Are consecutive model turns in the same agent trajectory significantly more similar in expert/rank routing than appropriate controls?

2. **Actionable EP headroom:**  
   If the next turn's routing were known or partially predicted, is there enough potential reduction in EP communication and/or rank imbalance to matter?

3. **Preparation feasibility:**  
   Can the required expert preparation plausibly fit inside realistic tool-call gaps?

If any fundamental condition fails, stop and report **NO-GO** rather than building a more complicated method.

---

# 2. Why Agentic Serving Is a Distinct Workload

A conventional serving request is approximately:

```text
Prompt
  -> Prefill
  -> Decode
  -> Done
```

An agent trajectory repeatedly invokes the model:

```text
Model Turn t
  -> Tool Call
  -> Tool Execution
  -> Tool Result
  -> Model Turn t+1
  -> Tool Call
  -> ...
```

Therefore, one user task becomes a sequence of causally related model invocations.

Important characteristics include:

- repeated model calls within the same semantic task;
- an accumulated conversation/context;
- long shared prefixes;
- relatively small newly appended suffixes in later turns;
- tool execution gaps between model invocations;
- potentially persistent task/domain semantics across turns.

These properties may create routing reuse opportunities that do not exist for independent one-shot requests.

---

# 3. Critical Validity Rule: Do Not Measure Trivial Shared-Prefix Similarity

This is the most important rule in the entire PoC.

Agent requests often reuse most of the previous prompt. If the full accumulated prompt is rerun and all routing decisions are compared, adjacent turns will appear highly similar simply because the input contains many identical tokens.

That is **not** evidence of useful agent-specific routing locality.

For each turn, conceptually separate:

```text
cached_prefix_t
new_prefill_suffix_t
decode_t
```

The primary locality analysis must use only:

```text
new_prefill_suffix_t
decode_t
```

and must exclude:

```text
cached_prefix_t
```

from the main similarity metrics.

If exact prefix caching is not available in the execution path, reconstruct the longest common token prefix offline and exclude those tokens from the routing analysis.

All reports must explicitly state that shared-prefix tokens were excluded.

---

# 4. Hardware Configuration

Use only the following GPUs:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

Target setup:

```text
GPU count: 4
Expert Parallelism: TRUE EP4
Tensor Parallelism: TP1 if compatible with the existing implementation
Data Parallelism: DP1
Precision: existing BF16/FP16 configuration
```

Do not introduce quantization unless absolutely required by the existing environment. Precision reduction is outside the scope of this PoC.

Before running the experiment, save:

```bash
nvidia-smi
nvidia-smi topo -m
```

Also benchmark the effective P2P bandwidth between GPUs 2, 3, 6, and 7 if an existing benchmark is available or can be implemented cheaply.

Do not use any other GPU.

---

# 5. Model

Preferred model:

```text
Qwen3-30B-A3B
```

Use the repository's existing model-loading and TRUE-EP4 path whenever possible.

Do not reimplement model execution if the repository already contains:

- TRUE EP4 launch code;
- DeepEP integration;
- router expert-ID tracing;
- routed-token counts;
- rank-load tracing;
- per-layer dispatch / expert / combine timing.

Reuse existing instrumentation before adding new hooks.

---

# 6. Workload

## 6.1 Primary Workload Requirement

Use **real semantic multi-turn agent trajectories** whose model inputs or equivalent action/observation history are available.

Preferred examples:

- SWE-agent trajectories;
- SWE-smith / SWE-agent generated trajectories;
- other coding-agent traces that preserve actual queries, actions, observations, and tool results.

The goal is not to benchmark agent quality. The goal is to obtain realistic sequences of semantically related model invocations.

Initial target:

```text
8-16 trajectories
>= 8 usable model turns per trajectory
target >= 100 adjacent-turn pairs
```

Minimum cheap-PoC target:

```text
8 trajectories
>= 60 valid adjacent-turn pairs
```

If obtaining large traces becomes expensive, start with the minimum target and expand only if confidence intervals are ambiguous.

---

# 7. Replay Strategy

Prefer deterministic replay rather than launching a fully autonomous live agent.

For each recorded agent trajectory:

1. reconstruct each model invocation;
2. identify the common prefix with the previous invocation;
3. separate cached-prefix tokens from newly added input tokens;
4. execute the model while tracing router decisions;
5. optionally teacher-force recorded assistant output tokens to trace decode routing;
6. store only the information required for systems analysis.

If teacher-forced decode replay is difficult, use this priority:

```text
Stage 1a: new-prefill-suffix routing
Stage 1b: decode routing
```

Do not delay the entire PoC just to obtain decode traces.

---

# 8. Required Routing Instrumentation

For each:

```text
trajectory
turn
layer
token
```

record at minimum:

```text
top-k expert IDs
routing weights, if cheaply available
source EP rank
destination EP rank
token category:
    new_prefill
    decode
```

Do not save large hidden states unless necessary.

For each turn `t` and layer `l`, construct an expert-load vector:

```text
E[t,l] = normalized assignment counts over all experts
```

and an EP-rank-load vector:

```text
R[t,l] = normalized assignment counts over the four EP ranks
```

Also construct turn-level aggregates.

Store raw counts as well as normalized distributions.

---

# 9. Experiment A: Cross-Turn Expert Locality

For every adjacent pair:

```text
Turn t -> Turn t+1
```

measure expert-level similarity using only newly computed tokens.

Required metrics:

```text
Cosine similarity
Jensen-Shannon divergence
Top-M hot-expert overlap
Weighted next-turn assignment recall@M
```

Recommended:

```text
M in {4, 8, 16, 32}
```

The most useful metric is weighted next-turn coverage:

> Among routed assignments in Turn t+1, what fraction is covered by experts predicted using Turn t?

Report both:

```text
per-layer metrics
turn-aggregated metrics
```

Keep new-prefill and decode results separate whenever both are available.

---

# 10. Controls: Is the Signal Really Agent-Specific?

At least three controls are mandatory.

## 10.1 Same-Trajectory Non-Adjacent Control

For each adjacent pair, compare against a non-adjacent turn sampled from the same trajectory.

Example:

```text
positive:
Turn 5 -> Turn 6

control:
Turn 5 -> Turn 10
```

This tests whether immediate temporal/semantic continuity matters.

---

## 10.2 Cross-Trajectory Length-Matched Control

Match turns from different agent trajectories with similar:

```text
new-prefill token count
decode token count, if applicable
```

This tests whether the apparent locality is merely a generic coding-domain effect.

---

## 10.3 Global Expert Popularity Baseline

From calibration trajectories, estimate globally popular experts.

Compare:

```text
previous-turn predictor
vs.
global-popularity predictor
```

This control is essential.

If the previous-turn policy does not materially outperform global expert popularity, the agent-specific routing story is weak.

---

# 11. Experiment B: EP-Rank Locality

Expert similarity alone is insufficient for an EP systems contribution.

Map routed experts through the actual TRUE-EP4 placement and compute:

```text
rank-load cosine similarity
rank-load L1 distance
max-rank routed assignments
max/mean rank load
coefficient of variation across ranks
identity of the hottest rank
```

Compare adjacent turns against all controls.

The key question is:

> Does cross-turn locality survive after expert assignments are projected onto the actual four-GPU EP topology?

A useful signal should affect at least one EP-relevant quantity such as:

- remote assignment volume;
- rank imbalance;
- hottest-rank persistence;
- estimated dispatch/combine traffic;
- critical-rank expert work.

---

# 12. Experiment C: Trace-Based Oracle EP Preparation

Do **not** implement real expert migration or a new scheduler yet.

Run an offline simulation using the collected routing traces.

Compare four policies.

## Policy A: Static EP4

```text
current fixed placement
no preparation
```

This is the baseline.

## Policy B: Global Popularity

Select additional prepared/replicated experts using only global calibration popularity.

This is the non-agent-aware baseline.

## Policy C: Previous-Turn Prediction

Use routing from Turn t to decide which experts should be prepared for Turn t+1.

This is the primary causal agent-aware policy.

## Policy D: Oracle Next-Turn

Use actual Turn t+1 routing.

This is an upper bound.

---

# 13. Preparation Model

The PoC should primarily model **temporary replication / preparation**, because fully remapping all experts between turns may be unnecessarily expensive for a first oracle study.

Possible conceptual operation:

```text
During tool gap:
    prepare a small set of likely-hot experts
    on additional EP ranks

Next model turn:
    assignments that hit a local prepared replica
    avoid part of the remote EP cost
```

This is only a simulation at this stage.

Do not commit to a final mechanism yet.

---

# 14. Preparation Budget Sweep

Read the actual expert parameter size from the deployed model.

Report approximate memory per expert.

Sweep realistic budgets in both forms.

### Expert-count budget

```text
+1 expert / rank
+2 experts / rank
+4 experts / rank
```

Add more points only if memory permits.

### Memory budget

Recommended approximate sweep:

```text
0.5 GB
1 GB
2 GB
4 GB
```

If these values are inappropriate for the real model/GPU memory, replace them with equivalent feasible points and explain why.

Never simulate an impossible memory configuration without labeling it clearly as hypothetical.

---

# 15. Oracle Objectives

For each policy and preparation budget, estimate changes in:

```text
remote routed assignments
dispatch communication volume
combine communication volume, if applicable
max-rank routed-token load
rank imbalance
estimated critical EP work
```

If measured per-layer timing exists, additionally estimate latency using:

```text
measured dispatch time
measured expert execution time
measured combine time
```

Do not convert routed-assignment reduction directly into claimed E2E speedup without a timing model.

Always separate:

```text
routing/communication headroom
```

from:

```text
modeled latency headroom
```

and from:

```text
measured latency
```

---

# 16. Predictor Effectiveness

Calculate:

```text
Recovered Oracle Headroom
    = Gain(previous-turn policy)
      / Gain(oracle-next-turn policy)
```

Example:

```text
Oracle EP-work reduction: 20%
Previous-turn reduction: 12%

Recovered Oracle Headroom = 60%
```

This tells us whether a simple causal signal is already actionable.

A strong oracle with a weak previous-turn predictor suggests that prediction itself would become a major research problem.

A weak oracle means the entire direction should be closed.

---

# 17. Experiment D: Tool-Gap Feasibility

The agentic opportunity matters only if preparation can happen outside the next turn's critical path.

Measure or estimate:

```text
expert bytes to prepare
effective P2P bandwidth
preparation latency
```

Evaluate the following hypothetical tool-gap windows:

```text
50 ms
100 ms
250 ms
500 ms
1000 ms
```

For each preparation policy/budget, report whether preparation is:

```text
fully hidden
partially hidden
not hidden
```

Also report:

```text
required minimum tool-gap duration
```

for each useful configuration.

Do not require a full live tool-calling framework for this first PoC.

---

# 18. Statistics

Turns from the same trajectory are correlated.

Therefore, use **trajectory-level bootstrap confidence intervals** rather than treating all turn pairs as independent samples.

For important metrics report:

```text
mean
median
p25
p75
95% trajectory-bootstrap CI
```

Required paired comparisons:

```text
adjacent - same-trajectory shuffled
adjacent - cross-trajectory matched
previous-turn - global popularity
oracle - static
previous-turn - static
```

Also include per-trajectory results to ensure that a few long trajectories are not dominating the aggregate.

---

# 19. GO / NO-GO Gates

The exact thresholds are heuristic, but use them consistently.

## Gate 1: Cross-Turn Locality

Desired GO signal:

```text
Adjacent-turn routing similarity is consistently
better than shuffled/matched controls,
and the trajectory-bootstrap 95% CI excludes zero.
```

Useful practical targets:

```text
expert-distribution cosine gap >= +0.05
and/or
weighted next-turn expert coverage improvement >= +5 percentage points
```

A different metric may substitute if it demonstrates a clearly stronger and more interpretable effect.

---

## Gate 2: Agent-Aware Prediction Must Beat Global Popularity

Desired:

```text
previous-turn predictor
> global-popularity predictor
```

with approximately:

```text
>= 5 percentage-point useful coverage improvement
```

or a comparable improvement in an EP-relevant metric.

If:

```text
previous-turn ~= global popularity
```

then the agent-specific motivation is weak.

---

## Gate 3: Oracle EP Headroom

Under a feasible memory/preparation budget:

```text
oracle EP-relevant reduction >= ~15%
```

is a strong GO signal.

Interpretation:

```text
< 10% oracle headroom:
    NO-GO

10-15%:
    CONDITIONAL / inspect bottleneck composition

>= 15%:
    worth continuing
```

Use measured or modeled critical EP work where possible rather than assignment counts alone.

---

## Gate 4: Causal Predictor Usefulness

Desired:

```text
previous-turn policy >= ~10% EP-work reduction
```

or:

```text
Recovered Oracle Headroom >= 50%
```

The purpose is not to require an optimized predictor yet, but to show that simple cross-turn information already captures a meaningful portion of the oracle opportunity.

---

## Gate 5: Preparation Feasibility

At least one useful preparation configuration should plausibly fit inside realistic tool gaps such as:

```text
100 ms
250 ms
500 ms
```

If meaningful preparation requires multi-second gaps or unrealistic memory, the serving opportunity is substantially weaker.

---

# 20. Overall Decision Logic

## GO

Continue only if all of the following are true:

```text
1. adjacent-turn locality exists;
2. it survives shared-prefix removal;
3. it beats same-domain/global controls;
4. meaningful EP-level oracle headroom exists;
5. a simple previous-turn predictor captures useful headroom;
6. preparation can plausibly be hidden in realistic tool gaps.
```

## CONDITIONAL

Use this only when there is one clearly identifiable missing piece.

Examples:

```text
strong locality + strong oracle headroom
but predictor is weak

or

strong routing headroom
but timing instrumentation is insufficient
```

Specify exactly one next diagnostic experiment.

## NO-GO

Close the direction if any fundamental condition fails, especially:

```text
adjacent ~= shuffled
previous-turn ~= global popularity
oracle EP headroom < 10%
preparation cost is structurally too large
```

Do not rescue a weak result by immediately introducing a complicated learned predictor, scheduler, or migration protocol.

---

# 21. Recommended Execution Stages

## Stage 0 — Environment Check

- confirm GPUs 2,3,6,7;
- confirm TRUE EP4 model launch;
- inspect topology;
- verify routing trace correctness.

## Stage 1 — Cheap Locality Probe

Use a small subset:

```text
8 trajectories
~60+ adjacent pairs
```

Collect only new-prefill routing first if necessary.

Decision:

```text
No clear adjacent-turn signal -> stop.
```

## Stage 2 — Controls and EP Projection

Add:

- same-trajectory shuffled controls;
- cross-trajectory matched controls;
- global popularity baseline;
- EP-rank-level metrics.

Decision:

```text
Signal disappears after controls -> stop.
```

## Stage 3 — Oracle Preparation Simulation

Evaluate:

```text
Static
Global
Previous-turn
Oracle
```

over budget sweeps.

Decision:

```text
Oracle < 10% -> stop.
```

## Stage 4 — Tool-Gap Feasibility

Measure P2P transfer cost and estimate whether useful preparation can be hidden.

Only after Stage 4 passes should a real runtime mechanism be considered.

---

# 22. Required Outputs

Recommended experiment directory:

```text
poc_agentic_ep/
├── scripts/
│   ├── collect_or_reconstruct_trajectories.py
│   ├── replay_ep4.py
│   ├── analyze_locality.py
│   ├── analyze_ep_rank.py
│   ├── simulate_preparation.py
│   └── benchmark_p2p.py
│
├── results/
│   ├── raw_routing/
│   ├── turn_metrics.csv
│   ├── pair_metrics.csv
│   ├── layer_metrics.csv
│   ├── ep_rank_metrics.csv
│   ├── preparation_budget.csv
│   └── summary.json
│
├── plots/
│   ├── adjacent_vs_controls.png
│   ├── next_turn_expert_coverage.png
│   ├── ep_rank_locality.png
│   ├── oracle_vs_previous_turn.png
│   └── preparation_gap_feasibility.png
│
└── report.md
```

Do not overwrite unrelated previous PoC results.

---

# 23. Required Plots

At minimum produce:

## Plot 1 — Adjacent vs Controls

Compare routing similarity distributions:

```text
Adjacent
Same-trajectory shuffled
Cross-trajectory matched
```

## Plot 2 — Next-Turn Expert Coverage

Compare:

```text
Random / shuffled
Global popularity
Previous turn
Oracle next turn
```

for several preparation budgets.

## Plot 3 — EP-Rank Locality

Show whether expert-level locality translates into actual four-rank load/communication locality.

## Plot 4 — Oracle Headroom

X-axis:

```text
additional memory / expert-replica budget
```

Y-axis:

```text
estimated EP-work reduction
```

Lines:

```text
Global
Previous-turn
Oracle
```

## Plot 5 — Preparation Feasibility

Show preparation latency versus:

```text
50
100
250
500
1000 ms
```

tool-gap windows.

---

# 24. Required Final Report

The final `report.md` must answer the following questions directly.

1. **Do adjacent turns within the same agent trajectory exhibit routing locality?**
2. **Does the signal remain after removing shared-prefix tokens?**
3. **Is it stronger than non-adjacent turns from the same trajectory?**
4. **Is it stronger than unrelated coding-agent turns with similar lengths?**
5. **Does previous-turn routing beat global expert popularity?**
6. **Does expert-level locality survive after projection onto the real EP4 placement?**
7. **What is the oracle EP optimization headroom?**
8. **How much of that oracle does a previous-turn-only causal policy recover?**
9. **What expert-replica memory budget is required?**
10. **Can the preparation cost fit inside plausible tool-call gaps?**
11. **What is the final verdict: GO, CONDITIONAL, or NO-GO?**
12. **If GO, what is the single smallest next experiment before implementing a real runtime mechanism?**

The report must distinguish:

```text
measured fact
trace-based oracle result
modeled latency estimate
speculation
```

Do not present a simulated communication reduction as a measured E2E speedup.

---

# 25. Scope Restrictions

For this PoC, do not spend substantial time on:

- production-quality autonomous agent orchestration;
- learned next-turn predictors;
- real dynamic expert migration;
- complex EPLB implementation;
- multi-node execution;
- accuracy optimization;
- quantization;
- new CUDA kernels;
- extensive scheduler refactoring.

The experiment is designed to answer whether the research opportunity exists **before** implementation investment.

---

# 26. Final Success Criterion

The desired evidence chain is:

```text
Agent trajectory continuity
        ->
non-trivial cross-turn expert locality
        ->
EP-rank / communication locality
        ->
meaningful oracle preparation headroom
        ->
simple causal predictor captures a useful fraction
        ->
preparation can be hidden during tool execution
```

If this chain is supported, the next research question becomes:

> How should a MoE EP serving system proactively and cheaply prepare experts during agent tool-execution gaps without introducing excessive memory overhead or harmful reconfiguration?

If the chain breaks at any stage, report NO-GO and stop the direction.
