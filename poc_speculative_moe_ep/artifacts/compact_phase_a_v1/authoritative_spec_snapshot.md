# PoC Specification: Speculative Decoding × Physical MoE Expert Parallelism

**Authoritative server path**  
`/home/esjung/MLLM-EP-github/poc_flashvep/reports/speculative_decoding_moe_ep_communication_discovery_spec.md`

**Project:** Generic MoE Expert-Parallel Inference  
**PoC type:** Discovery-first; trace + lightweight TRUE EP4 physical validation  
**Primary target model:** `Qwen/Qwen3-30B-A3B` unless an existing validated MoE SD stack provides a better target  
**Primary physical GPUs:** `2,3,6,7` only  
**Primary topology:** TRUE `TP2-DP2-EP4-PP1` if compatible with the SD implementation

## 0. Core research question

Speculative decoding replaces several sequential target-model decode steps with one wider target verification step.

For speculation depth `gamma`:

```text
AR:
target step 1
target step 2
target step 3
target step 4

SD, gamma=4:
draft proposes 4 future tokens
        ↓
target verifies 4 candidate positions in one wider forward
        ↓
accepts a prefix
```

For a dense target, wider verification can amortize weight loading and launch overhead.

For a MoE target, wider verification may activate more experts.

For a **physical EP** target, wider verification additionally changes:

- routed activation rows,
- source→destination rank edges,
- destination-rank fanout,
- per-rank send/receive volume,
- critical-rank traffic,
- Dispatch/Combine time.

This PoC asks:

> Does physical EP create a speculation-depth trade-off that cannot be explained well by verified-token count, accepted-token count, unique experts, or aggregate verification cost alone?

More concretely:

> As `gamma` increases, does EP first benefit from larger verification batches / collective amortization, then lose efficiency because speculative rows expand rank fanout and critical-rank A2A traffic, creating an EP-specific optimum?

Do not implement a new adaptive method until this missing cost dimension is established.

## 1. Prior-work boundary

Audit these works before implementation.

### SpecMoE — arXiv:2604.10152

**SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding**

Boundary:

- memory-constrained / CPU-offloaded MoE inference;
- self-assisted speculative decoding;
- amortizes expert parameter migration and memory/interconnect bandwidth.

Do not claim novelty for:

- applying SD to MoE;
- using SD to amortize MoE parameter transfer;
- self-assisted MoE SD.

Our candidate gap is GPU-resident **physical EP activation A2A**.

### Cascade

Audit the current version.

Boundary:

- MoE verification becomes expensive as speculation grows;
- adaptive speculation depth / verification utility.

Do not claim novelty for dynamically choosing speculation depth because verification cost increases.

### MoE-Spec — arXiv:2602.16052

Boundary:

- large draft trees activate many unique experts;
- expert budgeting / memory-cost control.

Do not claim novelty for expert-count-aware or expert-budget-aware verification.

### EVICT — arXiv:2605.00342

**Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding**

Boundary:

- lossless adaptive verification;
- truncates draft verification work using benefit vs profiled verification cost;
- reduces unnecessary expert activations.

### EcoSpec — arXiv:2607.12696

**Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts**

Boundary:

- expert scattering;
- marginal expert activation cost in candidate selection.

Do not claim novelty for avoiding disjoint expert activation itself.

### Speculative MoE — arXiv:2503.04398

**Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling**

This is the most important EP-specific prior.

Boundary:

- explicitly targets EP All-to-All;
- predicts token/expert routing;
- speculative token shuffling / expert grouping / pre-scheduling;
- reduces EP communication.

Its speculative mechanism is routing pre-scheduling, not ordinary draft→target→accept/reject SD.

If it already covers the exact interaction:

`draft verification width ↔ acceptance ↔ physical EP rank traffic ↔ Dispatch/Combine per committed token`

stop with:

`OVERLAP_WITH_SPECULATIVE_MOE`

## 2. Novelty claim boundary

The only candidate contribution in this PoC is:

> Physical EP rank traffic may be an independent cost dimension for MoE speculative verification, beyond verified-token count and unique-expert footprint.

Positive evidence would look like:

Two verification steps have similar:

- gamma,
- verified rows,
- accepted length,
- unique experts,

but materially different:

- remote routed rows,
- destination-rank fanout,
- max-rank bytes,
- rank skew,

and therefore different:

- Dispatch latency,
- Combine latency,
- target verification latency.

If physical EP metrics add little explanatory value beyond existing MoE-SD features, stop.

## 3. Exactness / decoding semantics

Use **lossless target semantics**.

Preferred first implementation:

- deterministic / greedy SD;
- accept the longest draft prefix matching target greedy decisions;
- emit target correction token on mismatch.

If the existing runtime already provides exact stochastic SD, it may be reused.

Do not:

- change target output semantics;
- drop experts;
- approximate router decisions;
- alter target top-k;
- use quality-changing expert budgets.

## 4. Target and draft model selection

### Target

Prefer:

`Qwen/Qwen3-30B-A3B`

because the repo already has validated TRUE EP4 infrastructure.

Audit actual config:

- layers,
- hidden size,
- routed experts,
- top-k,
- MoE intermediate size,
- dtype,
- tokenizer.

Expected project geometry:

- 48 layers
- hidden 2048
- 128 routed experts
- top-k 8
- MoE intermediate 768

Loaded config is authoritative.

### Draft

Use the cheapest compatible draft already supported by the runtime.

Candidate:

- a small Qwen3 dense model sharing tokenizer/vocabulary,
- or an existing self-draft / EAGLE / Medusa-style mechanism if already validated.

Priority:

1. correctness,
2. reproducibility,
3. low implementation burden.

Do not train a new drafter for this PoC.

Record:

- model/revision,
- draft latency,
- tokenizer compatibility,
- acceptance rate.

## 5. Physical GPU contract

For TRUE EP4 use only:

`CUDA_VISIBLE_DEVICES=2,3,6,7`

Do not use other physical GPUs.

Record:

- physical→logical mapping,
- GPU model/HBM,
- `nvidia-smi topo -m`,
- NVLink/NVSwitch topology,
- foreign GPU processes,
- CUDA/PyTorch/vLLM/DeepEP versions,
- branch/commit.

## 6. Two-phase PoC strategy

Do not begin with a production SD scheduler.

### Phase A — trace / structural discovery

Establish:

- acceptance behavior,
- expert-set expansion,
- rank-fanout expansion,
- remote routed rows,
- critical-rank traffic.

### Phase B — TRUE EP4 timing

Only if Phase A shows non-trivial rank-level variation.

Measure:

- Dispatch,
- Expert,
- Combine,
- target verification latency

on GPUs `2,3,6,7`.

If rank-level metrics collapse almost perfectly onto unique-expert count, stop before heavy physical implementation.

## 7. Workloads

Use at least two if cheap:

- reasoning: GSM8K or equivalent
- coding: HumanEval or equivalent

Target:

- > =64 reasoning requests
- > =32 coding requests

Prefer existing repo subsets.

Use identical prompts across AR and SD arms.

Cap output length if needed while retaining enough verification steps, e.g. 64–128 generated tokens/request.

## 8. Primary arms

### Arm A — target-only AR

No draft model.

### Arm B — SD gamma = 2

### Arm C — SD gamma = 4

### Arm D — SD gamma = 8

Optional gamma=16 only if cheap and stable.

All SD arms must share target, draft, acceptance semantics, EP placement, runtime, and dataset.

## 9. Fundamental SD accounting

For every target verification invocation record:

- `gamma_requested`
- `drafted_tokens`
- `verified_candidate_tokens`
- `accepted_draft_tokens`
- `correction_token_committed` (0/1)
- `total_committed_progress`
- acceptance prefix length
- acceptance ratio
- draft latency
- target verification latency

Recommended:

`committed_progress = accepted_draft_tokens + correction_token_committed`

when valid for the chosen exact greedy SD implementation.

Also retain accepted-draft-only normalization.

## 10. MoE accounting

For every target verification invocation and MoE layer:

- routed token rows
- total top-k routes
- unique experts
- active experts per rank
- expert-union size
- max expert token count
- mean expert token count
- expert-load skew

These are the existing MoE-SD cost dimensions.

## 11. Physical EP accounting

For every verification invocation/layer/source rank:

- local routed rows
- remote routed rows
- unique destination ranks/token
- source→destination rank edge count
- logical Dispatch bytes
- logical Combine bytes
- per-rank send bytes
- per-rank receive bytes
- max-rank send bytes
- max-rank receive bytes
- mean-rank bytes
- max/mean rank skew
- destination-rank fanout
- critical destination rank
- critical-rank persistence across layers

Use exact expert→rank placement.

Do not derive all EP cost from unique-expert count alone.

## 12. Physical timing

For TRUE EP4 target verification record on a defensible cumulative-critical-rank basis:

- Router
- Dispatch
- Expert
- Combine
- total MoE stage
- target verification forward

Do not use rank-local mean as EP stage latency.

Use:

- clean lane: minimal CUDA events, final latency
- trace lane: route capture/NVTX/hooks/profiler, diagnostics only

The question is not Dispatch-vs-Combine asymmetry.  
The question is how speculative width changes **EP cost per useful committed progress**.

## 13. Primary derived metrics

### EP communication time per committed token

`EP_comm_per_committed = sum(Dispatch + Combine) / committed_progress`

Primary physical metric.

### Target verification time per committed token

`target_verify_per_committed = target_verification_latency / committed_progress`

### E2E time per committed token

Include:

- draft time
- target verification
- correction overhead
- scheduler/runtime overhead

### EP bytes per committed token

`EP_bytes_per_committed = logical Dispatch+Combine bytes / committed_progress`

### Rejected-suffix communication fraction

Attribute verification rows belonging to draft tokens that are not committed:

`rejected_EP_byte_fraction = logical EP bytes attributable to rejected draft rows / total verification EP bytes`

Label this as logical attribution, not counterfactual saved latency.

## 14. Amortization vs waste decomposition

For increasing gamma quantify both sides.

### Benefit — batching / collective amortization

Compare:

- one gamma-wide verification invocation
versus
- gamma separate AR-like target invocations

Metrics:

- target invocations / committed token
- EP collective invocations / committed token
- Dispatch+Combine time / verified row
- Expert time / verified row

### Cost — speculative expansion

Measure:

- rejected rows
- unique experts
- remote rows
- rank fanout
- critical-rank bytes
- rank skew

Do not assume a crossover; test it.

## 15. Matched-pair analysis

This is more important than global correlation.

Find verification steps matched on:

- same gamma
- similar verified rows
- similar committed progress
- similar unique expert count

but substantially different physical rank structure.

Example:

```text
Case A:
gamma=4
verified=4
committed=3
unique experts=42
max-rank bytes=8 MB
fanout=2.1
EP time=?

Case B:
gamma=4
verified=4
committed=3
unique experts=42
max-rank bytes=16 MB
fanout=3.8
EP time=?
```

If EP time differs materially, unique-expert footprint is incomplete.

## 16. Cost-model comparison

Use request-grouped cross-validation.

### M0 — generic SD

- gamma
- verified token count
- committed progress
- acceptance ratio

### M1 — existing MoE-SD

M0 +

- unique experts
- active experts
- expert-union size
- expert-load skew

### M2 — physical EP

M1 +

- remote routed rows
- destination-rank fanout
- max-rank bytes
- rank skew
- source→destination edge count

Predict:

- Dispatch+Combine time
- total MoE stage
- target verification latency

Use interpretable models first:

- linear / ridge
- simple tree model only as secondary diagnostic

Report:

- grouped-CV MAE
- grouped-CV R²
- M2 error reduction vs M1

Do not leak verification steps from the same request across train/test folds.

## 17. Missing-cost-dimension gate

Physical EP is interesting only if it adds material information.

Strong evidence should satisfy at least one, preferably several:

1. M2 reduces grouped-CV error for EP communication by &gt;=15% vs M1;
2. M2 improves grouped-CV R² by &gt;=0.10 absolute;
3. matched steps with similar unique experts but different rank structure show &gt;=20% median EP-time difference;
4. the depth minimizing expert-count/HBM proxy differs from the depth minimizing measured EP cost for &gt;=25% of requests/decision points.

If none holds:

`EP_COST_COLLAPSES_TO_EXISTING_MOE_SD_COST`

Stop.

## 18. Speculation-depth crossover

For each gamma report:

- acceptance rate
- committed progress / target invocation
- target invocations / generated token
- EP communication / invocation
- EP communication / committed token
- target verification / committed token
- E2E latency / generated token

Required plots:

1. gamma vs acceptance
2. gamma vs unique experts
3. gamma vs remote routed rows
4. gamma vs destination-rank fanout
5. gamma vs max-rank bytes
6. gamma vs Dispatch+Combine / verification
7. gamma vs Dispatch+Combine / committed token
8. gamma vs target verification / committed token
9. gamma vs E2E latency / generated token

Desired but not forced pattern:

```text
small gamma:
poor amortization

medium gamma:
best EP cost per committed token

large gamma:
rank traffic + rejected work dominate
```

## 19. Offline EP-aware oracle

Only if the missing-cost dimension is confirmed.

At each speculation opportunity choose among:

`{1,2,4,8}`

the gamma minimizing measured or faithfully replayed physical EP cost per committed progress.

Compare with:

- best fixed gamma
- acceptance-only heuristic
- unique-expert-cost oracle/proxy
- aggregate target-verification-cost oracle where feasible

This is not yet an online method.

Question:

> Does EP-aware selection create meaningful headroom beyond expert-aware selection?

## 20. Method-promotion gate

Do not implement an online EP-aware scheduler unless:

- physical EP metrics pass the missing-cost gate;
- EP-aware oracle differs meaningfully from expert-aware oracle;
- EP-aware oracle improves target-side or E2E latency by &gt;=10% over the strongest expert-aware/fixed exact baseline;
- effect is broad across requests/datasets;
- improvement is not just acceptance-rate variation.

Preferred promotion target:

- > =15% E2E gain.

If oracle gain &lt;10%:

`EP_AWARE_POLICY_HEADROOM_TOO_SMALL`

Stop.

## 21. Prior comparison required in final report

### Versus SpecMoE

- CPU↔GPU expert-parameter movement vs GPU↔GPU activation EP
- self-assisted SD vs our draft setup
- whether physical EP effect is already implied

### Versus Cascade

- aggregate verification cost vs decomposed physical EP traffic
- whether EP-aware optimum differs

### Versus MoE-Spec

- expert budget vs exact EP communication
- expert dropping is outside this exact PoC

### Versus EVICT

- profiled aggregate verification cost vs physical rank-aware cost
- whether rank metrics add predictive value

### Versus EcoSpec

- marginal/unique expert scattering vs physical rank scattering
- same unique experts may map differently to ranks

### Versus Speculative MoE (2503.04398)

- routing pre-scheduling vs draft-target accept/reject SD
- audit any direct overlap in communication-cost modeling

## 22. Discovery order

1. Audit prior work.
2. Audit repo/runtime SD support.
3. Select the simplest exact draft mechanism.
4. Run target-only AR.
5. Generate gamma 2/4/8 traces.
6. Compute expert metrics.
7. Compute physical EP metrics.
8. Test whether EP metrics add information beyond unique experts.
9. Only if yes, run/expand TRUE EP4 timing.
10. Compute EP-aware oracle headroom.
11. Only if oracle headroom is strong, propose a method.

## 23. Negative verdicts

### `EP_COST_COLLAPSES_TO_EXISTING_MOE_SD_COST`

Physical rank metrics add little beyond gamma/acceptance/unique experts.

### `EP_COMMUNICATION_TOO_SMALL`

Dispatch+Combine is too small a fraction of target verification/E2E latency.

### `NO_SPECULATION_CROSSOVER`

No meaningful gamma-dependent EP trade-off.

### `EP_AWARE_POLICY_HEADROOM_TOO_SMALL`

EP-aware oracle &lt;10% gain beyond strongest baseline.

### `OVERLAP_WITH_SPECULATIVE_MOE`

Prior work already covers the same mechanism/cost model.

### `SD_RUNTIME_NOT_VALIDATED`

Target semantics or timing correctness is not trustworthy.

## 24. Positive verdicts

### `PHYSICAL_EP_MISSING_COST_CONFIRMED`

Physical EP features materially explain timing beyond expert footprint.

### `SD_EP_CROSSOVER_CONFIRMED`

Gamma exposes a clear amortization-vs-rank-traffic crossover.

### `EP_AWARE_SPECULATION_STRONG_SIGNAL`

Use only if:

- missing EP cost confirmed;
- oracle EP-aware policy &gt;=10% better than expert-aware/fixed baseline;
- preferably &gt;=15% E2E;
- broad effect;
- prior audit confirms novelty.

## 25. Statistics

Final physical key arms:

- warmup &gt;=3
- > =10 repeats where feasible
- request/verification-group bootstrap
- > =2,000 bootstrap replicates for headline deltas

Do not treat:

- layers,
- ranks,
- routed rows

as independent request-level samples.

For regression:

- group CV by request.

## 26. Instrumentation lanes

### Clean lane

- minimal CUDA events
- no profiler
- authoritative latency

### Trace lane

- route capture
- per-rank accounting
- NVTX/minimal hooks
- profiler only if needed

Profiler-heavy timing is diagnostic only.

## 27. Required artifacts

```text
poc_speculative_moe_ep/
  artifacts/
    prior_work_audit.json
    environment.json
    sd_config.json
    ar_traces/
    gamma2_traces/
    gamma4_traces/
    gamma8_traces/
    route_metrics.parquet
    verification_metrics.parquet
    matched_pair_analysis.json
    cost_model_results.json
    oracle_policy_results.json
    final_verdict.json
  reports/
    prior_work_audit.md
    runtime_audit.md
    sd_correctness.md
    expert_footprint_analysis.md
    physical_ep_analysis.md
    amortization_vs_waste.md
    cost_model_comparison.md
    oracle_headroom.md
    final_summary.md
    reproduction.md
  figures/
  scripts/
  tests/
```

## 28. Required figures

1. gamma vs acceptance / committed progress
2. gamma vs unique experts
3. gamma vs remote routed rows
4. gamma vs destination-rank fanout
5. gamma vs max-rank bytes
6. gamma vs Dispatch+Combine per verification
7. gamma vs Dispatch+Combine per committed token
8. gamma vs E2E latency per generated token
9. unique experts vs EP time, colored by max-rank bytes/fanout
10. matched unique-expert groups: rank traffic vs EP latency
11. M1 vs M2 grouped-CV prediction error
12. fixed/expert-aware/EP-aware oracle comparison

## 29. Required final summary block

```text
POC_TYPE:
TARGET_MODEL:
TARGET_REVISION:
DRAFT_MODEL:
DRAFT_REVISION:
TOKENIZER_COMPATIBILITY:
SD_SEMANTICS:
TARGET_OUTPUT_EQUIVALENCE:

GPU_SET:
PHYSICAL_TO_LOGICAL_MAPPING:
GPU_MODEL:
TOPOLOGY:
EP_SIZE:
DEEPEP_BACKEND:

DATASETS:
REQUEST_COUNT:
GENERATED_TOKENS:

AR_TARGET_MS_PER_TOKEN:
AR_EP_COMM_MS_PER_TOKEN:

GAMMA2_ACCEPTANCE:
GAMMA2_COMMITTED_PER_VERIFY:
GAMMA2_UNIQUE_EXPERTS:
GAMMA2_REMOTE_ROWS:
GAMMA2_RANK_FANOUT:
GAMMA2_MAX_RANK_BYTES:
GAMMA2_EP_MS_PER_COMMITTED:
GAMMA2_E2E_MS_PER_TOKEN:

GAMMA4_ACCEPTANCE:
GAMMA4_COMMITTED_PER_VERIFY:
GAMMA4_UNIQUE_EXPERTS:
GAMMA4_REMOTE_ROWS:
GAMMA4_RANK_FANOUT:
GAMMA4_MAX_RANK_BYTES:
GAMMA4_EP_MS_PER_COMMITTED:
GAMMA4_E2E_MS_PER_TOKEN:

GAMMA8_ACCEPTANCE:
GAMMA8_COMMITTED_PER_VERIFY:
GAMMA8_UNIQUE_EXPERTS:
GAMMA8_REMOTE_ROWS:
GAMMA8_RANK_FANOUT:
GAMMA8_MAX_RANK_BYTES:
GAMMA8_EP_MS_PER_COMMITTED:
GAMMA8_E2E_MS_PER_TOKEN:

BEST_FIXED_GAMMA:
BEST_EXPERT_AWARE_GAMMA_ORACLE:
BEST_EP_AWARE_GAMMA_ORACLE:

M1_GROUPED_CV_MAE:
M2_GROUPED_CV_MAE:
M2_ERROR_REDUCTION_PCT:
M1_GROUPED_CV_R2:
M2_GROUPED_CV_R2:

MATCHED_UNIQUE_EXPERT_EP_TIME_DELTA:
EP_COMM_FRACTION_OF_TARGET_VERIFY:
REJECTED_EP_BYTE_FRACTION:

EP_AWARE_ORACLE_VS_FIXED_GAIN:
EP_AWARE_ORACLE_VS_EXPERT_AWARE_GAIN:

PRIOR_OVERLAP_STATUS:
PHYSICAL_EP_MISSING_COST_STATUS:
SPECULATION_CROSSOVER_STATUS:
OVERALL_VERDICT:
NEXT_ACTION:
BRANCH:
COMMIT:
```

## 30. Claim boundary

A strong positive PoC may support:

> In speculative decoding for a GPU-resident MoE target, target-verification cost is not fully characterized by candidate count or unique-expert footprint. Physical expert placement and rank-level A2A traffic introduce an additional cost dimension, creating an EP-specific speculation-depth trade-off.

Do not claim yet:

- a new online scheduler,
- end-to-end speedup from a proposed method,
- superiority to Cascade/EVICT/EcoSpec,
- production relevance.

The discovery PoC should answer only:

1. **Does physical EP add a missing cost dimension to MoE speculative decoding?**
2. **Does that dimension change the best speculation regime enough to justify an EP-aware method?**

