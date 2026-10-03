# dLLM MoE-EP Request-Persistent Expert Residency PoC
## Cross-Block Persistence, Long-Horizon Amortization, and dLLM-Specificity

**Target repository:** `/home/esjung/MLLM-EP-github`  
**Recommended server path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_request_persistent_replica_spec.md`  
**Available GPUs:** `2, 3, 6, 7` only  
**Primary model/runtime:** reuse the validated `inclusionAI/LLaDA2.0-mini` + dInfer + DeepEP physical EP4 stack  
**Primary regime:** batch-1 / low-request, long-generation dLLM serving

## 1. Goal

The previous block-local replication PoC failed because cold expert copies were far too expensive to amortize over a single diffusion block, early-block causal prediction recovered too little oracle headroom, and large replica budgets created compute hotspots.

This PoC tests the only remaining exact-placement hypothesis:

> Can a small request-specific expert replica set be prepared once from the first one or two diffusion blocks, then kept resident across many later blocks so that the copy cost is amortized over a much longer lifetime?

The direction is only interesting if request-specific cross-block persistence provides materially better communication/load tradeoffs than a strong generic global/static replication baseline.

## 2. Prior / Novelty Boundary

Generic MoE serving already studies or supports:
- popular-expert replication;
- redundant experts;
- EPLB;
- periodically rebalanced expert placement;
- predictive expert replication.

Therefore, “keep popular experts replicated for a long time” is not enough.

The required dLLM-specific claim is:

> Within one long dLLM request, expert/destination demand learned from early diffusion blocks remains predictive across later blocks and beats a server-wide global/static expert prior at matched memory and load balance.

If request-specific persistence does not beat global/static, close the replication family.

## 3. Core Hypothesis

A long dLLM request contains multiple diffusion blocks:

```text
Request
  |- Block 0: iter 0 ... T0
  |- Block 1: iter 0 ... T1
  |- Block 2: iter 0 ... T2
  |- ...
```

Instead of cold copying experts inside every block:

```text
Block 0: copy -> use a few iterations -> evict
Block 1: copy -> use a few iterations -> evict
```

use request-persistent residency:

```text
Block 0: profile
         -> prepare selected replicas once

Block 1: reuse
Block 2: reuse
Block 3: reuse
...
Request end: release slots
```

The expected benefit is a much longer amortization horizon.

## 4. Exactness Requirement

This is still a placement-only direction.

The method must not change:
- router logits;
- top-k expert IDs;
- routing weights;
- routed scaling;
- acceptance decisions;
- denoising schedule;
- expert parameters;
- precision.

Only the physical execution location of an already-selected expert may change.

If a runtime prototype is reached, expected quality delta is exactly zero.

## 5. Layer-Aware / Source-Aware State

For LLaDA2.0-mini:
- 19 sparse MoE layers;
- 256 routed experts/layer;
- top-k 8;
- about 6 MiB per `(layer, expert)` routed weight tensor.

Planning unit:

```text
(layer, source_rank, expert_id)
```

Request profile:

```text
RequestProfile[request][layer][source_rank][expert_id]
```

Never merge an expert ID across layers.

## 6. Destination-Closure-Aware Accounting

Use the same validated destination-deduplicated communication accounting as the prior PoCs.

Do not optimize only expert-hit rate.

Primary structural objective:

```text
minimize future live-only destination-deduplicated Dispatch+Combine bytes
```

at matched memory and load constraints.

## 7. Two-Stage PoC

### Stage A — Offline cross-block persistence and amortization
Primary stage. No runtime replication yet.

### Stage B — Minimal exact request-persistent runtime prototype
Only if Stage A passes all key gates.

If Stage A fails, stop.

## 8. Hardware

Use only:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

Re-record:
- `nvidia-smi`
- `nvidia-smi topo -m`
- GPU UUIDs;
- logical rank mapping;
- expert ownership;
- available HBM;
- runtime/library revisions.

Do not modify other users' processes.

## 9. Model / Runtime

Primary:
`inclusionAI/LLaDA2.0-mini`

Reuse the validated dInfer + DeepEP TRUE EP4 environment from previous dLLM PoCs.

Do not silently upgrade the stack.

Revalidate physical EP4 with a tiny deterministic probe.

## 10. Workload Design

This PoC needs many diffusion blocks per request.

Use two lanes.

### Lane A — Existing short-horizon continuity
Reuse previous HumanEval/GSM8K traces where possible.

### Lane B — Long-horizon frozen cohort
Collect a small predeclared long-generation cohort.

Priority:
1. long-form coding/code completion;
2. structured generation;
3. long reasoning as secondary.

Use generation budgets that naturally create many blocks, e.g. 256 or 512 output tokens if supported.

Keep official/default decoding threshold and block behavior.

Do not tune acceptance thresholds to increase persistence.

Suggested cheap discovery size:
- 8–12 calibration requests;
- 16–24 held-out requests.

Freeze request IDs and generation settings before policy evaluation.

## 11. Accuracy / Quality

Stage A is offline and does not alter outputs.

If Stage B is reached, require:
- same final token IDs;
- same acceptance schedule;
- same NFE;
- same router top-k IDs;
- same routing weights within tolerance.

For benchmark-capable requests also report:
- HumanEval pass@1 / unit-test pass rate;
- GSM8K exact match.

Expected quality delta: 0.

Do not trade quality for speed.

## 12. Stage A1 — Cross-Block Persistence

For every request, sparse layer, source rank, and block build:

```text
P[b, layer, source, expert]
D[b, layer, source, destination_rank]
```

using the same live-only traffic scope from the prior residual analysis.

Measure:
- Block b -> b+1;
- Block 0 -> Block k;
- Blocks 0–1 -> all future blocks.

Required metrics:
- expert-distribution cosine;
- Jensen-Shannon divergence;
- top-M expert overlap;
- weighted future-block expert coverage;
- destination-rank similarity;
- destination-closure coverage.

Most important:
- how well Block 0 predicts all future blocks;
- how well Blocks 0–1 predict all future blocks;
- how quickly request-specific signal decays with block distance.

## 13. Controls

Mandatory controls:

### C1 — Global/static calibration prior
Server-wide per-layer/source popularity from calibration requests only.

### C2 — Other-request matched block
Use a block with similar live-token count from another request.

### C3 — Same-request distance
Measure adjacent and distant future blocks.

Decisive comparison:

```text
early blocks from this request
vs
global/static popularity
```

for future blocks.

If global is as good or better, close.

## 14. Persistent Replica Policies

### P0 — No replication
Live-only baseline.

### P1 — Global Warm Static
Server-wide static replica set from calibration traffic.
Assume it is prewarmed before the request.

This is a strong generic baseline.

### P2 — First-Block Persistent
Use Block 0 only.
Prepare once after Block 0 and keep for all later blocks.

### P3 — First-2-Blocks Persistent
Use Blocks 0 and 1.
Prepare/update once after Block 1 and keep for the rest.

### P4 — Bounded Incremental Persistent
Optional if cheap.
Allow at most one later update.
Charge every copy.

Do not allow per-block cold replacement.

### P5 — Request Oracle Persistent
Future-aware upper bound: one persistent set selected using all future blocks.

### P6 — Load-Aware Oracle Persistent
Future-aware, but constrained by rank-load cap.

No learned predictor in this PoC.

## 15. Memory Budget Sweep

Use:
- 0.5 GiB/rank;
- 1 GiB/rank;
- 2 GiB/rank;
- 4 GiB/rank.

Emphasize 0.5/1/2 GiB as realistic candidates because large budgets previously created hotspots.

## 16. Load Constraints

Report:
- max-rank expert assignments;
- max/mean;
- rank CV;
- worst layer/iteration rank increase.

Preferred deployable cap:

```text
<= +10% critical/max-rank load
```

Also show relaxed +15% and +20% curves.

Do not hide instantaneous hotspots with request-total averages.

## 17. Persistent Copy Accounting

Each selected expert should be copied at most once per request unless an explicit bounded update occurs.

Report:
- initial copy bytes;
- update bytes;
- total copy bytes/request;
- number of future blocks using each replica;
- number of future refinement iterations using each replica.

Define:

```text
byte ROI =
total destination-deduplicated D+C bytes removed
/
total expert weight bytes copied
```

Also estimate timing ROI.

## 18. Timing ROI

Use measured P2P copy service time and prior measured DeepEP D+C timing.

Do not equate bytes and time.

Report:
- copy_ms;
- D+C ms avoided;
- net communication-side benefit;
- break-even block index;
- break-even iteration count.

Define:

```text
time ROI = avoided D+C ms / copy ms
```

Use conservative estimates.

If timing mapping is too uncertain, mark unresolved.

## 19. Main dLLM-Specific Metric

Define:

```text
RequestSpecificGainOverGlobal =
D+C_bytes_after_global
-
D+C_bytes_after_request_persistent
```

and:

```text
RelativeGainOverGlobal =
RequestSpecificGainOverGlobal
/
D+C_bytes_after_global
```

This is decisive.

The direction only survives if request-specific persistence adds meaningful value beyond a generic warm static baseline.

## 20. Persistent Oracle Recovery Beyond Global

Define:

```text
OracleRecoverableBeyondGlobal =
bytes(global) - bytes(request_oracle)

CausalRecoveredBeyondGlobal =
bytes(global) - bytes(request_causal)

RecoveredPersistentOracle =
CausalRecoveredBeyondGlobal
/
OracleRecoverableBeyondGlobal
```

The denominator is headroom beyond the global baseline, not beyond no-replica.

## 21. Residual Amplification

Continue to normalize against:

```text
B_one_shot = 1.0×
```

Report:
- live-only residual;
- global warm residual;
- first-block persistent residual;
- first-2-block persistent residual;
- oracle persistent residual;
- load-aware oracle residual.

Do not silently mix different held-out cohorts.

## 22. Persistence Horizon

For block distance d = 1,2,3,... plot the request-specific advantage over global.

Desired qualitative signal:

```text
distance    +1    +2    +4    +8
gain        high  high  positive positive
```

Bad signal:

```text
+1: slight request advantage
+2 onward: global wins
```

That cannot support long-term amortization.

## 23. Optional AR Contextual Control

If cheap, use the already-working Qwen3-30B-A3B TRUE EP4 AR stack as a secondary contextual control.

Construct:
- early decode window;
- later decode windows.

Within each model compare:
- request-specific predictor gain over global;
- persistence decay with distance.

Do not compare raw routing vectors across different models.

This is supporting evidence only.

## 24. Stage A Gates

### A1 — Request-Specificity
At matched memory/load:

```text
request-persistent causal > global warm static
```

Desired relative D+C reduction beyond global:

```text
>= 10%
```

with request-bootstrap CI above zero.

Interpretation:
- <=0%: NO-GO
- 0–5%: weak
- 5–10%: CONDITIONAL
- >=10%: GO signal

This is the most important novelty gate.

### A2 — Persistence Horizon
Request-specific advantage should remain positive for at least 4 future blocks or a comparable meaningful fraction of the request.

### A3 — Copy Amortization
Require a feasible causal point with:

```text
time ROI >= 2×
```

If only byte ROI is trustworthy, use `byte ROI >= 2×` as a preliminary gate.

Break-even should occur before 50% of the remaining request lifetime; preferably before 25%.

### A4 — Load Balance
Preferred:
`critical/max-rank load increase <=10%`

10–15% may be CONDITIONAL.

>20% is NO-GO unless later measured runtime timing clearly contradicts the structural estimate.

### A5 — Oracle Recovery Beyond Global
Require:

```text
RecoveredPersistentOracle >= 50%
```

Preferred >=70%.

## 25. Stage A Decision

### NO-GO
Close persistent replication if:
- request-persistent does not beat global;
- advantage decays after only one block;
- copy still cannot amortize;
- compute hotspot remains severe;
- global warm captures almost all persistent headroom.

If this happens, close expert-weight movement/replication as a dLLM-specific direction and move to activation-side communication optimization.

### CONDITIONAL
Use if request-specificity is strong but timing profitability remains uncertain.

### GO
Proceed only if request-specificity, persistence, amortization, and load balance all pass.

## 26. Stage B — Minimal Exact Runtime Prototype

Only after Stage A passes.

Runtime behavior:

```text
Request starts

Block 0:
    normal execution + profiling

Optional Block 1:
    additional profiling

Prepare selected replicas once

Keep replicas resident across later blocks

Request end:
    release slots
```

No per-block reload.

A bounded late update is allowed only if Stage A shows it is profitable.

## 27. Runtime Correctness

Before performance claims require:
- identical final token IDs;
- identical acceptance schedule;
- identical NFE;
- identical router top-k IDs;
- routing weights equal within tolerance;
- exact expert tensor identity;
- same routed scaling/combine semantics.

Any correctness deviation invalidates performance conclusions.

## 28. Runtime Accuracy

For benchmark-capable requests:
- HumanEval pass@1 / unit-test pass rate;
- GSM8K exact match.

Expected delta = 0.

For long-form prompts without automatic scores, exact token parity is required.

## 29. Runtime Performance

Measure:
- replica preparation ms;
- Dispatch ms;
- Expert ms;
- Combine ms;
- D+C ms;
- MoE-path ms;
- denoising span;
- request E2E;
- extra HBM.

Compare at matched memory:
- Vanilla;
- Global Warm Static;
- Request-Persistent Causal.

The main runtime comparison is request-persistent vs global, not only vs vanilla.

## 30. Stage B Runtime Gates

Desired:

```text
>=20% additional D+C latency reduction vs Global Warm
>=5% E2E latency reduction vs Global Warm
>=7% E2E reduction vs Vanilla
```

with exact correctness.

>=10% E2E reduction vs Vanilla plus a clear gain over Global would be strong.

## 31. Required Plots

1. Cross-block predictor advantage vs block distance.
2. Residual amplification vs replica budget.
3. Gain over global vs future block index.
4. Copy ROI / break-even vs request length.
5. D+C saving vs max-rank load increase Pareto.
6. Layer-level persistent replica allocation.
7. Runtime timing breakdown if Stage B is reached.

## 32. Main Table Shape

| Method | Quality | Norm. EP Traffic | D+C ms | E2E ms | Extra HBM | Copy ms |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | same | measured | measured | measured | 0 | 0 |
| Live-only structural ref | n/a | reference | n/a | n/a | 0 | 0 |
| Global Warm Static | same | ... | ... | ... | ... | prewarm |
| Request-Persistent Causal | same | ... | ... | ... | ... | ... |
| Request-Persistent Oracle | n/a | ... | n/a | n/a | ... | ... |
| One-shot reference | n/a | 1.0× | n/a | n/a | n/a | n/a |

## 33. Final Report Questions

The final `report.md` must answer:

1. How stable are per-layer/source expert distributions across blocks?
2. How quickly does persistence decay with block distance?
3. Does an early request profile predict later blocks better than global/static popularity?
4. At matched memory/load, how much D+C traffic does request-persistent placement save beyond global?
5. How many future blocks retain that advantage?
6. What memory budget is required?
7. What are copy byte ROI and timing ROI?
8. Where is the break-even point?
9. Does persistent placement create compute hotspots?
10. How much persistent oracle headroom exists beyond global?
11. How much of that headroom does causal prediction recover?
12. Does long-generation coding improve amortization?
13. If optional AR control is run, is persistence stronger/longer in dLLM?
14. Was Stage B promoted?
15. If yes, is exact token/acceptance/router/NFE parity maintained?
16. If yes, are benchmark scores unchanged?
17. If yes, what measured D+C and E2E gains occur vs Global and vs Vanilla?
18. Final verdict: GO / CONDITIONAL / NO-GO.
19. If NO-GO, should the entire expert-replication family be closed?

## 34. Statistics

Use requests as independent units.

Report mean, median, and 95% request-bootstrap CI.

Do not treat blocks, layers, iterations, or tokens as independent samples.

Cluster block-distance analyses by request.

## 35. Outputs

Recommended directory:

```text
poc_dllm_request_persistent_replica/
├── scripts/
│   ├── reuse_short_traces.py
│   ├── collect_long_horizon.py
│   ├── analyze_cross_block_persistence.py
│   ├── simulate_persistent_replica.py
│   ├── estimate_copy_amortization.py
│   ├── analyze_load_balance.py
│   ├── optional_ar_control.py
│   ├── runtime_persistent_prototype.py
│   ├── validate_correctness.py
│   └── evaluate_accuracy.py
├── results/
│   ├── environment.json
│   ├── block_similarity.csv
│   ├── persistence_decay.csv
│   ├── policy_budget.csv
│   ├── gain_over_global.csv
│   ├── copy_roi.csv
│   ├── load_balance.csv
│   ├── runtime_timing.csv
│   ├── correctness.json
│   ├── accuracy.json
│   └── summary.json
├── figures/
│   ├── persistence_vs_block_distance.png
│   ├── residual_vs_budget.png
│   ├── gain_over_global_by_block.png
│   ├── copy_break_even.png
│   ├── comm_load_pareto.png
│   └── runtime_performance.png
└── report.md
```

Do not overwrite prior PoCs.

## 36. Evidence Integrity

- Freeze long-horizon cohort before policy results.
- Use calibration only for global/static prior.
- No future-block information in causal policies.
- Preserve exact layer/expert ownership.
- Validate destination-dedup accounting.
- Preserve raw traces and failed runs append-only.
- Record exact model/runtime revisions.
- Preserve unrelated dirty/staged work.

## 37. Git Hygiene

Suggested branch:

```text
flashvep/dllm-request-persistent-replica-poc
```

Commit/push only this PoC's files.

Report final branch, commit SHA, and report path.

## 38. Final Criterion

Continue only if:

```text
early request blocks
    ->
stable request-specific per-layer/source expert affinity
    ->
predict later blocks better than global popularity
    ->
persistent replicas stay useful across many blocks
    ->
one-time copy amortizes
    ->
load balance stays controlled
    ->
exact semantics are preserved
    ->
runtime beats Global Warm Static, not only Vanilla
```

If request-persistent placement does not beat the strong global/static baseline, close expert-weight replication as the dLLM-specific EP direction and move to activation-side communication optimization.
