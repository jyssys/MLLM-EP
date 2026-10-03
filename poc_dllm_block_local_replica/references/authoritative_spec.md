# dLLM MoE-EP Block-Local Expert Replication PoC
## Layer-Aware, Destination-Closure-Aware Replication for Repeated Live-Token Communication

**Repository:** `/home/esjung/MLLM-EP-github`  
**Server spec path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_block_local_ep_replication_spec.md`  
**GPUs:** `2,3,6,7` only  
**Primary model/runtime:** reuse the validated `inclusionAI/LLaDA2.0-mini` + dInfer + DeepEP physical EP4 setup from `poc_dllm_ep_comm`  
**Primary regime:** batch-1 / low-request serving

---

## 1. Goal

The previous physical EP4 PoC measured:

- Vanilla raw communication amplification: **13.825×**
- Optimistic live-only residual: **5.773×**
- Dispatch+Combine / request wall: **17.974%**
- Dispatch+Combine / denoising span: **19.255%**

The 5.773× value is not measured Epoch traffic. It is an optimistic structural reference that removes all later sparse-EP traffic from positions that have already been accepted, while keeping fresh routing/expert work for positions that are still live.

This PoC tests whether the remaining repeated live-token EP communication can be reduced by **temporarily replicating exact expert weights close to the source rank for the lifetime of a diffusion block**.

Core hypothesis:

> Within a diffusion block, the same layers repeatedly route the same live token positions over several refinement iterations. If some remote expert destinations recur, copying a small set of exact expert weights once and reusing them locally may be cheaper than repeatedly moving live hidden states through EP Dispatch/Combine.

The PoC must first prove structural headroom with offline traces. Only if that passes should a minimal physical runtime prototype be implemented.

---

## 2. Difference from Epoch

Keep this distinction explicit.

Epoch-like optimization asks:

```text
Does this token position need fresh MoE computation?

Stable decoded:
    cache/reuse

Live:
    fresh router + fresh expert computation
```

This PoC asks:

```text
If a live position still needs fresh expert computation,
must that fresh computation cross the network again?

Live:
    fresh router
    fresh top-k
    fresh expert computation
    but use an exact local temporary replica when available
```

Therefore the main claimed opportunity is **live-token communication localization**, not dead-token skipping.

All structural savings must first be evaluated on top of the previous `B_live_only` accounting.

---

## 3. Layer-Aware Requirement

Never aggregate an expert ID across layers.

For LLaDA2.0-mini:

```text
19 sparse MoE layers
256 routed experts per sparse layer
top-k = 8
```

`Layer 3 / Expert 17` and `Layer 10 / Expert 17` are different tensors.

The basic planning unit is:

```text
(block, layer, source_rank, expert_id)
```

Conceptually:

```text
ReplicaSet[block][layer][source_rank]
```

All memory, copy-cost, and oracle calculations must preserve layer identity.

---

## 4. Destination-Closure-Aware Requirement

Do not optimize only by expert hit frequency.

If one token selects:

```text
GPU1: E33, E36
GPU2: E70, E72
GPU3: E101
```

replicating only `E33` locally does not remove the GPU1 hidden-row transfer because `E36` is still remote.

The planner must directly evaluate the **destination-deduplicated Dispatch+Combine bytes** after replication.

Primary structural objective:

```text
minimize future live-only remote D+C bytes
```

not:

```text
maximize number of locally replicated expert hits
```

Reuse the validated remote-byte accounting from `poc_dllm_ep_comm`.

---

## 5. Hardware

Use only:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7
```

Record:

```bash
nvidia-smi
nvidia-smi topo -m
```

Also save GPU UUIDs, logical-to-physical rank mapping, expert ownership, free memory, and relevant library/runtime versions.

Do not terminate or modify other users' processes.

---

## 6. Model and Runtime

Reuse the exact validated LLaDA2.0-mini EP4 environment from the previous PoC whenever possible.

Record:

- exact model revision;
- dInfer revision;
- SGLang/DeepEP/PyTorch/CUDA/Triton versions;
- decoding parameters;
- expert tensor shapes;
- exact bytes per expert per sparse layer.

Do not silently upgrade the stack.

TRUE physical EP4 must be revalidated.

---

## 7. Workloads

Primary continuity workloads:

```text
HumanEval
GSM8K
```

The previous PoC showed:

```text
HumanEval:
Residual_Amp ≈ 4.03×
D+C / wall ≈ 24.78%

GSM8K:
Residual_Amp ≈ 7.52×
D+C / wall ≈ 11.16%
```

It is acceptable to prioritize **communication-exposed** workloads for the method evaluation, especially coding workloads, as long as the target regime is declared before policy evaluation.

If cheap, add one longer-generation coding/structured-generation workload to test:

```text
long output
+
low acceptance yield
+
high D+C share
```

Do not post-select examples because they produce favorable replica gains.

---

## 8. Accuracy and Correctness Are Mandatory

Because exact replication should preserve the mathematical model, the primary gate is stronger than ordinary benchmark accuracy.

### 8.1 Exact-path parity

For deterministic frozen probes compare baseline vs replicated runtime:

- final generated token IDs;
- acceptance schedule;
- number of denoising iterations / NFE;
- router top-k expert IDs;
- router weights within numerical tolerance;
- replica tensor values/hashes against owner tensors.

Expected:

```text
same output tokens
same acceptance decisions
same routing decisions
same NFE
```

If exact token parity is expected by the backend, require it.

A placement-only method must not intentionally trade quality for speed.

### 8.2 Benchmark accuracy

For the tested benchmark cohort also report:

```text
GSM8K exact match
HumanEval deterministic pass@1 / unit-test pass rate
```

for:

```text
baseline
runtime replication prototype
```

Expected quality delta:

```text
0
```

Any real accuracy regression is a correctness failure unless independently explained as nondeterministic evaluation noise.

Offline oracle simulation does not change accuracy by definition.

---

## 9. Reuse Previous Traces First

Inspect `poc_dllm_ep_comm` before rerunning the model.

Prefer reusing existing raw evidence if it includes:

- request/block/iteration/layer;
- token position;
- live/accepted state;
- source EP rank;
- top-k expert IDs;
- destination ranks;
- destination-deduplicated remote bytes.

If information is missing, collect only the minimum additional traces required.

---

## 10. Stage A — Offline Structural PoC

Do not implement runtime replication yet.

All policies are evaluated on `B_live_only`:

```text
already accepted before iteration i -> excluded
live before iteration i             -> included
```

This prevents claiming dead-token savings as part of the new method.

### Policy 0 — No replication

The live-only baseline.

### Policy 1 — Global/static popularity

Use only calibration traffic. This is the generic-replication baseline.

### Policy 2 — First-iteration expert frequency

Use only the first refinement iteration of the current block.

Select experts per:

```text
layer × source rank
```

### Policy 3 — First-iteration destination-aware

Use the first refinement iteration but choose replicas based on predicted destination-closure byte reduction rather than raw expert frequency.

### Policy 4 — First-2-iterations destination-aware

If cheap, observe the first two refinement iterations and plan for the remaining block lifetime.

### Policy 5 — Oracle

May use all future iterations inside the same block. This is nondeployable and provides the upper bound.

No learned predictor in this PoC.

---

## 11. Oracle Optimization

For each request/block, sparse layer, and source rank, choose replicas under a per-rank HBM budget.

Recommended budget sweep:

```text
0.5 GiB/rank
1 GiB/rank
2 GiB/rank
4 GiB/rank
```

Add larger budgets only if physically feasible.

Primary objective:

```text
minimize destination-deduplicated live-only D+C bytes
over future refinement iterations
```

Constraints/penalties:

- replica memory;
- one-time weight copy cost;
- per-rank capacity;
- critical-rank expert-compute load.

Report both:

```text
communication-only oracle
load-aware oracle
```

Do not treat a communication-only plan with severe compute imbalance as deployable.

---

## 12. Main Structural Metric

For each policy define:

```text
PostReplica_Residual_Amp
=
live-only D+C bytes after replica policy
/
B_one_shot
```

The previous reference is:

```text
No replica, live-only = 5.773×
One-shot             = 1.000×
```

Desired example:

```text
Live-only                5.77×
Global + 2 GiB/rank      4.50×
Causal + 2 GiB/rank      2.90×
Oracle + 2 GiB/rank      2.30×
One-shot                 1.00×
```

Do not claim `1.00×` is necessarily executable.

---

## 13. Oracle Recovery

Compute:

```text
OracleRecoverableBytes
= bytes(live-only) - bytes(oracle)

CausalRecoveredBytes
= bytes(live-only) - bytes(causal)

RecoveredOracleHeadroom
= CausalRecoveredBytes / OracleRecoverableBytes
```

Report request-bootstrap confidence intervals.

The causal policy must beat the global/static policy.

---

## 14. Copy Cost and Amortization

Temporary replicas are block-scoped:

```text
block starts
-> observe first iteration(s)
-> copy selected experts
-> reuse for remaining iterations
-> block ends
-> evict/reuse replica slots
```

Measure P2P bandwidth on GPUs 2,3,6,7 using:

- actual expert-sized copies;
- larger bulk copies.

For every block/policy report:

```text
replica bytes copied
copy latency
remaining refinement iterations
remote D+C bytes avoided
communication ROI
break-even remaining iterations
```

Use both a conservative serial estimate and a reasonable parallel/per-edge estimate.

A useful plan must amortize its one-time copy.

---

## 15. Compute-Imbalance Risk

Replication moves expert execution toward source ranks.

Recompute/simulate:

```text
expert assignments per rank
max-rank assignments
max/mean
rank CV
hottest rank
```

for every policy.

If measured expert costs are available, weight assignments by measured per-expert/per-rank cost.

Main Pareto curve:

```text
D+C byte reduction %
vs
critical-rank expert-load increase %
```

Do not accept a policy that saves network bytes but creates a larger compute straggler.

---

## 16. Stage A GO / NO-GO Gates

### Gate A1 — Oracle headroom

At a feasible budget:

```text
oracle live-only D+C reduction >= 35%
```

is a GO signal.

Interpretation:

```text
<20%     NO-GO
20-35%   CONDITIONAL
>=35%    GO
>=50%    strong
```

Prefer:

```text
PostReplica_Residual_Amp <= 3.5×
```

at a feasible budget.

### Gate A2 — Causal planner

Require:

```text
causal > global/static
```

and:

```text
RecoveredOracleHeadroom >= 50%
```

Preferred stronger signal:

```text
>=70%
```

If causal is not better than global, close the block-adaptive story.

### Gate A3 — Copy amortization

Require at least one useful causal point with:

```text
avoided communication benefit / copy cost >= 2×
```

or an equivalent measured timing argument.

### Gate A4 — Compute balance

Preferred:

```text
critical/max-rank expert-load increase <= 10%
```

If useful savings require >20% critical-rank compute increase, mark CONDITIONAL/NO-GO unless timing shows net benefit.

If Stage A fails, do not implement runtime replication.

---

## 17. Stage B — Minimal Exact Runtime Prototype

Only after Stage A passes.

Implement the smallest physical prototype able to:

1. create exact temporary replicas for selected `(layer, expert, destination/source rank)` tuples;
2. route matching expert work to the local replica;
3. keep router outputs/top-k unchanged;
4. keep routed scaling/combine semantics unchanged;
5. fall back to original ownership if no replica exists;
6. free/reuse replica slots at block boundaries.

It is acceptable to prototype:

- selected sparse layers;
- one chosen budget;
- the primary communication-exposed workload;

before building a complete system.

---

## 18. Stage B Baselines

### B0 — Vanilla measured runtime

Current dInfer/DeepEP.

### B1 — Global/static replication

Exact physical replication using a static popularity plan.

### B2 — Block-local causal replication

The proposed exact runtime policy.

### B3 — Structural combined accounting only

Offline combination of:

```text
perfect stable/dead-token elimination
+
block-local causal replication
```

This is not measured Epoch and must not be labeled `Epoch + Ours`.

Actual Epoch can only be added after a faithful implementation/reproduction.

---

## 19. Runtime Performance Metrics

Measure:

```text
Dispatch ms
Expert ms
Combine ms
D+C ms
MoE-path ms
denoising span
request E2E
```

Also report:

```text
D+C reduction %
E2E latency reduction %
speedup
extra HBM / rank
replica copy/preparation ms
```

Use the validated timing methodology from the previous PoC.

Do not infer runtime speedup from byte reduction.

---

## 20. Runtime Correctness Gate

Before any performance claim, require:

```text
same final token IDs
same acceptance schedule
same NFE
same router top-k IDs
same routing weights within tolerance
```

and validate expert replica identity.

If correctness parity fails, performance results are invalid until fixed.

---

## 21. Accuracy Gate

Evaluate benchmark quality on the same fixed cohort:

```text
GSM8K exact match
HumanEval pass@1 / unit-test pass rate
```

Expected:

```text
baseline accuracy == replica accuracy
```

Do not accept an accuracy/speed tradeoff for this method.

---

## 22. Stage B Performance Gate

Preferred promotion target on the communication-exposed workload:

```text
D+C latency reduction >= 30%
E2E latency reduction >= 7%
```

Interpretation:

```text
<5% E2E      weak / likely NO-GO
5-7%         CONDITIONAL
>=7%         GO
>=10%        strong
```

with zero quality loss.

---

## 23. Robustness

If the primary runtime result passes, evaluate a small matrix:

```text
HumanEval (primary communication-exposed)
GSM8K (secondary)

block32
block64 if supported
```

Equal gains across all workloads are not required.

It is acceptable to define the target regime as:

```text
communication-exposed block-wise MoE dLLM inference
```

as long as the paper clearly characterizes when the method helps.

---

## 24. Required Plots

1. **Residual amplification vs replica budget**
   - Global
   - First-iteration frequency
   - First-iteration destination-aware
   - First-2-iteration destination-aware
   - Oracle
   - reference lines: 5.77× live-only and 1.00× one-shot

2. **Recovered oracle headroom vs budget**

3. **D+C byte reduction vs max-rank expert-load increase**

4. **Layer-level replica allocation**
   - layer
   - source rank
   - remote destination
   - replica count

5. **Runtime stage timing**
   - Vanilla
   - Global replication
   - Block-local replication

6. **Accuracy/correctness summary**
   - token parity
   - acceptance parity
   - GSM8K
   - HumanEval

---

## 25. Recommended Main Table Shape

Prepare evidence for a table like:

| Method | Quality | Norm. EP Traffic | D+C ms | E2E ms | Speedup | Extra Mem |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | same | measured | measured | measured | 1.00× | 0 |
| Perfect dead-token elimination | n/a | 5.77× structural | n/a | n/a | n/a | 0 |
| Global replication | same | ... | ... | ... | ... | ... |
| Block-local replication | same | ... | ... | ... | ... | ... |
| Oracle block-local | n/a | ... | n/a | n/a | n/a | ... |
| One-shot reference | n/a | 1.00× | n/a | n/a | n/a | n/a |

Never put `5.77×` in a measured Epoch row.

---

## 26. Statistics

Use requests/problems as independent samples.

Do not bootstrap over layers, iterations, or tokens.

For headline results report:

```text
mean
median
95% request-bootstrap CI
```

for:

- D+C byte reduction;
- PostReplica_Residual_Amp;
- oracle recovery;
- copy ROI;
- max-rank load change;
- measured D+C reduction;
- measured E2E reduction;
- accuracy difference.

---

## 27. Required Outputs

Recommended directory:

```text
poc_dllm_block_local_replica/
├── scripts/
│   ├── reuse_or_extract_traces.py
│   ├── simulate_replication.py
│   ├── destination_closure_oracle.py
│   ├── causal_planner.py
│   ├── estimate_copy_cost.py
│   ├── analyze_load_balance.py
│   ├── runtime_replica_prototype.py
│   ├── validate_correctness.py
│   └── evaluate_accuracy.py
├── results/
│   ├── environment.json
│   ├── oracle_budget.csv
│   ├── causal_budget.csv
│   ├── layer_replica_plan.csv
│   ├── copy_roi.csv
│   ├── load_balance.csv
│   ├── runtime_timing.csv
│   ├── correctness.json
│   ├── accuracy.json
│   └── summary.json
├── figures/
│   ├── residual_vs_budget.png
│   ├── oracle_recovery.png
│   ├── comm_vs_load_pareto.png
│   ├── layer_replica_allocation.png
│   ├── runtime_performance.png
│   └── accuracy_summary.png
└── report.md
```

Do not overwrite `poc_dllm_ep_comm`.

---

## 28. Final Report Questions

The final report must answer directly:

1. How much of the 5.77× live-only residual is oracle-reducible?
2. What residual amplification remains after oracle replication?
3. How much memory per rank is required?
4. How strongly layer-dependent are useful replicas?
5. Does destination-aware selection beat raw expert-frequency selection?
6. Does the causal block policy beat global/static popularity?
7. How much oracle headroom does the causal policy recover?
8. How many remaining iterations are needed to amortize expert-copy cost?
9. Does replication create a critical-rank compute hotspot?
10. What is the best communication/load-balance Pareto point?
11. Was Stage B runtime implementation promoted?
12. If yes, how much measured D+C latency is reduced?
13. If yes, how much measured E2E latency is reduced?
14. Are token IDs, acceptance schedule, routing, and NFE unchanged?
15. Are GSM8K and HumanEval accuracy unchanged?
16. Final verdict: GO / CONDITIONAL / NO-GO.
17. What is the single smallest next experiment?

---

## 29. Decision Logic

### NO-GO

Close if any central condition fails:

```text
oracle reduction < 20%
causal ~= global
copy does not amortize
compute hotspot erases network savings
correctness/accuracy changes
```

### CONDITIONAL

Use if structural headroom is strong but physical runtime robustness remains unresolved.

### GO

Promote if:

```text
oracle D+C reduction >= 35%
causal recovers >= 50% of oracle
causal > global
copy ROI is favorable
critical-rank compute is controlled
exact correctness is preserved
accuracy delta = 0
measured E2E reduction >= ~7%
```

A >=10% measured E2E latency reduction with zero quality loss is a strong result.

---

## 30. Git Hygiene

Preserve all unrelated dirty/staged work.

Suggested branch:

```text
flashvep/dllm-block-local-replica-poc
```

Commit and push only this PoC's files after validation.

Report:

- final verdict;
- key structural and runtime numbers;
- accuracy/correctness result;
- branch;
- commit SHA;
- report path.

---

## 31. Final Research Criterion

Continue only if the full chain holds:

```text
5.77× live-only residual
    ->
large layer-specific repeated destination reuse
    ->
destination-aware block-local replicas remove substantial bytes
    ->
simple causal early-iteration policy captures the oracle
    ->
copy cost amortizes
    ->
compute imbalance remains controlled
    ->
exact outputs and accuracy are preserved
    ->
real D+C and E2E latency decrease
```

Do not rescue a failed signal with a complex learned predictor or approximate routing method.
