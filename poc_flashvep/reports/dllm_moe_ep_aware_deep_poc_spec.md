# Deep PoC Specification: Training-Free EP-Aware Inference for MoE Diffusion LLMs

## 0. Project Goal

This PoC has two goals:

1. Characterize how recent training-free MoE-dLLM inference methods scale from single GPU to EP2 and EP4.
2. Discover and, only if justified by oracle evidence, prototype a new training-free EP-aware inference method for MoE diffusion LLMs.

The target paper-level problem is:

> Existing training-free MoE-dLLM inference policies optimize algorithmic proxies such as denoising iterations, expert budgets, or unique expert counts, but those decisions do not explicitly optimize the physical communication and execution cost induced by multi-GPU Expert Parallelism.

Target preference:
1. reduce total physical work;
2. avoid expensive distributed work before it is created;
3. reduce large communication amplification;
4. co-design algorithmic utility and EP cost;
5. overlap/fuse only when supported by measured critical-path slack;
6. avoid small placement/load-balancing tweaks unless they expose a large E2E opportunity.

---

## 1. Hardware Constraint

Use exactly physical GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

No other GPUs may be used.

- single GPU: physical GPU 0 preferred;
- EP2: physical GPUs 0,1;
- EP4: physical GPUs 0,1,2,3.

Record UUID/topology with `nvidia-smi topo -m`.

---

## 2. Time Budget

The user has >10 hours of four-H100 availability.

Priority:

```text
faithful reproduction
→ single/EP2/EP4 characterization
→ measured mismatch
→ candidate oracles
→ strongest live prototype
```

Do not spend large engineering effort on a candidate whose perfect oracle is weak.

---

## 3. Methods

### TEAM
- official repo: `PKU-SEC-Lab/TEAM-MoE-dLLM`
- native model: `JetLM/SDAR-30B-A3B-Chat-b32`
- prior local result: official single-GPU positive trend already reproduced.

### REFLEX
- paper: `REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models`
- primary model: `LLaDA-MoE-7B-A1B-Instruct`
- training-free variable expert-budget method.

### DES
- paper: `Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs`
- primary model: `LLaDA-MoE-7B-A1B-Instruct`
- primary variant: DES-Vote.

Do not include TIDE in the main study.

Epoch is the principal distributed related-work baseline conceptually, but do not fabricate measured Epoch results if official reproducible code is unavailable.

---

## 4. Common Model / Dataset Policy

There is no common native model across TEAM, REFLEX, and DES.

Do not force all methods onto one model if it destroys reproduction fidelity.

Use:

- TEAM → SDAR-30B-A3B-Chat-b32
- REFLEX + DES → LLaDA-MoE-7B-A1B-Instruct

Common anchor datasets:

- GSM8K
- HumanEval

Add MBPP if time permits.

Compare within each method against its own native vanilla baseline.

Do not compare raw absolute latencies across different model families as if they are directly fair.

---

## 5. Main Characterization Table

Produce:

| Method | Model | Dataset | Setting | Accuracy | Latency | Speedup vs vanilla | NFE Δ | Expert-pairs Δ | Unique-experts Δ | Remote assignments Δ | EP bytes Δ | Rank fanout Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TEAM | SDAR-30B-A3B | GSM8K | single | | | | | | | N/A | N/A | N/A |
| TEAM | SDAR-30B-A3B | GSM8K | EP2 | | | | | | | | | |
| TEAM | SDAR-30B-A3B | GSM8K | EP4 | | | | | | | | | |
| REFLEX | LLaDA-MoE-7B | GSM8K | single | | | | | | | N/A | N/A | N/A |
| REFLEX | LLaDA-MoE-7B | GSM8K | EP2 | | | | | | | | | |
| REFLEX | LLaDA-MoE-7B | GSM8K | EP4 | | | | | | | | | |
| DES | LLaDA-MoE-7B | GSM8K | single | | | | | | | N/A | N/A | N/A |
| DES | LLaDA-MoE-7B | GSM8K | EP2 | | | | | | | | | |
| DES | LLaDA-MoE-7B | GSM8K | EP4 | | | | | | | | | |

Repeat for HumanEval if feasible.

A bounded but clean subset is acceptable, but label it clearly.

---

## 6. Stage 0 — Environment and Source Audit

Create an isolated branch, e.g.:

`flashvep/dllm-ep-aware-policy-deep-poc`

Suggested root:

`poc_dllm_ep_aware/`

Record:
- TEAM commit
- SDAR commit
- dInfer/LLaDA runtime commit
- REFLEX paper revision
- DES paper revision
- vLLM
- DeepEP
- PyTorch
- CUDA
- NCCL
- Transformers
- GPU topology
- model hashes

If REFLEX/DES official code is unavailable, state explicitly:
`paper-faithful minimal reproduction`.

---

## 7. Stage 1A — TEAM Single / EP2 / EP4

Reuse prior validated TEAM single-GPU and EP2 results where faithful.

The important new point is TEAM EP4 on GPUs 0,1,2,3.

Target true topology:
`TP1 / DP1 / EP4`
unless architecture requires another minimal topology.

Verify:
- 128 experts
- expert ownership
- remote dispatch
- local expert execution
- reverse combine
- work conservation

Use fused expert execution.

Measure:
- accuracy / bounded quality
- clean latency
- speedup vs SDAR baseline
- NFE
- accepted tokens
- token-expert assignments
- unique experts
- remote assignments
- remote hidden bytes
- destination-rank fanout
- dispatch
- expert
- combine
- route preparation
- router
- attention

Core scaling:
`single → EP2 → EP4`.

---

## 8. TEAM Questions

Answer:
1. Does TEAM speedup degrade with EP degree?
2. Does speculative exploration amplify remote assignment count?
3. Does EP traffic amplification grow on EP4?
4. Does destination-rank fanout grow?
5. Which stage explains lost single-GPU speedup?
6. Is the loss mainly communication, preparation, fragmentation, or another stage?
7. Is the pathology specific to speculative work creation?

---

## 9. Stage 1B — REFLEX Reproduction

Use `LLaDA-MoE-7B-A1B-Instruct`.

First reproduce single-GPU semantics:
- default fixed top-8
- REFLEX allocation
- AvgK
- expert-token pairs
- bounded GSM8K/HumanEval quality
- real latency when reduced-k execution is physically realized

Paper-faithful reference:
- router ranking preserved
- training-free
- expert budget follows refinement state
- about 15% selected-pair reduction on average in the paper

Do not proceed to EP if method semantics are not reproduced.

---

## 10. REFLEX EP2 / EP4

Run the same decisions under:
- EP2: GPUs 0,1
- EP4: GPUs 0,1,2,3

Measure:
- AvgK
- expert-pairs
- remote assignments
- remote fraction
- remote bytes
- destination-rank fanout
- dispatch
- expert
- combine
- E2E
- accuracy

Key question:
> Does pair reduction translate proportionally to physical EP cost reduction?

Distinguish removing a local marginal expert from removing a remote marginal expert.

---

## 11. Stage 1C — DES Reproduction

Use `LLaDA-MoE-7B-A1B-Instruct`.

Primary variant: DES-Vote.

Reproduce single-GPU structural trend:
- unique experts decrease
- quality approximately preserved
- MoE latency direction improves if kernel support permits

If official code is unavailable:
- implement only a paper-faithful minimal DES-Vote
- separate coreset-selection overhead from ideal/fused selection cost
- do not claim full paper latency reproduction without optimized kernel parity

---

## 12. DES EP2 / EP4

Run the same coreset decisions under EP2 and EP4.

Measure:
- unique experts
- expert-pairs
- remote assignments
- remote bytes
- unique destination ranks
- fanout
- dispatch
- expert
- combine
- E2E
- accuracy

Key question:
> Does unique-expert reduction translate into physical rank-fanout and communication reduction?

---

## 13. Production-Like EP Backend

Prefer production-capable sparse EP execution such as DeepEP where stable.

For TEAM/SDAR, use the best correct production-like path feasible from the prior port.

For REFLEX/DES on LLaDA-MoE, reuse the most stable true-EP runtime available.

Do not use a Python expert loop for final latency claims.

Always include a fused expert control.

If a backend cannot support ragged/dynamic-k routing, record the limitation.

---

## 14. Accuracy Policy

For bounded PoC:

### GSM8K
Exact-answer accuracy on identical samples.

### HumanEval
Executable pass@1 when practical, otherwise native postprocessing path.

Use identical samples across vanilla/method/single/EP2/EP4.

Report:
- absolute accuracy
- delta vs own vanilla baseline

Do not use token-exact identity as the sole quality metric.

---

## 15. Stage 1 Main Scientific Plots

### Plot A
Method speedup vs EP degree:
`single, EP2, EP4`

### Plot B
Algorithmic proxy vs physical EP cost:
- TEAM: NFE/activation change
- REFLEX: AvgK/pairs
- DES: unique experts

versus:
- E2E
- remote bytes
- remote assignments
- fanout

The hypothesis:
> Algorithmic efficiency proxies do not necessarily predict distributed efficiency.

---

## 16. Motivation Gate

Proceed strongly if at least one holds:

### M1
At least two methods show meaningful mismatch between algorithmic proxy and EP4 E2E.

### M2
One method degrades strongly with EP degree while another behaves differently and physical EP cost explains the difference.

### M3
A method already works on EP4, but a perfect physical-cost-aware action oracle has >=10% additional E2E headroom.

If none hold, do not force a universal EP-aware paper.

---

## 17. Action-Level Logging

### TEAM action units
Log candidate/speculative work:
- branch count
- branch identity
- hot/cold treatment
- limited-activation choice
- semantic/confidence utility
- realized acceptance
- expert IDs
- destination ranks
- remote bytes
- fanout

### REFLEX action units
For each token:
- allocated k
- ranked experts
- refinement role
- FPS/ranking signal if available
- marginal experts
- each marginal expert local/remote status
- marginal communication cost

### DES action units
For each block:
- candidate coreset
- saliency/vote score
- coreset size
- physical rank locations
- unique destination ranks
- expected remote payload

---

## 18. Shared EP Cost Features

For any optional action/work unit log:

- remote_assignment_count
- remote_fraction
- destination_rank_count
- dispatch_bytes
- combine_bytes
- critical-rank added work
- estimated dispatch latency
- estimated expert latency
- estimated combine latency
- fragmentation proxy

Find which features explain real marginal cost.

---

## 19. EP Cost Model

Start simple and interpretable.

### C0
`Cost = remote_bytes`

### C1
`Cost = a * remote_bytes + b * fanout`

### C2
`Cost = Tdispatch(shape, fanout) + Texpert(shape) + Tcombine(shape, fanout)`

### C3
rank-wise critical-path cost:
`max(predicted rank time)`

Calibrate on measured EP4 microbenchmarks/traces.

Validate on held-out traces.

This is runtime calibration, not model training.

---

## 20. Utility Model

Initially retain native method utility.

- TEAM → acceptance/confidence/speculation utility
- REFLEX → refinement role/FPS/router order
- DES → vote/saliency

The intended formulation is:
`native semantic utility + shared physical EP cost`.

---

## 21. Candidate Family A — Cost-Aware Re-Ranking

For candidate action `a`:

`Score(a) = Utility(a) - λ * EPCost(a)`

Examples:
- TEAM: prune low-utility high-fanout branch
- REFLEX: skip low-utility marginal remote expert
- DES: choose near-equivalent lower-fanout coreset

Sweep λ and measure quality-latency Pareto.

---

## 22. Candidate Family B — EP-Budgeted Work Shaping

Optimize:

`maximize Utility(actions) subject to EPCost(actions) <= budget`

Budgets may use:
- bytes
- destination ranks
- critical-rank predicted time

Evaluate oracle and causal variants.

---

## 23. Candidate Family C — Marginal Utility / EP Cost

For optional work:

`value = marginal algorithmic utility / marginal EP cost`

Execute highest-value units until threshold/budget.

Mapping:
- TEAM: speculative branch
- REFLEX: marginal expert above minimum k
- DES: expert added to coreset

This is a promising common formulation.

---

## 24. Candidate Family D — Topology-Aware Action Selection

Use physical expert placement/topology.

TEAM:
two semantically similar branches may have different fanout.

REFLEX:
marginal experts may be local vs remote.

DES:
similar coresets may occupy different numbers of ranks.

Novelty must be in dLLM work-creation/allocation decisions, not generic rerouting after work is fixed.

---

## 25. Candidate Family E — EP-Aware Speculation Width

Required TEAM-focused candidate.

Estimate:

`ExpectedTotalTime(k) = ExpectedNFE(k) * ExpectedDistributedForwardCost(k)`

Test available speculation widths, e.g. k=1,2,3,4.

Measure:
- NFE
- accuracy
- remote bytes
- fanout
- E2E

Single-GPU optimal width may differ from EP4 optimum.

---

## 26. Candidate Family F — Joint Decode-EP Perfect Oracle

Construct a future-aware offline oracle that chooses the action minimizing total future E2E while satisfying a quality/correctness target.

This is only for upper-bound estimation.

If the oracle <5%, kill the action family.
If >10–15%, build a causal approximation.

---

## 27. Candidate Competition

After Stage 1:

1. build action-level traces
2. validate cost model
3. evaluate A–F or equivalent
4. compute perfect E2E oracles
5. prior-art attack
6. rank candidates

Table:

| Candidate | TEAM oracle | REFLEX oracle | DES oracle | Robustness | Quality risk | Prior-art risk | Complexity |
|---|---:|---:|---:|---|---|---|---|

---

## 28. Candidate Gates

- <5% perfect E2E → kill
- 5–8% → weak
- 8–12% → promising
- >=12% → strong

For a general paper direction prefer:
- >=10% oracle on at least two method/model settings, or
- >=15% on one and clean generalization on another.

Do not build a complex controller for a 3% oracle.

---

## 29. Quality Constraint

For each candidate compare:

- accuracy vs vanilla
- accuracy vs original method
- latency vs vanilla
- latency vs original method

Preferred:
`Original method + EP-aware policy`
with similar quality and lower EP4 latency.

---

## 30. Training-Free Definition

Allowed:
- runtime profiling/calibration
- deterministic cost model
- hardware latency tables
- lightweight latency-predictor calibration
- threshold/lambda calibration

Not allowed for primary claim:
- LM retraining
- router/model fine-tuning
- RL policy training that changes model behavior

---

## 31. EP4 Microbenchmark Surface

Build an EP4 cost surface over:
- rows/assignments
- remote fraction
- fanout = 1,2,3
- expert work size

Measure:
- dispatch
- expert
- combine
- total

Use production-like backends.

---

## 32. EP Degree Scaling

For strongest candidate compare:
- single
- EP2
- EP4

Do not infer EP4 from EP2.

---

## 33. Epoch Boundary

Do not claim measured superiority to Epoch without reproducible code.

Conceptual distinction:

Epoch:
> efficiently executes a compact/fresh required worklist under EP.

Candidate here:
> decides which optional dLLM inference work is worth creating under EP cost.

Run sensitivity analysis assuming communication becomes faster:
- ×1.0
- ×0.75
- ×0.5
- ×0.25

If candidate benefit disappears under 2× faster communication, note durability risk.

---

## 34. Prior-Art Audit

Compare strongest candidate against:
- TEAM
- REFLEX
- DES
- Epoch
- generic communication-aware MoE routing
- expert placement/scheduling
- communication-aware pruning
- speculative dLLM inference

Novelty target:
> EP cost integrated into dLLM-specific work-creation/allocation decisions.

Not:
> communication-aware MoE.

---

## 35. Overlap Exploration

The user is interested in overlap.

Only explore after measured profiling.

Potential windows:
- speculative branch compute overlapping another branch's EP communication
- attention of independent branch overlapping communication
- cross-request independent work

Calculate an overlap oracle first.

- <5% E2E → kill
- >10% → serious candidate

Audit DICE/staleness-related prior art.

---

## 36. Cross-Request Opportunities

Only after per-request work shaping.

Optionally test:
- coalescing small communications
- shared coresets
- batching similar optional work
- smoothing speculative fanout

Do not let scheduling distract from the main problem without large oracle evidence.

---

## 37. Required Reports

Create:

- `reports/environment_and_runtime.md`
- `reports/team_single_ep2_ep4.md`
- `reports/reflex_single_ep2_ep4.md`
- `reports/des_single_ep2_ep4.md`
- `reports/cross_method_characterization.md`
- `reports/action_level_ep_cost.md`
- `reports/candidate_oracles.md`
- `reports/prior_art_audit.md`
- `reports/best_candidate_prototype.md`
- `reports/final_decision.md`

---

## 38. Required Plots

At minimum:

1. TEAM speedup vs EP degree
2. REFLEX speedup vs EP degree
3. DES speedup vs EP degree
4. Accuracy delta vs EP degree
5. Remote assignments vs EP degree
6. Remote bytes vs EP degree
7. Rank fanout vs EP degree
8. Algorithmic proxy improvement vs actual E2E
9. EP4 stage breakdown per method
10. Action utility vs EP cost
11. EP-cost predictor accuracy
12. Candidate perfect-oracle E2E comparison
13. Quality-latency Pareto
14. Faster-communication sensitivity
15. Original vs +EP-aware policy for implemented candidates

---

## 39. Characterization Success

Interesting outcomes may differ by method.

Example:
- TEAM: algorithmic gain good, EP traffic worse
- REFLEX: pair reduction also reduces EP traffic
- DES: unique expert reduction reduces fanout

This diversity is valuable.

Target conclusion:
> algorithmic proxies have different relationships to real distributed cost.

---

## 40. Method Discovery Success

Ideal:
- TEAM + ours improves
- REFLEX + ours improves
- DES + ours improves

But the method does not need identical code paths across all three if:
- the common cost formulation is shared
- at least two distinct policy families benefit

---

## 41. Final Labels

### CHARACTERIZATION-ONLY
Scaling behavior is interesting but no strong new candidate.

### HOLD-CANDIDATE
>=8% oracle with plausible novelty; live evidence incomplete.

### STRONG-CANDIDATE
>=12% oracle and live EP4 prototype has meaningful gain with bounded quality.

### NO-GO
No meaningful physical-cost-aware headroom after production controls.

---

## 42. Final Questions

The final report must answer:

1. How do TEAM, REFLEX, DES scale single→EP2→EP4?
2. Which algorithmic proxy best/worst predicts EP latency?
3. Does TEAM communication amplification worsen on EP4?
4. Do REFLEX pair reductions reduce remote bytes?
5. Does DES coreset reduction reduce rank fanout?
6. Is there a common EP cost model that predicts these outcomes?
7. Can that model improve inference decisions without training?
8. Which candidate has the largest realistic E2E opportunity?
9. Does the candidate remain useful under faster Epoch/DeepEP-like communication?
10. Is there enough evidence for a paper-level method and later EP8 validation?

---

## 43. Final Instruction

Do not optimize for a positive result.

Use the available 4-GPU time to search multiple method families deeply before concluding NO-GO.

Workflow:

`reproduce native methods → characterize single/EP2/EP4 → expose mismatch → build EP cost model → search candidates → oracle competition → implement strongest candidate only`

Target paper framing if evidence supports it:

`Training-Free EP-Aware Inference for MoE Diffusion Language Models`

TEAM, REFLEX, and DES should serve as representative inference-policy families; Epoch should serve as the principal distributed related-work baseline.
