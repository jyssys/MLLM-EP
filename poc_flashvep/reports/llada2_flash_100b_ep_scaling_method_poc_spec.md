# Deep PoC Specification
# Large-Scale dLLM-MoE Expert Parallel Inference on LLaDA2.0-Flash 100B

## 0. Research Goal

The previous SDAR study established an important boundary:
- SDAR-30B-A3B fits on one H100.
- Under the exact reference runtime, EP2 and EP4 were slower than single GPU.
- Therefore, that substrate did not represent a regime where Expert Parallelism (EP) is required for capacity.

This PoC intentionally changes the regime.

Primary target:
> LLaDA2.0-Flash 100B, a large sparse MoE diffusion language model where multi-GPU execution is practically meaningful.

Goals:
1. establish a correct, production-like multi-GPU EP substrate for a genuinely large dLLM-MoE;
2. characterize latency/throughput versus topology and batch/concurrency;
3. identify the true large-scale bottleneck;
4. discover a training-free latency/throughput optimization;
5. use previous SDAR/TEAM results only as guidance, not assumptions.

Target paper framing if successful:
> Training-Free Efficient Expert-Parallel Inference for Large MoE Diffusion Language Models

---

## 1. Hardware

Use only physical GPUs 0,1,2,3.

- Single GPU only for small auxiliary microbenchmarks if the full 100B model cannot fit.
- Multi-GPU model experiments: GPUs 0,1,2,3.

Record:
- `nvidia-smi -L`
- `nvidia-smi topo -m`
- GPU UUIDs

No other GPU may be used.

---

## 2. Primary Model Facts

Use:
`inclusionAI/LLaDA2.0-flash`

Before execution, verify the downloaded config revision and record it.

Expected official configuration:
- model type: MoE diffusion language model
- total parameters: ~100B
- active parameter class: ~6B
- hidden layers: 32
- routed experts: 256
- experts selected per token: top-8
- shared experts: 1
- hidden size: 4096
- MoE intermediate size: 1024
- router groups: 8
- top-k groups: 4
- context length: 32,768
- recommended block length: 32

Do not rely on memory: dump and archive the actual config used.

---

## 3. Correct Interpretation of EP

Do not require:
> EP must beat single-GPU latency.

For a 100B sparse model, EP should be studied as:
- capacity scaling;
- sparse execution;
- throughput scaling;
- avoiding CPU/offload where possible;
- distributing expert weights and expert computation.

Research questions:
1. Can the 100B model be served fully on 4×H100 without CPU expert offload?
2. What 4-GPU topology gives the best latency/throughput?
3. At what batch/concurrency does EP amortize communication?
4. Which stages dominate at small versus large batch?
5. Which dLLM-specific repeated/refinement work is worth optimizing?

---

## 4. Runtime Priority

Prefer the most production-like official execution path.

Priority:
1. official LLaDA2 / dInfer / SGLang path;
2. true EP using a supported production sparse MoE backend;
3. DeepEP or another production sparse A2A backend if compatible;
4. reference NCCL only as correctness fallback.

Do not use Python per-expert loops for headline latency.

A mandatory first task is to determine whether the official 4-GPU path is:
- TP4;
- EP4;
- TP+EP aligned;
- or another topology.

Do not assume multi-GPU implies EP.

---

## 5. Stage 0 — Runtime / Topology Audit

For each feasible configuration record:
- TP degree
- EP degree
- DP degree
- sequence parallel degree if any
- expert ownership per rank
- dense/non-expert placement
- shared expert placement
- router placement
- A2A backend
- fused expert kernel

Verify true EP:
- 256 routed experts partitioned across ranks;
- expected expert IDs per rank;
- remote hidden dispatch;
- owner-rank expert execution;
- reverse combine.

Capture profiler/runtime evidence.

Do not label simple TP weight sharding as EP.

---

## 6. Stage 1 — Correctness / Quality Baseline

Establish one quality-valid baseline before timing sweeps.

Begin from official generation recommendations where applicable:
- temperature = 0.0
- block_length = 32
- steps = 32

Use bounded identical samples:
- GSM8K
- HumanEval
- optional MBPP

Measure:
- accuracy/pass@1
- output length
- NFE
- accepted tokens per iteration
- numerical consistency between semantically equivalent topologies where expected

Do not continue with performance conclusions if topology changes break quality.

---

## 7. Stage 2 — Topology Characterization

Because a 100B model may not have a practical single-GPU baseline, compare feasible 4-GPU decompositions.

Audit and benchmark if supported:
- TP4
- EP4
- TP2+EP2
- runtime-specific aligned TP/EP hybrid

Do not force unsupported combinations.

For each:
- HBM per rank
- model load success
- request latency
- throughput / tokens per second
- NFE
- attention
- router
- route preparation
- dispatch
- expert
- combine
- idle/synchronization
- communication bytes
- destination fanout
- rank-load imbalance
- expert kernel sizes

Primary question:
> Under the same 4×H100 budget, which parallel decomposition best serves the 100B sparse dLLM?

---

## 8. Stage 3 — Batch / Concurrency Scaling

Mandatory.

Do not benchmark only batch=1.

Sweep as far as memory/runtime allows, for example:
- 1
- 2
- 4
- 8
- 16
- 32

If runtime uses continuous batching, use controlled concurrent requests.

For every point measure:
- per-request latency
- p50/p95
- aggregate throughput
- tokens/s
- NFE
- attention share
- MoE share
- dispatch
- expert
- combine
- bytes
- communication call count
- message size distribution
- expert GEMM shapes
- GPU utilization
- rank imbalance

Create curves.

Hypothesis to test, not assume:
- low batch: startup/communication heavy
- larger batch: expert GEMMs and sparse compute amortize communication better

---

## 9. Stage 4 — Denoising-Phase Characterization

Split decoding into:
- early
- middle
- late

For each phase and batch:
- active masked positions
- accepted positions
- physical MoE rows
- token-expert pairs
- unique experts
- fanout
- rank-load CV
- dispatch/expert/combine
- GEMM shapes
- route persistence

Ask:
> Does denoising phase change which EP bottleneck dominates?

---

## 10. Stage 5 — Temporal Structure at Large Scale

Repeat temporal analysis on LLaDA2.0-Flash; do not inherit SDAR conclusions.

For lag 1/2/4/8:
- route Jaccard
- same-expert probability
- destination-rank agreement
- rank-load cosine
- critical-rank persistence
- router-score similarity
- hidden cosine / rel-L2
- expert-output similarity

Condition on:
- batch
- layer
- denoising phase
- confidence
- accepted vs masked tokens

Routing persistence alone is insufficient for reuse claims.

---

---

## 10A. Secondary Characterization — TEAM-Style Single/EP2/EP4 Scaling on LLaDA2.0-Flash

This experiment is valuable, but it must **not block the primary vanilla large-model EP study**.

The goal is not to assume that TEAM degrades under EP on LLaDA2.0-Flash.

The goal is to test whether the previously observed SDAR phenomenon transfers to a genuinely large sparse dLLM:

```text
strong algorithmic gain on a local/small setting
→ weaker or negative gain as EP degree / distributed cost increases
```

### Important execution caveat

TEAM's released implementation is SDAR-specific.

Therefore:

1. first audit whether TEAM's mechanism can be reproduced faithfully on LLaDA2.0-Flash without changing its intended semantics;
2. do not call a custom approximation "TEAM" if the official algorithmic contract cannot be preserved;
3. if a faithful port is not practical, record `TEAM-PORT-NOT-ESTABLISHED` and continue the vanilla study.

TEAM transfer is a secondary experiment, not a substrate gate.

### Preferred experiment

If a faithful TEAM-style implementation is established, compare the same model / prompts / generation configuration under the closest feasible topology series.

Ideally:

```text
TEAM-style local/non-EP reference
TEAM + EP2
TEAM + EP4
```

However, because LLaDA2.0-Flash 100B may not fit on one H100, a literal single-GPU experiment may be impossible.

In that case, use the closest capacity-feasible hierarchy, for example:

```text
lowest-distributed-cost feasible topology
→ EP2-like / hybrid
→ EP4
```

and state clearly that it is **not a literal single→EP2→EP4 comparison**.

Do not use CPU/offload as the "single" point merely to manufacture a scaling curve.

### Required metrics

For vanilla and TEAM-style policy at each feasible topology:

```text
accuracy / pass@1
latency
throughput
NFE
accepted tokens / iteration
token-expert assignments
activated experts
unique experts
remote assignments
remote bytes
fanout
dispatch
expert
combine
router / preparation
```

Report both:

```text
absolute runtime
and
TEAM-style speedup versus matched vanilla at the same topology
```

Example table:

| Topology | Vanilla latency | TEAM latency | TEAM speedup | NFE Δ | Expert-pair Δ | Remote bytes Δ | Fanout Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| lowest-cost feasible | | | | | | | |
| EP2 / hybrid | | | | | | | |
| EP4 | | | | | | | |

### Interpretation cases

#### Case T1 — degradation reproduces

Example:

```text
low-distributed-cost topology: TEAM 1.6x
EP2:                       TEAM 1.3x
EP4:                       TEAM 1.0x
```

with increased remote work / fragmented sparse waves.

This would support a broader observation:

> A dLLM inference policy that is algorithmically efficient can lose its benefit when the same decisions create more expensive distributed sparse work.

This becomes a strong motivation for any new large-scale EP-native method.

#### Case T2 — TEAM remains strong on EP4

Example:

```text
EP4 TEAM speedup >= 1.4x
```

Then the SDAR reversal was not a general property of TEAM.

Investigate why:
- larger expert work amortizes communication;
- production backend absorbs the larger sparse forward;
- higher batch improves kernel utilization;
- topology is different.

This is also valuable because it identifies the regime where TEAM and EP are compatible.

#### Case T3 — TEAM becomes better as batch grows

This is particularly important.

For each supported topology, sweep the same batch/concurrency points as vanilla.

Measure whether:

```text
TEAM EP penalty at batch 1
→ disappears or reverses at batch 8/16/32
```

If so, the key research variable is not only EP degree but:

```text
policy × EP degree × batch/workload scale
```

### Main-paper role

TEAM should remain:

```text
motivation / transfer baseline / composition target
```

not the main method substrate.

The main new method must still improve **vanilla LLaDA2.0-Flash EP** independently.

A particularly strong final experiment would be:

```text
Vanilla
TEAM-style policy
Ours
TEAM-style policy + Ours
```

on the same EP4 large-model regime.

This tests whether the new method is complementary rather than a TEAM-specific fix.


## 11. Stage 6 — Production Bottleneck Table

For representative regimes produce:

| Regime | Attention | Router/Prep | Dispatch | Expert | Combine | Other | Dominant bottleneck |
|---|---:|---:|---:|---:|---:|---:|---|
| B1 | | | | | | | |
| B4 | | | | | | | |
| B8 | | | | | | | |
| B16 | | | | | | | |
| B32 | | | | | | | |

Let this table determine the candidate direction.

---

## 12. Method Search Rule

Only pursue a candidate if:
1. target stage has meaningful production E2E share;
2. there is a dLLM/MoE/EP-specific structural reason;
3. request-level or throughput-level oracle is meaningful;
4. novelty survives prior-art audit.

Oracle gates:
- <5% E2E: KILL
- 5–8%: WEAK
- 8–12%: PROMISING
- 12–20%: STRONG
- >20%: VERY STRONG

Prefer >=12% oracle before expensive implementation.

---

## 13. Candidate A — Denoising-Aware Expert Work Packing

At larger batch, bottleneck may be fragmented expert GEMMs rather than raw bytes.

Explore:
- same-expert coalescing across requests;
- expert-major packing;
- merging small sparse waves;
- exact metadata reconstruction.

Oracle:
- current GEMM shape distribution
- ideal coalesced same-expert GEMMs
- kernel efficiency gain
- waiting cost
- E2E / throughput impact

Novelty must exploit dLLM refinement cadence, not generic MoE batching alone.

---

## 14. Candidate B — Refinement-Aware EP Microbatch Coalescing

Independent requests may be in different denoising iterations but create compatible sparse work.

Coalesce:
- request A iteration t
- request B iteration t+3
- request C iteration t+1

into fewer larger EP waves when latency budget permits.

Target:
- A2A startup
- expert GEMM efficiency
- combine launch count

Build zero-wait and bounded-wait oracles.

---

## 15. Candidate C — Block-Scoped Expert Residency / Replica Planning

Revisit replication only in this larger traffic regime.

Use block temporal persistence to:
- identify hot remote experts;
- replicate a small set under HBM budget;
- retain replica for one diffusion block;
- refresh only on block boundary or strong distribution shift.

Measure:
- hot expert persistence
- remote-demand coverage
- saved critical-path time
- copy cost
- HBM overhead

Strong prior-art risk: kill if it becomes generic replication.

---

## 16. Candidate D — Denoising-Aware Load Shaping

Only if large-batch traces show material rank stragglers.

Possible actions:
- near-tie expert reassignment with bounded router-score loss;
- hot replica for overloaded rank;
- phase-specific balancing.

Optimize max-rank latency, not total assignment count.

Previous small-model straggler results were weak, so require a new strong signal.

---

## 17. Candidate E — Phase-Adaptive Parallel Topology

For a 100B model, EP1 may be infeasible, but different multi-GPU decompositions may still win by phase.

Example:
- early/high-work: EP-heavy topology
- later/smaller-work: TP/EP hybrid

Only pursue if weights/state can remain resident without expensive migration.

Build perfect per-phase topology oracle first.

---

## 18. Candidate F — Fresh-Work Compaction Residual

Epoch strongly occupies fresh/live work compaction.

Still inspect whether the chosen production runtime processes:
- accepted/stable positions
- dead rows
- redundant padding
- unnecessary full-sequence MoE work

If large residual exists, compare explicitly with Epoch mechanisms.

Do not reinvent Epoch.

Proceed only for a clearly different residual.

---

## 19. Candidate G — Router / Dispatch Decoupling and Prefetch

If large-batch routing is temporally predictable and route-prep/dispatch is significant:

explore:
- pre-post receives
- prebuild send offsets
- pre-allocate communication buffers
- prefetch expert metadata
- next-step or next-layer route-plan speculation

Measure:
- route prediction accuracy
- how early the prediction is known
- route-prep/dispatch E2E share
- perfect prefetch oracle

Audit speculative routing / expert prefetch prior art.

---

## 20. Candidate H — Communication / Compute Pipeline

Only if a production Nsight timeline shows independent work.

Possible overlap across requests:
- A2A of one ready sparse wave
- compute/attention/router work of another request
- combine of one wave with another independent request

Do not violate single-request dependencies.

Compute perfect overlap oracle first.
Kill <5%.

Audit DICE and generic MoE overlap work.

---

## 21. Candidate I — Batch-Aware Expert Locality Scheduling

Revisit request grouping only if large-model traces show:
- route diversity across requests;
- load-to-latency correlation;
- grouping can materially change max-rank load or kernel shape.

Otherwise kill immediately.

---

## 22. Candidate J — Expert-Support-Aware Request Grouping

Group requests/blocks with similar expert working sets to:
- make same-expert GEMMs larger;
- reduce launch/metadata overhead;
- improve expert locality.

Measure:
- cross-request expert-set similarity
- GEMM size growth
- A2A call reduction
- latency/throughput tradeoff

Prior-art audit required.

---

## 23. Candidate K — Denoising-Phase Precision / Kernel Adaptation

If sensitivity varies by denoising phase:
- lower precision expert compute
- lower precision communication payload
- exact precision for sensitive phases

Requires a quality-latency Pareto.

Audit REFLEX/ReaLB/quantized MoE serving work.

---

## 24. Candidate Tournament

Create:

| Candidate | Target bottleneck | Perfect E2E/throughput oracle | Quality risk | Prior-art risk | Complexity | Faster-backend durability | Decision |
|---|---|---:|---|---|---|---|---|

Do not implement all candidates.

Implement only strongest one or two.

---

## 25. Workload-Regime Requirement

Every candidate must name its target regime:
- batch-1 latency
- batch-8 interactive serving
- batch-32 throughput
- etc.

Do not average away opposite trends.

---

## 26. Quality Policy

Use identical prompts/settings for system variants.

Primary:
- GSM8K
- HumanEval

Optional:
- MBPP

Measure:
- accuracy/pass@1
- latency
- throughput
- generated length
- NFE

For exact system transforms, verify numerical equivalence.

---

## 27. TEAM as Secondary Adaptation

Only after a vanilla LLaDA2 method succeeds.

TEAM was designed for SDAR, so do not force a TEAM port if execution semantics are unavailable.

If a faithful TEAM-like policy becomes available, compare:
- vanilla
- existing policy
- ours
- policy + ours

Main claim remains independent of TEAM.

---

## 28. Epoch Comparison

Epoch is a major large distributed dLLM-MoE baseline conceptually.

Until official reproducible code exists:
- do not fabricate measured Epoch results;
- compare mechanisms and published results only;
- use sensitivity analysis.

If official code appears, reproduce it on the same model/hardware before superiority claims.

Prefer methods complementary to Epoch.

---

## 29. Faster-Backend Durability

For candidates depending on communication/runtime cost, test sensitivity:
- cost ×1.0
- ×0.75
- ×0.5
- ×0.25

If practical, compare more than one real backend.

Reject fragile gains that vanish under modest backend improvement unless that backend is unavailable in the target deployment.

---

## 30. Required Reports

Create:
- `reports/00_environment_model_runtime.md`
- `reports/01_topology_correctness.md`
- `reports/02_topology_performance.md`
- `reports/03_batch_scaling.md`
- `reports/04_denoising_phase_breakdown.md`
- `reports/05_temporal_structure.md`
- `reports/06_production_bottlenecks.md`
- `reports/07_candidate_oracles.md`
- `reports/08_prior_art_audit.md`
- `reports/09_best_candidate_live_poc.md`
- `reports/10_quality_latency_throughput.md`
- `reports/11_policy_adaptation.md`
- `reports/final_decision.md`

---

## 31. Required Figures

1. HBM usage by topology
2. latency by topology
3. throughput by topology
4. latency vs batch
5. throughput vs batch
6. stage share vs batch
7. dispatch/expert/combine vs batch
8. expert GEMM shape distribution vs batch
9. rank imbalance vs batch
10. denoising-phase workload
11. temporal route persistence
12. candidate oracle comparison
13. quality-latency Pareto
14. faster-backend sensitivity
15. best method vs baseline across batch regimes

---

## 32. Final Labels

- `SUBSTRATE-FAIL`
- `CHARACTERIZATION-ONLY`
- `HOLD-CANDIDATE`
- `STRONG-CANDIDATE`
- `VERY-STRONG-CANDIDATE`

Definitions:
- SUBSTRATE-FAIL: cannot establish correct/quality-valid large-model production-like EP
- CHARACTERIZATION-ONLY: substrate works but no candidate >=8% credible oracle
- HOLD-CANDIDATE: >=8% oracle and plausible novelty
- STRONG-CANDIDATE: >=12% oracle and live >=8% improvement
- VERY-STRONG-CANDIDATE: >=15% live E2E or strong throughput gain across meaningful regimes, preserved quality, durable under production backend sensitivity

---

## 33. Final Questions

1. What exact parallel topology does the official 4-GPU runtime use?
2. Can true EP4 serve LLaDA2.0-Flash without CPU expert offload?
3. How do TP4, EP4, and feasible hybrids compare?
4. At what batch/concurrency does EP become economically favorable?
5. What dominates at low vs high batch?
6. Does denoising phase change EP economics?
7. Does large-scale temporal routing persistence expose useful headroom?
8. Which candidate has the strongest oracle?
9. Is the candidate training-free and independent of TEAM?
10. Is it complementary to Epoch?
11. Does it survive faster backends?
12. Is EP8/larger-scale validation justified?

---

## 34. Final Instruction

Do not interpret the previous SDAR result as:
> dLLM-MoE should not use EP.

Interpret it as:
> a smaller model that fits on one H100 was the wrong substrate for discovering large EP-system gains.

This PoC must study:
`large sparse capacity + genuinely useful multi-GPU execution + batch-dependent EP economics`.

Only after that should the next paper method be selected.
