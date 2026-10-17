# Recurrent Memory–Compute Asymmetry Discovery
## Measurement-first Feasibility Study for New Multi-GPU Parallelism in Looped Language Models

**Status:** Research PoC specification — no experimental results are claimed.  
**Date:** 2026-10-17  
**Target model:** `ByteDance/Ouro-1.4B-Thinking` (default); preserve checkpoint revision  
**Precision:** BF16, no weight/activation quantization  
**Physical GPUs permitted:** NVIDIA GPU indices **4, 5, 6, 7 only**  
**Primary mode:** Exact **fixed R4** (24 shared transformer layers executed across four recurrent steps); prefill and autoregressive decode measured separately  
**Deliverables:** Reproducible traces and metrics, bottleneck decomposition, two architecture feasibility studies, cost-aware oracles, explicit GO/NO-GO decisions

---

## 0. Executive decision and scope

This is **one broad, measurement-first discovery PoC** containing **two explicitly prioritized, focused feasibility tracks**:

- **Track A (priority 1): Recurrent KV-aware Distributed Attention.** Ask whether exact, distributed attention and recurrent KV layout/prefetch can yield useful gains, particularly for long-context, low-concurrency decode, beyond conventional context-parallel execution.
- **Track B (priority 2): Depth–Sequence Wavefront Parallelism.** Ask whether an **exact-dependency** pipeline over recurrent depth × prefill token chunks can reduce genuine bubbles and critical-path time, beyond conventional pipeline scheduling and chunked prefill.

Both tracks are **discovery/oracle studies**, not permission to build either distributed engine. Do not choose a winning architecture in advance. If broad profiling reveals that neither track addresses a substantial bottleneck, terminate early or report an alternative bottleneck with evidence. **Priority A is not presumed to be faster than B.**

**Out of scope:** predictive/stale hidden states; approximate loop execution; activation prediction; RL policies; speculative decoding; approximate KV compression/sharing; training; implementing a production PP/CP engine; fabricating accuracy or speedup claims. Dynamic-depth serving is optional, gated, and explicitly separated from exact R4 results.

**Research question:** Does weight reuse across recurrent depths, combined with per-depth KV and exact causal dependencies, create a **measurable, looped-specific memory/compute/communication asymmetry** that supports a new, faster multi-GPU execution architecture under realistic overheads?

**Minimum successful outcome:** A defensible, reproduced **negative** result is success as discovery. Engineering a new parallelism framework without evidence is **not** success.

---

## 1. Hypotheses (must be tested, not assumed)

**H1. Recurrent memory/compute asymmetry:** increasing recurrent depth increases computation and logical per-depth KV while physical model parameters remain tied. Determine whether this difference materially changes latency, memory traffic and optimal GPU partitioning. It is not sufficient to restate parameter counts.

**H2. Loop-specific KV/attention bottleneck:** long-context decode attention and depth-indexed KV storage/access may become sufficiently costly that distributing exact attention and/or pre-positioning KV is useful; otherwise conventional on-device execution wins.

**H3. Exact wavefront opportunities:** prefill token chunks and recurrent depth induce a two-dimensional DAG with tasks that may overlap **without stale or predicted activations**. Any proposed overlap must satisfy *all* layer, recurrent and causal-KV dependencies.

**H4. Real gain survives communication:** model/trace-based overlap benefits remain nontrivial after measured activation transfers, synchronization, collectives, extra launches, loss of GEMM efficiency, KV traffic, pipeline fill/drain and scheduling overhead.

**H5. Looped-specific delta:** gains are not explained entirely by ordinary pipeline parallelism (PP), context parallelism (CP), tensor parallelism (TP), chunking or simple request-level data parallelism (DP). An apparently good result with no loop-specific advantage must be classified as **GENERIC_ONLY**.

---

## 2. Prior experiment boundary and non-negotiable correctness

Prior Predictive Asynchronous Recurrent Pipeline PoC (reported by user): exact baseline verified for logits/KV/tokens; free-running greedy on 16 GSM8K and 16 MATH-500 items found catastrophic quality degradation with stale, delta, or warm-up-zero prediction. Therefore this PoC **must preserve exact recurrent computation**, avoiding the earlier failure mode. This document does not re-open that direction or request additional predictor ablations.

### 2.1 Reference model audit

1. Reuse the previous PoC's **validated exact R4 implementation and checkpoint**, when reproducible; verify code/checkpoint revision, tokenizer/chat template, Transformers/CUDA/torch versions, EOS semantics and generation parameters. Do not blindly assume the current upstream checkout matches the validated revision.
2. Inspect the *actual installed implementation*, config and KV indexing. The published Thinking config specifies `num_hidden_layers=24`, `total_ut_steps=4`, `hidden_size=2048`, `num_attention_heads=num_key_value_heads=16`, `head_dim=128`, with 4K training context and a longer configured maximum. These are initial expectations, not substitutes for runtime inspection.
3. Record exact prelude / recurrent core / exit gate / coda, and whether the runtime physically skips recurrent computation after an exit decision. **Gate-predicted depth is not proof of executed-work savings.**
4. Keep **depth-indexed KV** for exact R4 and audit each `(loop, physical_layer, request, position)` read/write mapping. No single-slot overwrite or cross-loop aliasing in the exact baseline.
5. Compare baseline token stream and logits (or agreed numerical tolerance) to the validated implementation on deterministic short prefills, one-token decode, and free-running greedy requests. Report actual matching criteria and failures.
6. Before any new experimental kernel path, prove exact output equivalence: matched token IDs and accuracy checks, measured logit differences with tolerance, and correct KV states across several generated tokens.
7. No `device_map='auto'` that silently occupies GPUs 0–3; no silent CPU offload in the primary benchmark. Do not treat a different attention backend or KV manager as an exact apples-to-apples runtime until validated.

### 2.2 Quality guardrail

No full 1,819-problem accuracy campaign is required for a **profiling-only** study that does not modify arithmetic or scheduling. Nevertheless:
- Run a fixed smoke/regression suite from the previously validated exact R4 prompts and report **matched/total tokens**, max/mean logit delta and **actual correct/total** on a small fixed GSM8K/MATH-500 subset when executing new code paths; label small-sample estimates clearly.
- If any real execution reordering/partitioning is implemented, establish token/answer consistency against exact R4 and inspect KV position/masking before reporting speedup.
- In a proposed architecture that preserves mathematical operation order but changes floating-point reductions, report tolerances rather than promising bitwise equality.

---

## 3. Machine setup and safety

**Allowed physical GPU indices:** `4,5,6,7`; `CUDA_VISIBLE_DEVICES=4,5,6,7` maps these to **process-local CUDA 0,1,2,3**. Always log both maps. Never touch other GPUs, users' processes or shared system configuration.

Initial inventory (read-only):

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
nvidia-smi -i 4,5,6,7 --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
nvidia-smi topo -m
nvidia-smi -i 4,5,6,7 -q | grep -E 'Product Name|Bus Id|Driver Version|Compute Mode' -A1
python -c 'import torch; print(torch.__version__, torch.version.cuda); print([(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])'
```

- Check PCIe/NVLink (or equivalent peer path), intra-host NUMA/CPU topology, NCCL/P2P availability, shared contention, power/clock behavior, and other GPU tenants **without killing processes**.
- Pin the CUDA mask in **every** profiler process, launcher, subprocess, distributed rank and benchmark command. Confirm actual device UUIDs after initialization. On conflicts, skip the affected run and log `RESOURCE_BLOCKED`.
- Prefer pre-existing pinned environment; never upgrade PyTorch, transformers, CUDA, NCCL or drivers globally. Save `pip freeze`, git SHA, GPU topology, model revision and all commands.
- Storage/logging: store large traces only where permitted, use bounded captures and avoid filling shared `/tmp` or Ray storage. No root/system-wide settings, no force-kill of other processes.

**Suggested repository layout** (adapt to existing branch without clobbering the prior PoC):

```text
poc_recurrent_asymmetry/
  reports/
    spec.md
    environment.md
    baseline_validation.md
    profiling.md
    memory_accounting.md
    dependency_dag.md
    track_a_kv_attention.md
    track_b_wavefront.md
    oracle.md
    final_report.md
  scripts/
  traces/
  metrics/
  figures/
```

Store raw measured CSV/JSON, profiler trace pointers and reproducible run manifests. Report `NOT_RUN`/`UNAVAILABLE`, never substitute invented measurements.

---

## 4. Measurement design: common control variables

### 4.1 Workload matrix

**Primary:** fixed R4 and exact depth-indexed KV; BF16; same checkpoint; deterministic greedy, same dataset prompts and comparable KV histories. Split timed **prefill** from timed **decode**.

| Dimension | Minimum desired cases | Notes |
|---|---|---|
| Decode history/context | 1K, 4K, 8K, 16K tokens | Respect feasible context and available HBM; do not silently truncate |
| Active requests | 1, 2, 4, 8 | Include low-request target; skip OOM with reason |
| Decode horizon | 128 and 256 new tokens, or fixed measured token steps | Report actual EOS completions; fixed-step timing is distinct from user-visible generation |
| Prefill prompt length | 1K, 4K, 8K tokens | Match tokenization and causal mask |
| Prefill chunk size | 128, 256, 512, 1024 tokens | Only legal, implementation-supported configurations |
| Physical GPU count | 1, 2, 4, using indices drawn from {4,5,6,7} | Choose/document mapping; measure topologies |
| Effective recurrence | R=1,2,4 *diagnostic only* | Changing depth changes model semantics/accuracy; do not use R1/R2 for apples-to-apples accuracy/speedup claims |

If 16K is infeasible or slow, record failure and coverage rather than shrinking contexts invisibly. Prefer a small yet reliable matrix over uncontrolled combinations; choose representative cases for deep profiling based on the initial sweep.

### 4.2 Timing methodology

- Warm up kernels, allocations and runtime state; separate cold vs steady-state runs. Record GPU clock, memory usage and interference.
- Wall latency (user-visible) and GPU kernel elapsed time (CUDA events / profiler) are **distinct**. Synchronize at defined boundaries; CUDA-event timing without sync discipline can be misleading.
- Repeat stable configurations (e.g. >= 5 timed repetitions) and report median, p90/p95 where meaningful, spread and number of samples. For p95 request latency use enough requests or clearly label its instability.
- Record TTFT, prefill-only latency, decode time per token (TPOT/ITL), E2E per request, requests/sec or tokens/sec **only for workloads actually executed**. State whether prefill/generation is included; avoid dividing by output tokens to mislabel TTFT.
- When comparing runs, hold output length, prompt length, concurrency, attention kernel/backend, GPU count and batch shape fixed wherever possible. Separate runtime/kernel changes from architecture changes.
- Report hardware and **utilization at useful granularity** (SM occupancy or active cycles, GEMM shape/batch, memory throughput, launch gaps, interconnect occupancy). `nvidia-smi` utilization alone is not a bottleneck diagnosis.

### 4.3 Profiling approaches

Use NVTX scopes such as `PREFILL`, `DECODE`, `LOOP_1..4`, `ATTENTION`, `QKV_PROJ`, `ATTN_CORE`, `O_PROJ`, `MLP`, `NORM`, `KV_READ`, `KV_WRITE`, `H2D`, `P2P`, `NCCL`, `SCHEDULER`, and shape/length metadata in a **controlled profiling run**. GPU profiling tools may be unavailable; record permissions, device/backend, and substitute high-quality CUDA-event/torch-profiler timing. Avoid wrapping every tiny kernel in sync for headline latency.

Report two breakdowns:

1. **Exclusive critical-path wall time** (non-overlapping intervals; components sum to total within residual/unattributed margin).
2. **Inclusive activity by component** (may overlap and must **not** be summed into a speedup denominator).

Classify CPU launch/scheduler gaps, GPU attention, QKV/out projections, MLP/GEMM, norm/other, KV copy/allocation, H2D/P2P/collective, synchronization and unclassified time. Where attention is fused, do **not** fake per-op timings; use supported kernel metrics or label fused groups.

---

## 5. Stage 0 — Reference baseline and exactness validation

**Procedure**

1. Locate prior exact-R4 validation artifacts and logs; reuse only verified baseline code/config. Record the prior approximation study's `QUALITY_FAIL` as closed (no predictor reruns).
2. Inspect running architecture and build a table: logical depth index, physical layer ID, device, KV key, number/shape of tensors, prelude/coda and gate execution.
3. Validate greedy tokens/logits/KV for representative prompt shapes and 2+ subsequent decode tokens, including padded batches and varying prompt lengths if supported.
4. Verify that prefill and decode actually execute all 4 loops in fixed R4; do not accidentally measure an exit-gated path.
5. Save baseline timings per case. Verify the current code's baseline accuracy on a small deterministic smoke set when inference code is touched.

**Gate:** `BASELINE_BLOCKED` if reproducibility, KV semantics or GPU safety fails. Do not profile an unvalidated baseline as if it were Ouro exact R4.

---

## 6. Stage 1 — Broad recurrent memory–compute profiling

**Goal:** find actual bottlenecks without assuming KV/attention or PP is the answer.

### 6.1 Time decomposition

For each selected `(phase, context, batch, loop)` measure:
- attention vs FFN/MLP vs projections vs norms/other;
- prelude/coda, gate if active, and runtime/synchronization gaps;
- decode step vs prefill chunk overhead;
- compute-bound vs bandwidth-/launch-/communication-bound indicators;
- per-loop differences, batch scaling, context scaling, and resulting *critical-path* contribution.

**Output:** a per-loop and per-shape breakdown with total measured time, exclusive % of total, and `unattributed` residual.

### 6.2 Logical memory accounting vs measured traffic

Separate:
- **Resident tied weight bytes** (unique physical storage) versus total recurrent weight *accesses*;
- **Logical KV bytes** per `(loop, layer, batch, history length, dtype)` versus *actually allocated* KV / reserved allocator memory;
- **Actual HBM read/write traffic** where supported, versus mere tensor capacity;
- activation intermediates, attention scratch/workspace, CUDA graph pools, page tables, fragmentation and runtime overhead;
- host memory/offloading, if present (mark as deviation; exclude from primary GPU-resident analysis).

A first-principles upper-level check for full-depth MHA KV at sequence length `S` and batch `B`:

`KV_bytes_logical = B × S × R × L × 2 × n_kv_heads × head_dim × bytes_per_element`

Apply the formula only after auditing actual cache layout (including padding, allocation granularity, aliasing, compression and skipped depths). **Allocated bytes != HBM bytes transferred != latency.**

Estimate weight residency, aggregate repeated reads and L2 hit behavior only with supporting counters; do not claim weight bytes are reloaded from HBM R times just because the loop invokes R times.

### 6.3 Recurrence-specific scaling

Compare R1/R2/R4 **as diagnostics** at the same sequence length to isolate scaling of work, KV and memory footprint. Clearly label that changing R changes model behavior. Compare to a parameter/compute/KV-matched unrolled **analytical control** or existing valid implementation when available; a synthetic R×24-layer transformer is not automatically a quality-matched baseline. Differentiate `expected by any deep transformer` from `specifically caused by weight tying`.

**Stage 1 decision:** rank latency constituents by verified removable *critical-path* share. If attention/KV only contributes little, down-rank Track A regardless of interesting cache structure. If pipeline overlap opportunities are absent in exact DAG, down-rank Track B.

---

## 7. Stage 2 — Recurrent dependency and structural audit

Construct an **exact execution DAG** of the real model. Let task `F(r, l, c)` denote recurrent step `r`, physical layer `l` and prefill token chunk `c`. Create edges for:

1. Layer dependency: `F(r,l,c)` needs output of preceding layer `F(r,l-1,c)` (or recurrent boundary for `l=1`).
2. Recurrent dependency: `F(r,1,c)` depends on completed `F(r-1,L,c)` for the same token chunk. Do **not** break this with a stale or predicted state.
3. Causal attention: `F(r,l,c)` needs **the correct per-depth/per-layer KV for past positions**, including earlier chunks at the same layer and recurrence where applicable. Formalize the input-state and KV-generation dependencies from the actual model, not by assuming chunk independence.
4. Mask/position/normalization and embedding/coda dependencies; precise handling of chunk boundaries and unfinished prior requests.
5. Device transfer, output assembly and scheduler prerequisites.

Output a DAG diagram, topological order, explicit **ready-set** computation and a table of potentially concurrent tasks. A proposed overlapping pair is legal only if neither is a prerequisite of the other **and** all its KV inputs are materialized.

Use a tiny 2-chunk × 2-loop × 2-layer toy model to unit-test the DAG: compare computed ready sets and topological schedules against actual serial exact execution; reject illegal schedules. Verify recurrence boundary wrap-around (last stage to first stage) and any cross-chunk same-depth causality.

**Important:** An exactly scheduled prefill chunk may still have causal dependencies on previous chunks. Do not draw optimistic GPU occupancy diagrams as proof of legal or profitable parallelism. The baseline exact DAG may reveal little/no parallelism, and this is a valid negative result.

---

## 8. Stage 3A (priority 1) — Recurrent KV-aware distributed attention discovery

### 8.1 Question and falsifiable opportunity

Can existing exact per-loop KV caches support a faster **4GPU low-request long-context decode** by distributing attention's sequence/head workload or changing KV residency/placement **without approximating K/V or the model state**? What, if anything, is recurrent-specific relative to ordinary CP/TP or KV prefetch?

### 8.2 Measurements

1. At contexts 1K,4K,8K,16K and concurrency 1,2,4,8, measure exact attention fraction of E2E/token critical path and full KV read/write activity. Determine if attention is memory-bandwidth-bound or compute-bound; distinguish QKV projections from attention score/softmax/value aggregation.
2. Record depth-specific KV footprints and reuse/read access; check whether per-depth caches reside on device, whether R× storage actually stresses HBM, and whether long-context throughput degrades.
3. Microbenchmark **actual physical GPU paths** among indices 4–7: P2P bandwidth/latency at relevant small activation and large KV payload sizes; peer access versus staged CPU path; NCCL collective/all-reduce where applicable. Establish whether topology is NVLink or PCIe empirically.
4. Model **exact** attention sharding by sequence or head dimension, with realistic attention softmax reductions (global max/sum normalization, weighted value merge for sequence sharding), K/V placement and gather/communication. Do not assume independent local softmax results can be concatenated as a correct attention output.
5. If a kernel surrogate is needed, use representative tensor shapes and exact mathematical communication patterns; label it as a **microbenchmark**, not Ouro E2E speedup.
6. Compare to fair conventional CP/TP attention and to current single-GPU Ouro. Any recurrence-awareness must contribute a measurable **delta versus a matched generic CP baseline**.

### 8.3 Cost-aware Amdahl + communication oracle

For measured baseline decode total `T` and removable attention critical-path fraction `f`:

`S_upper(f,p) <= 1 / ((1-f) + f/p)` (optimistic Amdahl bound, assuming other costs unchanged).

Then account explicitly for measured/estimated:
- shard compute time, K/V read bandwidth, GPU-to-GPU transfers and softmax merge;
- exposed communication and synchronization after achievable overlap;
- additional launches and kernel efficiency losses due to smaller shards;
- KV replication/sharding capacity and update overhead;
- transfer contention, P2P/collective topology and other running ranks.

Report **zero-cost**, **measured-cost optimistic**, and **conservative** scenarios, plus assumption sensitivity; provide real trace/benchmark provenance per parameter. No unsupported `S=4×` or linear scaling claims.

### 8.4 Track A gate

- `A_NO_BOTTLENECK`: measured attention/KV removable share too small for a meaningful E2E gain.
- `A_COMM_DOMINATED`: theoretically reducible attention, but communication/merge eliminates gains.
- `A_GENERIC_ONLY`: feasible gain exists but is essentially standard CP/TP, not looped-specific.
- `A_CONDITIONAL_GO`: cost-aware gain and a **looped-specific** mechanism both supported.
- `A_INCONCLUSIVE`: insufficient instrumentation/topology/execution evidence.

Do not build a CP engine in this PoC.

---

## 9. Stage 3B (priority 2) — Depth–sequence wavefront prefill discovery

### 9.1 Question and falsifiable opportunity

Can a correct recurrent-depth × prefill-chunk **wavefront** produce materially less fill/drain and idle time than standard stage PP and chunked prefill using the same four GPUs and the same exact numerical computation?

### 9.2 Exact dependency and schedule analysis

1. Use the Stage 2 DAG; enumerate legal tasks for chunk sizes 128/256/512/1024 and contexts 1K/4K/8K. Carefully represent `(depth, layer group, chunk)` and distinct depth-indexed KV prerequisites.
2. Define physical stage mapping options: **layer-stationary PP** (6 unique physical layers per GPU, recurrent wraps), **loop-stationary partition** (one shared full block per GPU, per-depth KV placement; larger duplicated weight footprint), and a simple hybrid **only if** the DAG and memory budget allow it. Separate static PP from model-parallel recurrence-domain sharding; map exact activation transfers explicitly.
3. Build a discrete-event scheduler with **exact DAG readiness**, real/benchmarked per-task compute durations and device links. Include per-GPU serialization, KV readiness, transfer capacity, bubble/fill/drain, wrap-around and memory capacity. Overlap can occur only between **independent** ready tasks.
4. Sanity check the scheduler against an exact serial execution (same work, no missing layers or chunks). Prove there is no fabricated concurrency or negative wait time; export timelines and utilization/critical-path breakdowns.
5. Compare: **single-GPU exact prefill**, naive PP4, well-formed chunked PP4, best reasonable greedy/list scheduler for exact DAG, and proposed recurrent-specific wavefront. A generic PP advantage must not be misreported as new loop parallelism.
6. If all exact legal schedules bottleneck on a critical dependency chain, stop rather than introduce stale activation, cross-token speculation or model modifications without separate authorization.

### 9.3 Track B gate

- `B_DAG_BLOCKED`: exact dependencies leave insufficient concurrent ready tasks.
- `B_BUBBLE_TOO_SMALL`: a strong chunked-PP baseline already captures essentially all opportunity.
- `B_COMM_OR_LAUNCH_DOMINATED`: dispatch/activation/KV overhead defeats scheduling gain.
- `B_GENERIC_ONLY`: any gain is ordinary PP/chunking rather than recurrence-specific.
- `B_CONDITIONAL_GO`: cost-aware gap remains over a strong exact PP baseline and originates specifically in recurrence/wrap-around/depth–sequence interactions.
- `B_INCONCLUSIVE`: insufficient traces/validated scheduler.

Do not implement a full wavefront engine in this PoC.

---

## 10. Stage 4 — Multi-GPU microbenchmarks and oracle methodology

This stage combines common costs for Tracks A/B. Execute cheap, **real 4GPU** communication experiments where safe, not full inference frameworks:

- P2P point-to-point send/recv on actual topology, representative BF16 hidden state tensor sizes (`batch×chunk×hidden`); several message sizes and route pairs.
- NCCL collectives relevant to correct attention sharding or stage aggregation; synchronize correctly and distinguish overlap from standalone cost.
- Representative KV shard/page moves and remote-access surrogates; distinguish sustained bandwidth vs tiny-transfer latency.
- Single-/multi-stream compute + comm overlap, measuring interference and exposed latency. Report kernel-side and host-side elapsed times separately.
- Memory allocation and per-rank KV/weight duplication budget for each proposed mapping.

Produce two independent oracles:

**Track A:** attention acceleration with correct merge and capacity, coupled to measured E2E attention share (include TP/CP generic baseline).  
**Track B:** earliest-finish exact DAG schedule with physical stage mapping, device/link occupancy and measured per-task timings (include generic chunked PP baseline).

For each, compute best case, plausible case and worst case; document all assumptions and use bootstrap/repeated-run intervals for measured inputs when practical. **A zero-communication upper bound is never a measured speedup.**

### Practical decision thresholds (research heuristics, not published truths)

- A **meaningful follow-on prototype** should show a **plausible >=15%** E2E/TTFT or decode-latency improvement, **relative to the strongest fair same-4GPU baseline in the target regime**, after measured/defensible communication costs; or a compelling lower gain with an explicitly argued new scaling/memory-capacity regime.
- A strong upper bound below ~10% suggests NO-GO. An optimistic upper bound >15% but realistic bound <=0 indicates overhead-dominated NO-GO or inconclusive (depending on uncertainty).
- **Looped-specific originality gate:** show recurrence-dependent behavior that would not reduce to generic CP/PP/chunking on a similarly shaped conventional transformer. A `GENERIC_ONLY` result does not justify a paper framed as a novel recurrent parallelism architecture.
- Never claim final performance gains solely from models or simulators. Prototype work requires a separate GO decision.

---

## 11. Stage 5 — Dynamic depth (optional characterization, not in main comparison)

Only after fixed R4 baseline and both track diagnostics are stable:

1. Audit whether selected runtime actually skips loop work or merely selects exit outputs **after computing all loops**.
2. If testing early exits, distinguish `online gate`, `recorded trace replay`, and `nominal gate without physical skip`; log per-token exit depth and **actual executed loop kernel count**.
3. Distinguish cache policies: full depth-indexed versus `last_exited`/ragged/aliased or `single` shared; each changes semantics/capacity. Do not pool quality or speed results from different KV policies.
4. If exploring dynamic depth, run separate correct/total GSM8K/MATH-500 sampling (size and CI reported) and model expected timeline variation. Do not assume dynamic depth always improves speed; less work can also reduce pipelining opportunities.
5. Consult CDB and *Enabling Dynamic Computation in Looped LMs* when interpreting claims about physical compute savings and KV aliases.

**Never let optional dynamic-depth runs delay the primary discovery report.**

---

## 12. Reporting checklist and reproducibility artifacts

For every chart/table, report real vs modeled, units, configs, code SHA, physical GPU indices, software/backend, number of repetitions, sample sizes, error bars where relevant, and scope of inference. A **single-run trace** is useful for mechanisms but not a distribution.

Required files (or clear `NOT_RUN` reason):

| File | Required contents |
|---|---|
| `reports/environment.md` | GPU mapping/topology, versions, reservations, dataset/model hashes |
| `reports/baseline_validation.md` | Exact R4 check, token/logit/KV matches, small accuracy smoke test if paths changed |
| `reports/profiling.md` | Prefill/decode time breakdown, per-loop and context/batch sweep, raw trace pointers |
| `reports/memory_accounting.md` | Unique weight vs loop KV, physical allocations vs traffic, measured vs calculated bytes |
| `reports/dependency_dag.md` | Per-depth/chunk/layer causal graph, legality proof/unit checks, real ready sets |
| `reports/track_a_kv_attention.md` | Track A attention share, KV access, communications, generic CP comparison, decision |
| `reports/track_b_wavefront.md` | Track B exact DAG simulator, schedule/timeline, PP baseline comparison, decision |
| `reports/oracle.md` | Upper/plausible/conservative bounds, measured provenance, critical-path/comm terms |
| `reports/final_report.md` | Executive summary, 2-track verdict table, limitations, recommended next *separate* PoC or NO-GO |
| `metrics/*`, `traces/*`, `figures/*`, `scripts/*` | machine-readable measurements, bounded profiles, exported diagrams, commands |

**Minimum final summary tables:**

A. `phase × context × batch × loop -> E2E/TTFT/TPOT, GPU critical-path %, attention %, FFN %, KV %, overhead %, HBM/PCIe/NVLink bytes when measured`.

B. `weight resident GiB, logical full-depth KV GiB, allocated KV GiB, observed HBM bytes (if available), KV share of capacity/traffic`.

C. `baseline 4GPU type, theoretical oracle, cost-aware oracle, estimated speedup, source of gain, novelty vs generic, quality status, confidence and verdict` for Tracks A/B.

D. `question -> evidence -> supported / unsupported / inconclusive -> next action` (explicitly include failure modes).

Suggested summary verdicts per track: `CONDITIONAL_GO`, `NO_GO`, `GENERIC_ONLY`, `INCONCLUSIVE`, with sub-reasons above. Overall: `DISCOVERY_GO_A`, `DISCOVERY_GO_B`, `DISCOVERY_GO_BOTH`, `GENERIC_ONLY`, `NO_ARCHITECTURE_HEADROOM`, `BASELINE_BLOCKED`, `RESOURCE_BLOCKED`, or `INCONCLUSIVE`. The overall label must be justified by metrics and timing scope.

### Exit conditions and execution discipline

1. **After Stage 0:** stop if exactness/compatibility fails.
2. **After Stage 1–2:** if high-quality evidence shows neither significant attention/KV costs nor legal wavefront overlap, write early NO-GO with measurements rather than spending days profiling.
3. **After Stage 3–4:** do not implement speculative or approximate execution; no final multi-GPU production engine in this scope.
4. Commit scripts and reports reproducibly; push the working branch **only if repository permissions/network permit and the push would not overwrite others' work**. Report push status precisely; never claim success without confirmation.

---

## 13. References and novelty boundaries (read/check before final paper claims)

- Ouro model/checkpoint: https://huggingface.co/ByteDance/Ouro-1.4B-Thinking ; config: https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/config.json
- Ouro research: https://arxiv.org/abs/2510.25741
- Continuous Depth Batching (CDB): https://arxiv.org/abs/2608.09444
- Enabling Dynamic Computation in Looped LMs (early-exit KV/cache caveats): https://arxiv.org/abs/2610.09013
- Looped Latent Attention (cross-loop KV compression; **not** exact KV-sharding novelty): https://arxiv.org/abs/2607.15456
- TeraPipe / token-chunk PP prior: https://proceedings.mlr.press/v139/li21y.html
- Ring Attention / context-parallel prior: https://arxiv.org/abs/2310.01889
- WaveFront Decoding (decode/speculation; distinct from exact prefill DAG study): https://arxiv.org/abs/2609.23033

Use the current installed runtime and actual repository sources as the authority for behavior. Literature describes proposed methods, not guaranteed speedups on this machine. Do not borrow speedup figures across workloads or GPU generations without validation.

---

## 14. Final scientific standards

- **Accuracy first:** Exact R4 is the contract. Never reintroduce stale/predictive hidden states as a shortcut.
- **Discovery before design:** derive architectures from observed bottlenecks and a legal task DAG.
- **Counterfactual fairness:** same hardware budget, request mix and semantics for baselines.
- **Provenance:** `measured`, `inferred`, `simulated`, and `speculative` must be labeled separately.
- **Honest failure:** return a strong NO-GO if neither exact mechanism has margin. Negative evidence is valuable.
