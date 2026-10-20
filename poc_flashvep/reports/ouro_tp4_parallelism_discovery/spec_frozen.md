# Ouro-1.4B TP4 vs Request-Level Pipeline Parallelism: Matched-Runtime Discovery PoC

**Status:** implementation and benchmarking specification (not a results report)  
**Date:** 2026-10-20  
**Physical GPUs permitted:** `4,5,6,7` ONLY (four H100 GPUs, subject to re-checking live hardware and process occupancy)  
**Intended working repository:** `/home/esjung/MLLM-EP-ouro-request-pp` (discover and verify its actual root before writing)  
**Specification destination:** `/home/esjung/MLLM-EP-ouro-request-pp/poc_flashvep/reports/ouro_tp4_parallelism_discovery_spec.md`  
**New result directory:** `<repo>/poc_flashvep/reports/ouro_tp4_parallelism_discovery/`  
**Model:** `ByteDance/Ouro-1.4B`, BF16, fixed recurrent depth **R=4** for all primary experiments.

## 0. Research question and hard scope

Determine whether loop-wise partitioning of an exact Ouro-1.4B recurrent forward path provides **measurable benefits over TP4**, beyond advantages caused by an unoptimized Python single-process stage scheduler. The relevant comparison is not just Layer-PP4 vs Loop-PP4; it must include an actual four-GPU tensor-parallel inference implementation and a more efficient common PP transport control.

The research hypothesis is *conditional*: for a weight-shared 24-layer core executed four times, loop-wise PP may exchange many per-layer collective synchronizations for only loop-boundary hidden-state transfers, while independent requests fill the four-stage pipeline. TP4 may instead win at small concurrency because it parallelizes computation for a single request and may have faster optimized kernels. Neither outcome is presumed.

Primary work: **fixed R4, exact full KV, no prompt chunking, request-level parallel execution, 4 physical GPUs**. Do **not** spend experimental budget on fixed R2/R3-only sweeps. True online token-wise dynamic depth (CDB or *Enabling Dynamic Computation* semantics) is a separately gated follow-up; do not silently replace it with post-hoc depth selection or a fixed-R2 control.

**No new adaptive scheduler, cross-loop compression, approximate KV policy, or architecture claim** in this PoC.

## 1. Mandatory input evidence and baseline contract

Read the existing artifacts **before implementation**:

- `poc_flashvep/reports/ouro_request_pp_discovery/analysis.md`
- `poc_flashvep/reports/ouro_request_pp_discovery/benchmark_summary.md`
- `poc_flashvep/reports/ouro_request_pp_discovery/correctness.md`
- `poc_flashvep/reports/ouro_request_pp_discovery/timing_contract.md`
- `poc_flashvep/reports/ouro_request_pp_discovery/reproduce.md`
- Existing implementation, source hashes, raw metrics, stage traces and profiler commands.

Relevant reported evidence (do not treat these as new measurements):

- 27 paired scenarios, 2 PP placements, 3 repeats each (162 full-cohort executions). Loop PP won the median makespan in 25 scenarios, including most fixed-R4 scenarios.
- `R4/C4/P128/O32`: Layer-PP4 `1.8765 s`, Loop-PP4 `1.0621 s` (`1.767x`).
- `R4/C16/P512/O128`: Layer-PP4 `20.7039 s`, Loop-PP4 `12.3288 s` (`1.679x`).
- Heavy-profile `R4/C4/O8`: actual P2P copies `508 vs 124`, P2P payload `31.641 vs 6.328 MiB`, sum P2P activity `1.608 vs 0.613 ms`, heavy window `906.187 vs 509.185 ms` (Layer vs Loop). These heavy-wall numbers MUST NOT be compared directly with clean-clock numbers.
- R4 has 16 Stage items/forward in Layer-PP4 vs 4 in Loop-PP4, with 15 vs 3 hidden-state inter-GPU handoffs per forward (excluding next-token return and controls). Recorded stage operations include queueing, publication, graph replay, KV staging, and waits. The observed speedup is **not explained by payload bandwidth alone**.
- In the heavy trace, no recorded kernel/copy activity is around 85% (Layer PP) vs 75% (Loop PP) for R4/C4. This is not a validated production utilization estimate or a decomposition of host vs architectural bubbles.
- Weight allocation per GPU `0.9493 GiB` for Layer PP vs `2.6722 GiB` for Loop PP, both including auxiliary replicas. R4 logical KV allocation is the same per GPU across these two static partitions. Retained per-shape graph memory substantially inflates PyTorch peaks; the existing matrix did not measure maximum admitted requests and had no OOM.
- Both PP implementations matched the single-GPU reference **bitwise on tested numeric probes**, generated IDs matched in the measured primary workloads, and different-request GPU kernel overlap was physically observed. This is **not** benchmark accuracy or dynamic-depth correctness.
- Two buffer/RoPE controls retained Loop PP's advantage but did **not** eliminate the single-process host scheduler / publication waits.

Preserve these results. Do not overwrite previous report files, raw data, or source snapshots. Record the exact existing code commit before making changes; create a new branch if appropriate.

## 2. Platform, model and topology preflight

1. Verify `pwd`, `git rev-parse --show-toplevel`, branch, commit SHA, dirty working tree and all relevant local code versions. Treat `/home/esjung/MLLM-EP-ouro-request-pp` as an expected path, not an excuse to create a fake repository if absent. **Do not modify** `/home/esjung/MLLM-EP-github`.
2. Check GPU 4/5/6/7 model, free HBM, current processes, `nvidia-smi topo -m`, NVLink status, driver/CUDA/NCCL/torch/vLLM versions. Avoid GPU 0–3 completely; never kill another user's jobs.
3. All GPU-running commands must use `CUDA_VISIBLE_DEVICES=4,5,6,7`. Logical distributed ranks 0,1,2,3 map to physical GPU 4,5,6,7 respectively. Assert world size is exactly four.
4. Pin model checkpoint revision, tokenizer revision, BF16, `total_ut_steps=4`, attention backend and sequence/position semantics. Confirm actual tensor dimensions in the loaded checkpoint rather than hardcoding guessed TP layouts.
5. Ouro-1.4B nominal architecture is 24 shared physical Transformer layers, four recurrent applications, hidden size 2048, with MHA and SwiGLU. Check `num_attention_heads`, `num_key_value_heads`, head dimensions, normalization and output projection layout from the actual config and implementation. Record them.
6. Distinguish `vLLM 0.26.0` (reported last native Ouro-supported release in the official project documentation) from later unsupported versions. If testing vLLM, establish a compatible isolated environment or otherwise report the incompatibility; do not replace the existing environment destructively.
7. Warm-up, graph capture, model loading, preprocessing, tokenizer loading and NCCL initialization are excluded from the **primary serving clock** but must be reported separately. All arms receive equivalent pre-clock preparation.

## 3. Comparison matrix and ownership

| Arm ID | Placement | Model weight placement | KV placement | Primary role |
|---|---|---|---|---|
| `LPP4_legacy` | Physical layer PP4, 6 physical layers/GPU and every recurrent loop returns to first stage | partitioned core + documented replicated auxiliary weights | `(loop, layer)` slots owned by layer rank | locked prior reference and replication check |
| `LoopPP4_legacy` | Loop PP4, GPU4=R1, GPU5=R2, GPU6=R3, GPU7=R4 | full shared core on each rank | all 24 layer KV slots for owned loop | locked prior reference and replication check |
| `LPP4_mp` | exactly the same placement as LPP4_legacy | same model math | identical full KV semantics | efficient common multiprocess/event-driven PP control |
| `LoopPP4_mp` | exactly the same placement as LoopPP4_legacy | same model math | identical full KV semantics | efficient common multiprocess/event-driven PP control |
| `TP4_matched` | 4-way intra-layer tensor parallelism, all four ranks jointly compute each loop and layer | shard appropriate linear weights/heads, replicate small norms as required | heads sharded across four ranks | **mandatory** engine-matched TP4 baseline |
| `TP4_vLLM` (conditional) | optimized external vLLM `tensor_parallel_size=4` | engine-managed | engine-managed | strong optimized external baseline, explicitly different engine |
| `DP4` (optional, time permitting) | four independent entire-model replicas and disjoint request routing | full model on each rank | independent per-request local full KV | capacity/throughput context only, not the primary TP test |

Do not label an external vLLM TP result as *engine-matched* to a custom PP runtime. Do not compare single-rank TP, CPU offloading, TP2, or partial GPU use while calling the result TP4.

### Shared execution semantics

- Fixed **R4** for all primary comparisons; every token executes all 4 loops and all 24 physical layers each loop.
- Full exact layer- and recurrence-indexed KV, without KV sharing, compression, offloading, quantization, sparse approximation or dynamic exit.
- **Whole-request prefill** (no intra-prompt chunking), one generated token at a time for autoregressive decode.
- A request can be in flight on different pipeline stages only according to its causal dependencies. Different requests must overlap on separate physical GPU ranks in the PP arms.
- For primary workload, freeze input token IDs and output/teacher-forced token IDs independently of the model's generated outputs; lengths are forced. This isolates hardware cost and avoids stop-token divergence. Run a separate *real greedy generation* correctness lane.
- One request per PP stage item, as in the previous discovery; do **not** merge requests into larger GEMMs in only one PP arm. In `TP4_matched`, clearly specify request batching/admission to minimize confounding, then separately evaluate optimized dynamic batching if supported.
- Keep workload ordering, FIFO policy, admission timestamps / offered arrivals, model and precision consistent; record when executable scheduling necessarily differs among placements. Never force a GPU-wide barrier between different requests simply to match timing.

## 4. Implement a correct four-rank TP4 execution

Prefer reusing established tensor-parallel layers and NCCL/PyTorch distributed APIs already present in the repository or a compatible Ouro implementation. Do not build unsupported model algebra from scratch when a validated sharding implementation can be reused. Use a dedicated process per GPU and an explicit four-rank NCCL process group.

A minimum valid TP4 must:

1. Shard Q/K/V projection and attention heads appropriately for the verified Ouro MHA geometry. With 16 KV heads and TP4, each rank may own four KV heads **if confirmed by checkpoint config**. Maintain precise head index and RoPE semantics.
2. Split attention-output projection and MLP projections consistently (column-/row-parallel patterns for QKV, SwiGLU up/gate/down and their transposes as applicable). Insert mathematically required collective reductions/gathers. Do not duplicate the full dense core on each GPU and merely call it TP4.
3. Replicate or shard RMSNorm, residual paths, embedding and LM head correctly. Explicitly log any auxiliary replication and full-vocab reductions; maintain exact positional embeddings, attention masks, rotary offsets and loop index.
4. Use the same original checkpoint weights and model math. Preserve weight sharing across loops and four distinct recurrent KV slots per physical layer. No per-loop weight reparameterization.
5. Store KV on its owning TP rank by head, for each of `R1...R4 × L1...L24`, and correctly update it over prefill and decode. Log per-rank ownership / allocated and reserved bytes.
6. Implement mathematically correct rank synchronization, final logits assembly / top-1 token selection, token broadcast and next-step state management. Avoid hidden CPU round-trips of intermediate full hidden tensors.
7. Correctly overlap only operations with actual independence. Do not report a standard NCCL all-reduce as an MoE all-to-all. Count the **actual** collective type, count, tensor bytes and its timeline activity, not only the expected textbook number.
8. Validate against single-GPU Ouro reference with numerical tolerances appropriate for reordered BF16/FP32 reductions. TP4 may **not** be bitwise equal due to reduction order. Require exact token IDs in stable greedy tests and report max-absolute/max-relative error and any differing logits or argmax for adversarial near-tie probes. Never fake bitwise matching by copying reference outputs.

Report critical TP4 details: collective and GEMM algorithm/backend, `NCCL_*` overrides, CUDA graph mode, TP shard shapes, attention kernels, GQA/MHA mapping, rank-level memory and exact process topology. If full implementation is blocked, produce a documented blocker and a small operator-level TP control; **do not fabricate full TP4 results**.

## 5. Common optimized PP transport control — mandatory for any architecture claim

The previous 1.3–2.0x PP-vs-PP result is heavily confounded by frequent Python stage operations. Provide a matched, more efficient `LPP4_mp` and `LoopPP4_mp` control if feasible **before** attributing a TP-vs-PP difference to the parallelism architecture.

- Prefer a true 4-process, GPU-affine worker arrangement. Stage queues, ownership, event publication, CUDA stream waits and transfers must not require the previous central Python/GIL-bound scheduling on every stage boundary.
- Reuse receive buffers and prepare position-only RoPE constants ahead of measured execution in **all** arms where relevant.
- Preserve same model math, exact KV semantics and request-level execution. Be explicit about which graph shapes are captured and pool reuse to avoid the previous explosive per-length retained graph memory.
- Keep stage processing comparable in both PP variants: same scheduler semantics, same buffers/communication API, CUDA graph policy and telemetry. Investigate actual inter-rank P2P or NCCL send/recv availability and correct stream ordering. P2P copies are permitted; avoid host staging of hidden states.
- Record graph launches/forward, host handoff latency, queue-ready to kernel-start delay, stream wait gaps, synchronization calls and different-request GPU kernel overlap.
- If the multiprocess control proves too costly to implement, **stop short of asserting architecture superiority**; retain both `legacy` and matched TP comparisons with an explicit implementation-confounder verdict.

## 6. Functional correctness gates

### Gate C0: environment and existing baseline reproduction

Re-run **a small locked subset** of original fixed-R4 Layer PP and Loop PP scenarios and verify output / measurements plausibly reproduce prior medians. Report differences and never silently change backend. Check physical GPUs, source hashes and legacy reference.

### Gate C1: operator-level TP correctness

Compare TP vs single GPU for layer attention, MLP and norm/residual on deterministic inputs; include prefill, one decode step with populated KV, all ranks and all loops. Test different sequence lengths and attention masks. Validate gradients are irrelevant (inference only).

### Gate C2: full-model TP correctness

Compare R4 full model on deterministic token IDs against single GPU: numerical logits tolerance, rank-consistent greedy IDs, ownership and KV shapes/values (reconstructed or matching head slices). Include at least P128/O8, P512/O32 and P2048/O32 where feasible. Run a few EOS-enabled capped greedy probes *outside* the forced-output performance matrix.

### Gate C3: concurrency and actual GPU execution

Show full TP4 participation (all four ranks perform expected model compute); show cross-request kernel overlap in both PP implementations; ensure each generated token is causally ordered. Validate no request mixes KV from another request. Use low-overhead traces for timing and Nsight only in a separate profiler lane.

**No benchmark claims until C1-C3 pass.** Correctness failures are first-class output, not a reason to skip the baseline.

## 7. Primary benchmark: locked fixed-R4 conditions

### A. Smoke and focused controls

Use 4 GPUs simultaneously for every arm. Warm up and measure representative locked cases:

| R | C (requests) | P input tokens | O output tokens/request | Motivation |
|---|---:|---:|---:|---|
| 4 | 1 | 128 | 32 | low-concurrency / single-request TTFT and TPOT |
| 4 | 4 | 128 | 32 | previous 1.767x PP comparison |
| 4 | 4 | 512 | 32 | previous 2.044x PP comparison |
| 4 | 8 | 512 | 128 | medium concurrency / long decode |
| 4 | 16 | 512 | 128 | previous 1.679x PP comparison |
| 4 | 32 | 128 | 32 | large cohort and capacity constraints |
| 4 | 4 | 2048 | 32 | longer-prompt sensitivity |

B. If correctness and overhead gates pass, repeat **all 21 fixed-R4 rows** from the prior 27-scenario table, with exact frozen request sets, initial order and token IDs. Fixed R2/R3-only rows are *not* primary targets. Optional longer prompts (`P4096` or beyond) should be labeled *new* and gated by memory/quality validity.

C. At least 3 independent paired physical repeats per case after identical warm-up; more repeats for noisy/crossover cases. Rotate arm order to limit thermals and caching biases. Log run ordering, device clocks/occupancy, output artifacts and reasons for failures. Do not overclaim statistical significance from three repeats.

D. Keep clean benchmark and heavy profiler executions separate. No graph capture, model initialization, communicator establishment, recompilation or large one-off mallocs inside the measured serving window. Verify no unintended leftover graph pools between cases, or isolate a process per shape and report that policy.

### Metrics and clocks

- Primary: **finite cohort makespan** (per previous timing contract) and **output tokens/s** for exactly the same forced O lengths and request IDs.
- Per-request: TTFT P50/P90, TPOT P50/P90, E2E P50/P90 with explicit timestamps and queue-wait inclusion/exclusion; also Requests/s.
- GPU compute: kernel-service time, per-GPU active timeline, critical path, pipeline fill/drain, host publication gap, queue-ready wait and time with no recorded kernels. Never equate a heavy-trace inactivity complement with clean-serving utilization.
- Communication: actual NCCL all-reduce/all-gather/reduce-scatter or send/recv operations, startup and total bytes, GPU kernels/copies, timeline duration and exposed (non-overlapped) critical-path impact when demonstrable. Distinguish cumulative operation time from wall time.
- Memory: weight, logical KV, reserved KV, graph pool, temporary activations, PyTorch allocated/reserved and NVML peak, by physical GPU. Report max feasible request count only if an admission/capacity sweep is explicitly executed; otherwise no capacity claim.
- Report exact GPU ownership of request/KV slots, request-level token data hashes, bitwise-vs-tolerance interpretation, throughput/latency dispersion and OOM or timeout counts.

### Comparisons and fairness labels

- **Legacy comparison**: `TP4_matched` vs old `LPP4_legacy`, `LoopPP4_legacy`. Label *implementation-level*, not architecture-proof.
- **Improved comparison**: `TP4_matched` vs `LPP4_mp`, `LoopPP4_mp`. This is the core causal control if the common worker optimization is successful.
- **Strong external baseline**: `TP4_vLLM` on the same 4 GPUs and checkpoint, if compatible. Report it in a clearly separated table with kernel/batching/graph differences. Do not conflate high-level model server and custom runtime results.
- **Optional DP4**: only to characterize throughput of independently handled requests and per-request capacity; use exactly the same frozen cohort and explicitly identify different admission semantics. Not mandatory to complete TP4 discovery.

Do not assert PP beats TP due to lower *communication count alone*, since a TP collective and PP P2P are different operations and TP has a different compute decomposition.

## 8. Profiling and ablations

Run a short representative heavy-profile case for each arm, ideally R4/C4/P128/O8 and one decode-heavy case, always distinct from clean timing.

1. Mark request ID, recurrent loop, physical layer range, step, rank, CUDA stream, queue ready, publication, launch and completion timestamps.
2. Export Nsight or equivalent timeline and machine-readable breakdown of `compute`, `collectives`, `P2P`, `host boundary`, `queue`, `graph`, `idle/inactivity`, `prefill` and `decode`.
3. Verify the actual collective count per logical forward for TP4. The *nominal* two reductions per transformer layer (`24 * 4 * 2 = 192`) is only a hypothesis/upper-level orientation; fusions, sharding choices and backend scheduling may change actual operations.
4. Compare `legacy` versus `mp` for both PP placements to estimate how much of the historical advantage came from per-stage Python/host overhead. Attribute only demonstrated exposed latency, never sums of nested events.
5. Control for per-request GEMM size, graph mode, receive-pool allocation, position constants and output head. If a setting cannot be matched across engines, highlight the limitation prominently.
6. Explicitly separate *weight/KV memory capacity* from *runtime workspace*. Full KV is expected to be shardable in TP4 by KV head and LoopPP4 by recurrence; neither inherently compresses total KV.

## 9. Dynamic depth: future gated extension, not part of primary R4 claim

The final research may target both static R4 and **true online token-level dynamic depth** under a selected published contract (CDB OR *Enabling Dynamic Computation in Looped LMs*). Do not combine their thresholds/cache policies and claim they are identical.

Proceed ONLY if fixed-R4 controls show a meaningful, reproducible advantage or a well-specified new mechanism. Before implementing dynamic TP4/PP4, audit the chosen paper and code for:

- Token-wise learned exit-gate evaluation and **actual** skipping of later loops (not post-hoc selection);
- Which historical KV is consulted at deeper recurrence indices after earlier tokens have exited, `last_exited`/best-available semantics, and exact cache/update policy;
- Consistent per-token stopping, EOS, quality, and output positions across all parallelism arms;
- Active-token/ragged KV batching semantics, collective participation and deadlock safety in TP4;
- Per-loop active-token distribution and resulting PP rank imbalance, plus latency of communication and KV affinity.

Dynamic comparisons must use quality-matched, algorithm-matched policies. Run task-level quality/accuracy and matched workloads; a depth trace replay is useful for scheduler isolation but is **not** proof of an executable true-dynamic model.

## 10. Decision gates and scientific reporting

### G0 — Cannot execute TP4 correctly

**BLOCKED**, not negative TP4 evidence. Preserve logs and minimal operator tests. No fabricated latency.

### G1 — TP4 correct but common engine confounder unresolved

**CONDITIONAL IMPLEMENTATION COMPARISON** only. Do not claim TP is architecturally worse. Show both clean-clock and heavy-trace evidence and list missing fair controls.

### G2 — Optimized common transport control succeeds

Compare TP4, LPP4_mp, LoopPP4_mp on fixed R4. Interpret across low/medium/high C, latency and capacity. A heuristic *research continuation* signal is ~10–15% or more repeatable favorable makespan/goodput advantage in several meaningful settings after overhead controls; this is not a statistical significance threshold or a paper result. Also consider wins in long-context cases where one arm cannot fit a request, with capacity correctly measured.

### G3 — External optimized TP4 wins

Be explicit. Characterize the gap and practical relevance. If TP4 is consistently better under representative concurrency while its per-GPU KV memory is also competitive, **NO-GO for a generic LoopPP4 superiority claim**. A niche long-context advantage must be demonstrated rather than presumed.

### G4 — Dynamic opportunity

Only after a real online exit/quality-matched experiment. Depth-only fixed R2/R3 cases from the older report are useful supporting evidence of stage utilization, not an acceptable replacement for dynamic evaluation.

Final verdict: `GO / CONDITIONAL / NO-GO / BLOCKED`, with separate scores for correctness, controlled-performance evidence, practical significance, production-baseline robustness, and novel loop-specific mechanism. **Do not equate a TP4 performance win with research novelty.**

## 11. Required output artifacts

Write the following in a new directory:

`<repo>/poc_flashvep/reports/ouro_tp4_parallelism_discovery/`

- `environment.md`: model SHA, code SHA/branch, GPU topology/occupancy, driver, NCCL, torch, backends and dependency versions.
- `implementation.md`: layer/loop/TP weight maps, collectives, full KV ownership, exact cache and residual semantics, diagrams and architectural differences.
- `correctness.md`: per-operator and end-to-end comparisons, numerical tolerances, greedy ID matches, KV checks, failures and limitations.
- `benchmark_summary.md`: complete comparable tables, percentile semantics, 3-repeat medians / spread and useful plots.
- `raw/` + machine-readable CSV/JSONL including work definition, request tokens, output tokens, timing, bytes, memory, source SHA and rank.
- `profiles/`: Nsight evidence, annotated GPU timelines, exposed vs cumulative communication, stage boundary evidence.
- `transport_control.md`: legacy PP vs common efficient PP controls, expected/hypothesized vs actually removed host/queue gaps.
- `tp4_external_baseline.md`: compatible vLLM baseline if attempted, exact version and comparability notes (otherwise explicit `NOT_RUN`/`BLOCKED`).
- `reproduce.md`: runnable commands and all environment variables for each arm; no destructive cleanup.
- `analysis.md`: primary research hypotheses, measured conclusions, why wins/losses occur, failure modes, figure-ready tables, limitations and verdict.
- `status.md`: implemented/tested/not run; blockers, follow-ups, commit SHA; exact tests performed.

Keep original `ouro_request_pp_discovery/` reports unchanged. Commit tested code and reports in the separate Ouro repository. **Do not push without explicit user permission**. Do not spend time on a new speculative/adaptive scheduler before this TP4 comparison is trustworthy.

## 12. Minimum run order (to avoid wasted engineering)

1. Read all prior reports, inspect repo/branch and GPUs 4–7, pin baseline.
2. Select practical TP4 implementation route: verified native TP support or custom sharding; document tradeoffs.
3. Pass operator-level TP smoke, then full R4 correctness with exact KV.
4. Run a small clean TP4 vs legacy PP pilot with frozen inputs; record limitations.
5. Implement common low-overhead PP transport control and re-check correctness, if feasible.
6. Run focused matched comparison, then all 21 prior fixed-R4 scenarios (if gates pass).
7. Profile and explain collective vs P2P vs host gaps, memory and request concurrency behavior.
8. Optional separate compatible vLLM TP4 baseline; optional DP4 context.
9. Write all artifacts; decide GO / CONDITIONAL / NO-GO / BLOCKED.
10. Plan true dynamic-depth work *only if warranted*; do not report it as done.

## References / source-of-truth pointers

- Local prior discovery: `<repo>/poc_flashvep/reports/ouro_request_pp_discovery/analysis.md` and `benchmark_summary.md` (user-provided copies inspected in this specification's preparation).
- Official Ouro project and pinned vLLM support statement: https://ouro-llm.github.io/
- Official model card: https://huggingface.co/ByteDance/Ouro-1.4B
- vLLM tensor and pipeline parallel guidance: https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
- CDB project (dynamic-phase audit only): https://github.com/kschwethelm/continuous-depth-batching
