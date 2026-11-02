# Looped PD: KV Admission Recovery, Profiling Stabilization, and Adaptive Prefill-Heavy Headroom PoC

**Specification filename:** `looped_pd_kv_admission_profiling_adaptive_prefill_headroom_poc_spec.md`  
**Required repository destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom_poc_spec.md`  
**Repository:** `/home/esjung/MLLM-EP-github`  
**Reference serving implementation (inspect, do not assume compatibility):** `/home/esjung/MLLM-EP-github/poc_flashvep/ref_loopserve`  
**Baseline evidence commit:** `f4ecea59f36f620c69c4fef09c590bf790802cd9`  
**Models:** `ByteDance/Ouro-1.4B`, fixed R4; `tomg-group-umd/huginn-0125`, fixed R32  
**Hardware:** 8 NVIDIA H100 80GB GPUs, physical IDs 0–7; two GPUs per 1P1D or DP2 serving instance  
**Scope:** First fix/characterize admission and instrumentation. Then establish or refute measured sustained prefill-compute bottlenecks with SLO-safe decode-side spare capacity. A full Dynamic A scheduler is out of scope.

## 1. Purpose and non-negotiable research questions

The original sweep produced long TTFT and prefill-side queues, but **did not simultaneously demonstrate** sustained GPU-compute-limited Prefill (P) and usable Decode (D) spare capacity in the same serving start. This study is designed to close that evidence gap, without selecting only positive conditions or fabricating GPU idle time.

The final outcome must answer, separately for Ouro and Huginn:

1. Why are requests rejected/blocked by KV admission despite apparent device compute headroom? Which exact resource (P KV, D KV, slot, H2D/peer transfer credit, staging buffer, acknowledgment/retention, CPU scheduling) is binding, and when?
2. Can native 1P1D serving remain correct and make forward progress after targeted, *safe* admission improvements?
3. Can valid P-kernel activity be captured on Huginn without materially changing normal serving, and why did earlier profiled runs stop making progress?
4. At which input/output length, chunk budget, and offered QPS do we see **growing P-ready compute queues + high actual P compute activity + near-capacity P service throughput** while D still meets decode TPOT/ITL targets with valid KV and workspace credits?
5. Is this condition reproducible over >=3 independent server starts, and can the decode stage safely execute a small amount of real prefill/continuation work without violating decode SLOs?
6. Does a fair two-GPU aggregated baseline reveal the expected TTFT / TPOT / throughput trade-off against 1P1D at matching inputs, arrivals, memory budget, accuracy, and model runtime?

**Do not equate:** (i) high TTFT with P GPU compute saturation, (ii) low NVML utilization with usable compute headroom, (iii) a prepared-D finite completion rate with sustainable online D service, (iv) a successful internal exact-output comparison with original Hugging Face fidelity, or (v) an offered-QPS overload with a valid stable prefill-heavy regime.

## 2. Required evidence and reference-code audit

Read these before editing/launching anything:

- `poc_flashvep/reports/looped_pd_prefill_decode_bottleneck_final_summary.md`
- `poc_flashvep/reports/looped_pd_phase_a_independent_capacity.md`
- `poc_flashvep/reports/looped_pd_phase_b_bottleneck_crossover.md`
- `poc_flashvep/reports/looped_pd_phase_c_dynamic_a_headroom.md`
- `poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover_final_summary.md`
- `poc_flashvep/reports/looped_pd_atomic_prefill_qps_search.md`
- `poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover/{configuration_matrix.csv,requested_search_grid.csv,serving_runs.csv,bottleneck_classification.csv}`
- Relevant raw `queue_observations.jsonl`, `admission_reasons.jsonl`, `after_measured.json`, `closed_stats.json`, `host_execution_*.json`, profiler/watchdog records, exact token/state checks, and preserved run-specific source archives.

Inspect the complete local tree `poc_flashvep/ref_loopserve` **read-only first**. Identify with exact file:line locations whether it has reusable code/ideas for KV block accounting, context reservation, staged handoff, release/ACK, request scheduling, mixed prefill/decode batching, recurrent loop state, CUDA graph handling, or profiling. Produce a **reference-code applicability matrix** (`feature`, `ref_loopserve source location`, `current runtime location`, `directly reusable?`, `correctness or API incompatibilities`, `decision`). Do not blindly transplant kernels or assume this local reference is a supported drop-in runtime. Respect provenance/licensing and retain the existing qualified native RLT execution path unless an explicitly justified, parity-qualified change is made.

Pin all new run names to the actual code archive, repository commit, model revision, effective execution flags, configuration hash, GPU UUIDs and driver versions. Preserve all previous data and staged changes.

## 3. Baseline GPU-memory accounting (existing evidence, not new claims)

All volumes below are **binary GiB**, and assume a nominal 80 GiB HBM for percentage illustration. Record the *actual* device total and CUDA available bytes in the new experiment before applying an admission limit.

### 3.1 Ouro R4

Previously observed for a qualified 8192-input/64-output 1P1D run:

| Item | Value | Meaning |
|---|---:|---|
| Total nominal HBM per GPU | 80.00 GiB | Capacity reference, **not** allocatable KV capacity |
| P/D KV pool reservation | 66.00 GiB | Pool capacity; not simultaneously live KV usage |
| Model weight storage report | 2.67 GiB | Excludes other runtime allocations |
| P peak allocated | 74.55 GiB | Observed per-worker high-water mark |
| D peak allocated | 73.75 GiB | Observed per-worker high-water mark |
| Valid/allocated KV per token | 786,432 bytes = 0.75 MiB | Structural accounting |
| KV for 8192/64 request | 6.047 GiB | 8,256-token capacity before possible block rounding |
| Blocks per 8192/64 request | 2,064 | Observed reservation geometry (4 tokens/block) |
| Total KV blocks | 22,528 | 66-GiB pool |
| 10 reserved long requests | 20,640 blocks / 60.469 GiB | ~91.62% pool occupancy |
| Free pool then | 1,888 blocks / 5.531 GiB | Insufficient for an additional 2,064-block request |

**Important:** Previous synchronized high-watermarks cannot be inferred by adding unrelated peak counters. The P/D peaks above each describe a different worker. Under an admission refusal, capture same-timestamp live KV block and GPU memory snapshots, including whether reservations are held on both workers before the P computation starts.

### 3.2 Huginn R32

| Item | Value | Meaning |
|---|---:|---|
| Previous default KV pool | 46.406 GiB | From 49,828,331,520 bytes |
| Allocated KV/token | 5,406,720 bytes = 5.15625 MiB | Current uniform allocation geometry |
| Valid KV/token | 2,787,840 bytes (~2.659 MiB) | Structural valid-state accounting only |
| Allocated vs valid planes | 256 vs 132 | ~48.44% of allocated planes structurally unused |

Valid-plane storage reduction is a **hypothesis requiring an exact implementation and equivalence proof**; never treat unused planes as automatically reclaimable GPU bytes or transport bytes. The current native NIXL path already avoids some unused boundary transfers.

### 3.3 Per-request capacity envelope to guide a lighter workload search

The idealized simultaneous-request count is `floor(previous KV pool / per-request allocated KV)`; it excludes block rounding, in-flight reservations, workspace, scheduler slots, fragmentation, and D handoff. These are **planning estimates, not guaranteed concurrency**.

| Model | Input/output | Input:output ratio | KV/request (GiB) | Ideal count within prior KV pool | Priority |
|---|---|---:|---:|---:|---|
| Ouro | 4096/64 | 64:1 | 3.047 | 21 | Lower-memory fallback |
| **Ouro** | **6144/64** | **96:1** | **4.547** | **14** | **Initial primary** |
| Ouro | 8192/64 | 128:1 | 6.047 | 10 | Original high-pressure control |
| Ouro | 6144/32 | 192:1 | 4.523 | 14 | Short-output candidate |
| Ouro | 8192/32 | 256:1 | 6.023 | 10 | Prior figure-like control |
| Huginn | 2048/32 | 64:1 | 10.474 | 4 | Lower-memory fallback |
| **Huginn** | **2560/32** | **80:1** | **13.052** | **3** | **Initial primary** |
| Huginn | 3072/32 | 96:1 | 15.630 | 2 | Moderate-long candidate |
| Huginn | 3500/32 | 109.4:1 | 17.785 | 2 | Original high-pressure control |
| Huginn | 3900/32 | 121.9:1 | 19.799 | 2 | Context-limit stress only |
| Huginn | 2560/64 | 40:1 | 13.213 | 3 | Secondary output-length control |

The *ratio* does not prove GPU-compute intensity: long-context attention, recurrent depth, batching, scheduler timing, and KV admission determine the actual bottleneck. Do not shrink prompts until all P compute pressure disappears merely to remove an admission refusal. Prefer the **largest valid input that retains enough D-side KV/concurrency to demonstrate useful spare capacity**.

## 4. Phase 0 — Safety, reproducibility and model correctness

Before any performance experiment:

1. Validate exact input token counts *after* tokenizer/template; Huginn input + reserved output must be <=4096. Check Ouro runtime context settings and long-prompt semantics without assuming model-card maximum context means tested runtime correctness.
2. Freeze model revisions, BF16, qualified numerical fixes, R4/R32, seeds, request IDs, full-depth recurrent work counters, graph geometry, and prepared-state parity rules.
3. Preserve previous runs, source archives, unrelated work, and staged Git index entries. Create new immutable namespace and branch/worktree as appropriate; never overwrite prior CSVs/results.
4. Repeat internal token sequence, valid KV byte/state, hidden state, partial-tail, NIXL imported-state, and baseline-vs-new-run parity gates; distinguish these from original HF/downstream accuracy (not established).
5. Keep clean timings on an isolated GPU lane unless independently demonstrated not to contend; 8 GPU IDs 0–7 may be used in pairs (0/1, 2/3, 4/5, 6/7) for noninterfering preparation, parity, and short diagnostics.
6. Use existing resource guards and safe restoration; never terminate unrelated processes. Avoid unbounded overload, arbitrary timeouts, GPU OOM loops or data loss.

**Gate 0:** All exact-output/state and model-resource checks pass. Failed configurations must remain visible and cannot be folded into method results.

## 5. Phase A — Root cause and safe mitigation of KV admission

### A1. Observe the actual memory state and event lifetime

Instrument correlated host+GPU timestamps for request arrival; P-ready eligibility; P admission/slot/KV reserve; D reserve; rejection or denial reason; Prefill queued/start/end; handoff enqueue/complete; D import/activate/decode; ACK; P/D KV release. Record the distinct populations of (a) P-ready compute-eligible waiting requests, (b) D-credit-blocked requests, and (c) not-yet-admitted requests; do not combine them into one "P queue".

Capture **same-instant** per-worker:

- Device HBM total/available, PyTorch/CUDA allocator allocated/reserved, independent model weights, graph pools, persistent workspace, NIXL staging, fragmentation/nonreclaimable memory.
- KV total/reserved/allocated/active/free blocks and bytes; reservation per request and expected max output token margin; P/D slots, NIXL in-flight bytes, drain/receive credits, ACK lifetime.
- Request-level reasons for every unsuccessful reservation attempt; separate binding refusal vs raw co-occurring refusal, and refusal *sample frequency* vs exclusive wait time.
- Admission graph: P reserve -> D reserve -> P computation -> handoff -> D activation -> state release. Identify whether P is actually blocked by D pre-reservation and whether it could safely proceed under an alternative protocol.

**Mandatory graphics:** side-by-side HBM budget, time-aligned P/D KV occupancy and refusal markers, and exact admission-state timeline for one representative request. All values in bytes, GiB, and % of both total HBM and configured KV pool.

### A2. Controlled mitigations, one dimension at a time

0. **Baseline:** replay the original 8192/64 (Ouro) and 3500/32 (Huginn) admissible conditions, then run lighter 6144/64 and 2560/32 with identical arrival seeds after workload re-tokenization.
1. **Supported capacity controls:** tune P/D KV pool, compute slots, graph/token budgets, transfer slots, max draining/receiving requests, ACK release, and wire-credit limits. Keep sufficient safety space for both models and the actual running graphs. Do not assume maximizing KV pool is optimal.
2. **Admission-aware scheduling:** where legal, separate P compute-ready waiting from D reservation wait; release stale/excess reservations promptly after a verified ACK, bound staging backlog, and avoid holding both P/D full-KV reservations longer than necessary. Any delayed-D-reserve proposal must handle handoff failure and bounded P-state lifetime safely. Never bypass D capacity checks.
3. **Exact geometry improvements (only if justified):** for Huginn, investigate compact valid-plane KV allocation or model-specific reservation geometry, and for Ouro investigate block fragmentation/over-reservation. These are optional engineering changes with required exact parity and explicit memory accounting. Avoid changing math, attention semantics, approximation, output length mid-run, or precision.

Every control must record memory gain/loss, goodput/throughput, TTFT, TPOT/ITL tails, P compute submissions, transfer bytes, queue/credit delays and parity. A putative "fix" that only reduces P workload or disables logging does not pass.

**Gate A:** Document an admission cause supported by request-block evidence; demonstrate a parity-safe reduction in admission stalls for at least one real workload, OR report `ADMISSION_UNRESOLVED` with causal limits. Do not assume the problem is solved simply because all offered requests eventually complete.

## 6. Phase B — Stabilize Huginn GPU profiling (and validate Ouro measurements)

Prior Huginn profiled runs stopped after **80/128** and **14/128** completed requests; watchdog reason `PROFILE_NO_RESPONSE_PROGRESS_180S`. Clean serving completed independently. The precise causal reason is **unknown**.

1. Reproduce without heavy instrumentation, compare OFF/ON matched starts, and inspect whether the hang is instrumentation-only or a genuine runtime progress issue.
2. Validate **P-only one request**, **D-only prepared state**, **native PD 1**, **PD 4/8**, **PD 24**, then a **short 2–5-second capture** during steady traffic; scale only after a clean short run.
3. Instrument worker/request progress heartbeat, stuck CPU threads, CUDA synchronization, CUDA graph replay, NIXL progress and ACK, profiler buffer/flush, GPU kernels, OOM, allocator pressure and Windows of no kernel submission.
4. Ensure the profiler sees kernels *inside* CUDA graphs; graph-active time is not equivalent to kernel-busy duration or SM occupancy.
5. Predeclare a matched-profile perturbation threshold (default inherited <=5% end-to-end wall difference). Reject unmatched or materially perturbed profiles; preserve traces as diagnostics only.
6. If full-run instrumentation stalls, measure minimal short windows in several independently reproducible starts with matching kernel-hash/interval context; do not equate arbitrary sampled low NVML utilization with valid P kernel saturation.

**Gate B:** Capture low-perturbation, correctly aligned P-kernel activity (and if possible D activity) without stalling; otherwise explicitly `HUGINN_PROFILING_BLOCKED` and retain watchdog evidence. Never mark a missing profile as 0% busy.

## 7. Phase C — Adaptive workload and QPS crossover search

The first step is **safe admission tuning**; only then adapt input/output lengths. Avoid re-running a broad blind Cartesian product of hundreds of configurations. Hold non-target settings constant in any one comparison.

### C1. Length-selection ladder

**Ouro R4:** begin `6144/64` (primary) and check `8192/64` as original/long-prompt control; if admission remains dominant use `4096/64` then `4096/32` while checking P compute intensity; if P is not compute-limited but D still has room, try `6144/32`, `8192/32`, or `8192/64` again after admission fixes. `16384/32` is a last-resort stress/diagnostic, not the flagship motivation workload.

**Huginn R32:** begin `2560/32` (primary), then `3072/32` for a stronger Prefill workload; use `2048/32` if KV admission still dominates; preserve `3500/32` as an explicitly high-pressure control. `3900/32` only if its actual full-context reservation and state correctness gates pass. Optional `2560/64`, `3072/64`, and `2048/16` diagnostic cells test output sensitivity; 16 outputs should never silently replace the primary serving workload.

**Escalation rule:** Prefer the longest input that (i) keeps D admission/credits safely valid, (ii) preserves real D overlap/concurrency, (iii) permits sufficiently high *P GPU compute* occupancy. If every smaller setting removes the compute bottleneck, record the trade-off instead of forcing a positive verdict.

### C2. Fair baselines

- **Static PD:** one full replica on each of P and D, native KV handoff; compare tuned chunked and genuinely atomic full-prompt execution; record real `[0,L)` boundaries. Do not intentionally handicap P with C256.
- **Aggregated DP2:** two independent colocated replicas, mixed continuous batching where actually supported, decode priority, chunked prefill, and a documented per-GPU token budget. Match DuetServe's Agg-vLLM semantics as closely as feasible (including round-robin or documented equivalent). If the current RLT cannot share a forward batch for mixed P+D, label it `DP2_SEPARATE_PHASE_CHUNKED`, not `AGG_VLLM_EQUIVALENT`; do not invent a result for the missing baseline.
- No full Dynamic A scheduling. A tiny controlled D-side borrowing probe is permitted only in Phase D, with exact states and separately identified results.

Freeze model, numerical fixes, 2-GPU total budget, prompt IDs, tokenized lengths, EOS policy, arrival offsets/seeds, request caps, state checks, and qualifying scheduler/geometry within each paired comparison.

### C3. Open-loop offered-QPS search and overload diagnostics

Create actual arrival offsets for every target QPS; do not edit `arrival_rate` without regenerating arrivals. Distinguish **target**, **realized arrivals**, **completed req/s**, and **SLO goodput**. Never let the load generator wait for the prior response (closed-loop mistake).

The following are **initial candidate grids**, to be refined around independently measured **online** capacity boundaries:

| Model | Primary search (target req/s) | Optional bounded overload only |
|---|---|---|
| Ouro | 0.5, 1, 1.5, 2, 2.5, 3, 4, 5 | 7, 10 |
| Huginn | 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2 | 3, 5, 10 |

An arbitrary identical QPS=10 does *not* mean equal relative loading of two different models. Do not claim a high-QPS success when TPOT/ITL or KV space is already saturated. Use stable-rate measurements at λ below and near usable D capacity to look for the bounded window `µ_P < λ < µ_D`, treating µ estimates as workload- and batch-dependent, not interchangeable isolated-prepare constants.

Use 24-request smoke tests; 128+ offered requests and >=3 independent clean starts for the selected candidates; where possible include a predeclared steady measurement interval plus drain, and graph both the arrival window and post-arrival drain separately. For overload, limit duration/requests, ensure full offered-denominator accounting, preserve incomplete runs without treating completed subsets as successful throughput, and prevent indefinite queue growth.

### C4. Required evidence per run

**P:** valid P-ready queue slope and max, P blocked-by-D queue count, actual kernel busy in representative valid windows, active SM/cycle metrics where supported, P completed req/s, input tok/s, throughput plateau, independent *matched* P capacity, compute vs submission gaps, P active batch size, transfer/credit stalls.

**D:** native valid KV free/reserved bytes, binding D admission refusals, active Decode batch and readiness, actual graph-internal kernel activity, idle intervals and HBM activity, D completion rate, TPOT p50/p95/p99, token ITL p50/p95/p99, token maximum interval, KV/workspace headroom, decoded output exactness.

**Global:** target and realized arrival QPS, cumulative submitted/completed by time, TTFT/E2E p50/p95/p99, total and output tokens/s, SLO-goodput, all timeouts/cancellations/denials, per-run ID/start hardware/configuration.

**Distinguish SLO dimensions:** The primary paper motivation may *allow* a large TTFT under P congestion; D headroom is judged independently using predeclared Decode TPOT/ITL limits. Separately report the joint TTFT+TPOT SLO (usually poor under overload), rather than silently switching SLO definitions.

## 8. Phase D — SLO-safe D-side capacity probe (not Dynamic A)

Once a candidate exhibits P pressure and eligible D spare, introduce a **small actual** correct Prefill/continuation workload onto D at controlled increments. Use existing qualified exact connectors, correct KV/hidden state and strict memory lifetime; do not bypass admission or use synthetic GEMM as the only positive evidence.

For each increment capture added D work (tokens, recurrent iterations, GPU ms, data bytes), D KV reserve, transfer overhead, P throughput impact (if integrated), and *preexisting* Decode request TPOT/ITL distributions against a matched no-borrowing run. Increase work only while the frozen Decode SLO/tail constraints remain satisfied; bounded no-op/synthetic control may be supplemental.

A passed prepared-D replay does not prove safe *online* borrowing. Use online staggered arrivals where the D stage is active. If the D cannot safely accept real work, report `NO_SLO_SAFE_BORROW_CAPACITY` without proceeding to Dynamic A.

## 9. Strict success and causal classification

Pre-register thresholds based on calibration, with suggested initial screen values P actual kernel busy >=80%, P service >=85% of matched strongest finite P service, and D request-TPOT p95 <=90% of frozen Decode TPOT limit. **None alone is proof.** Require a growing **P-ready** queue in an interval that excludes D-reservation-blocked requests, valid P kernel attribution, throughput plateau consistent with compute saturation, and no dominant host/transfer/KV stall.

For D: measurable usable compute interval, legal free KV blocks and credits for *actual proposed work*, matched staggered-D test, SLO margin for request-average TPOT and token-level ITL, plus a no-regression D borrowing probe if feasible. Check these in the same start and overlapping observation window; never combine favorable evidence from different starts. Reproduce strict positives in >=3 independent starts.

Classification priority per start:

- `TRUE_P_COMPUTE_BOUND_AND_SLO_SAFE_D_SLACK`
- `P_COMPUTE_BOUND_BUT_D_SLACK_UNPROVEN`
- `P_SERVICE_QUEUE_BUT_NOT_GPU_COMPUTE_BOUND`
- `KV_ADMISSION_BOUND` (P / D component identified)
- `DECODE_BOUND`
- `PROFILING_OR_MEASUREMENT_BLOCKED`
- `MIXED_OR_INCONCLUSIVE`

A high TTFT and low TPOT curve similar to DuetServe Figure 2 is **motivation evidence, not a sufficient identification of a compute-bound P stage**. Record evidence categories and missing gates for *every* scanned point. If neither model passes, report that outcome without claiming recurrent impossibility.

## 10. Mandatory figures, trace contracts, and user-friendly presentation

Build clear publication-style English-labeled figures with a consistent legend, model/length/QPS/N/start annotations, 95% start-level uncertainty for repeated runs, and no invented curve points. Include a companion table listing the exact evidence file/CSV rows used for each plotted value.

1. **DuetServe-style three-panel figure (per model):** offered QPS vs TTFT p95, request TPOT p95 and separately token ITL p99 (do not conflate), total token throughput. Curves for true mixed aggregated DP2 when supported and Static PD. Show any separate-phase DP2 under its true name.
2. **P compute saturation proof:** *time* vs P-ready queue and P-blocked-by-D queue, aligned above/with actual P-kernel occupied intervals; separate plot offered QPS vs P completed service rate and P valid kernel-busy%. Mark when service plateaus and refusal begins.
3. **D spare proof:** real graph-internal GPU compute activity/idle intervals on an aligned P/D timeline, D running batch count, D KV free blocks/bytes and admission credit with refusal events. No blank intervals are labeled idle without device observability.
4. **SLO-safe D borrowed-work curve:** extra D Prefill/continuation work (GPU time or tokens) vs Decode TPOT p95 and token ITL p99, both with frozen SLO lines, plus D KV/credit occupancy. Show the maximal observed SLO-safe point and uncertainty, or `NOT_CONFIRMED`.
5. **HBM waterfall/stacked utilization over time:** actual concurrently sampled P/D memory budgets, base weights, KV reserved vs active, workspaces, staging, total vs available HBM and admission refusal markers. Do not stack incompatible allocator or static pool metrics.
6. **One-page executive view:** a simple two-column Ouro/Huginn figure showing which of P-ready queue, P actual compute, D admission, D SLO and actual D borrowing passed. All evidence labels must match data (MEASURED, DERIVED, DIAGNOSTIC, NOT_MEASURED).

Export PNG and vector PDF/SVG when technically supported, alongside plotting code, data extract, and figure manifest. Do not crop axes deceptively or treat finite-cohort plateau as sustainable capacity without stable-window proof.

## 11. GPU campaign execution and stopping rules

Available physical GPUs: **0,1,2,3,4,5,6,7**. Prefer noninterfering two-GPU pairs for setup and correctness checks; preserve earlier isolated clean-timing policy unless concurrent-pair contention is measured and bounded. Never infer four independent 2-GPU campaigns are contention-free. Ensure each run records physical IDs and device UUIDs.

Execute in order:

- Phase 0 correctness and reference-code audit.
- Phase A short memory/admission microdiagnostics and only necessary mitigation controls.
- Phase B short profiler recovery, with safety/watchdog limits.
- Phase C pilot length/QPS crossover, focused repetition only on promising causal candidates.
- Phase D small SLO-safe borrowing probe and figures, if prerequisites met.

Time-budget priority: **root cause > accurate profiling > P/D causal gate > repeated candidates > attractive plots > optional high-QPS stress**. Do not spend most GPU hours at QPS=10 when the KV blocker is already established. If a phase fails, preserve all evidence, report blockers, and perform safe independent subtasks rather than modifying thresholds post hoc.

No precision reduction, quantization, early exit, training, removal of recurrent loops, truncation of prompts to fake a longer-context workload, or selective omission of failed requests. Do not run mixed workloads under a different hardware allocation without labeling it. Do not overwrite previous runs.

## 12. Deliverables, paths, and hand-off

Create the final report at:

`/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom_final_summary.md`

Place new raw results under a distinct directory:

`/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom/`

Required outputs:

- `reference_code_applicability.md`
- `gpu_memory_accounting.csv`, `kv_admission_events.csv`, `kv_occupancy_timeline.csv`, `worker_progress_diagnostics.md`
- `huginn_profiler_recovery.md` and bounded profiler/watchdog traces
- `config_matrix.csv`, `independent_capacity.csv`, `serving_runs.csv`, `p_d_gate_results.csv`, `paired_repetitions.csv`
- `correctness_and_parity.md`, with exact token/KV/hidden-state counts and scope
- `figures/` and `figure_manifest.json` with all exact run IDs, axes and provenance
- `failures_and_unrun.md`, `reproduction_commands.md`, `final_per_model_verdict.json`
- A concise discussion of *what changed at admission*, and the before/after **HBM allocation/reservation + TTFT/TPOT + measured GPU activity** for both models.

Commit and push only the new intended files; preserve staged external work. Report branch, final SHA, completed/failed/unrun counts, key evidence for Ouro and Huginn, any code changes to native reservation and profiler, and whether the next Dynamic A static loop/token/layer comparison is justified. Do not claim success merely because the desired figures can be drawn.

---

### Final scientific interpretation required

Distinguish (i) a long-input, short-output workload, (ii) an observed P service queue, (iii) a verified P **GPU-compute** bottleneck, (iv) a D GPU with low sampled utilization, and (v) a D GPU with **SLO-safe, KV-legal borrowing capacity**. Only (iii) plus (v), proven for the *same* serving regime and repeated starts, justify the target hypothesis. Failure to find this regime is `NOT_CONFIRMED_ON_TESTED_GRID`, **not** proof that recurrence makes it impossible.
