# Adaptive Loop-vs-Token PD Serving: 2-GPU W1–W4 PoC Specification

**Status:** Proposed, not executed  
**Project:** `MLLM-EP` / `poc_flashvep`  
**Time budget:** target 10 h; **hard stop at 12 h elapsed**, including analysis and safe closeout  
**Scope:** Ouro R4 and Huginn R32; **2 GPUs per serving configuration**; physical devices 0–7 available for safe assignment  
**Primary deliverable:** reproducible comparative evidence, including failures and unmeasured cells, rather than a guaranteed speedup

## 1. Objective and scientific boundaries

Evaluate whether **online, load- and SLO-aware recurrent-loop cut selection** improves serving over **strong fixed-loop, token-prefix migration, and colocated PD baselines** under realistic, longer W1–W4 workloads. Separate:

1. The *partition axis*: token prefix/suffix versus recurrent depth.
2. The *decision policy*: static versus online adaptive cut.
3. The *transport protocol*: native completed-token-chunk streaming versus completed-prompt snapshot.
4. The *serving environment*: heterogeneous arrivals, Decode pressure, and P/D resource contention.

**Excluded:** generic internal-layer migration and Layer-Static/Layer-Dynamic, 4-/8-GPU pooled serving, new cross-depth GEMM co-batching, speculative decoding, model retraining, approximation, early exit, and changed recurrent depth. Generic logical-layer equivalence remains an acknowledged novelty limitation in reports; absence of the Layer baseline is **not** proof of Loop-specific novelty.

Co-batching / Native Mixed is **OFF for the main 7-arm causal comparison**, including a guard against mixed Prefill and Decode recurrent spans in the same core batch. This is an isolation ablation, not a claim to outperform fully tuned mixed-serving systems. If time permits, a strong native mixed-serving baseline may be noted in a **separately labeled optional** context test; it must not be combined with or substituted for the seven arms.

## 2. Reusable code and provenance

Working repository: `/home/esjung/MLLM-EP-github`  
Private pinned engine: `/home/esjung/external/looped-pd-abc-rlt`  
Reference engine, read only: `/home/esjung/MLLM-EP-github/poc_flashvep/ref_loopserve`  
Previous PoCs: `poc_flashvep/looped_pd_abc/`, `poc_flashvep/rcc_pd_poc/`, `poc_flashvep/adaptive_recurrence_cut/`  
Reports: `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`

Verified historical pins:

- Numerical-correctness base: `3b7a722898dab5f6dd797b474c1ff461659d8fa1`
- RCC Stage 1: `b53a8fed36a1706ac74754b53753f20edb8b913e`
- Dynamic A snapshot-cut evidence: `6c0a597c4ee58d54bde738307e05ac39f008fec0`
- The existing private RLT source is version-pinned and may contain uncommitted patches. Archive and hash its exact runtime and overlays; do **not** overwrite, clean, or replace them.

Model pins:

| Model | Checkpoint | Frozen depth | Type | Precision | Context |
|---|---|---:|---|---|---:|
| Ouro Base 1.4B | `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | R4 | dense looped | BF16 | Model/engine validated limit; do not infer from the 4,096-row staging capacity |
| Huginn-0125 | `tomg-group-umd/huginn-0125`, revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | R32 | dense looped with prelude/coda/injection | BF16 | 4,096 total tokens |

**Important inheritance:** The previous whole-prompt A connector only handled a single chunk. The new `adaptive_recurrence_cut` implementation retains per-token intermediate states across multiple chunks and passed finite KV/hidden/output checks. Reuse the qualified mechanism and revalidate it under the final workload. Correctness against the official HF reference and downstream benchmark accuracy are **not yet established**.

## 3. Seven required 2-GPU arms (no Layer arms)

All comparisons use the **same model, checkpoint, precision, 2 physical GPUs, input IDs, output targets, arrival schedule, numerical patches, measurement harness, and separately tuned common GPU execution parameters**. The P/D split for all disaggregated methods is 1P:1D. Both colocated DP configurations use two independent full-model replicas, one per GPU.

| ID | Arm | Definition | Required distinctions |
|---|---|---|---|
| B1 | `DP` | 2 independent colocated Prefill+Decode replicas; baseline policy and an appropriately large unchunked Prefill limit | Verify it is distinct from DP_Chunked |
| B2 | `DP_Chunked` | 2 independent colocated replicas; token-chunked Prefill and separate-phase interleaving, **without fused Mixed/GEMM co-batching** | Token chunk-size tuning; same replica count |
| B3 | `PD_G1` | 1P+1D; full prompt Prefill on P, completed full-depth token-chunk KV streaming to D | Native proven streaming policy; do not replace with snapshot silently |
| B4 | `Loop_Static` | 1P+1D; fixed recurrent cut `n` for all eligible target requests; D performs R−n remaining Prefill loops | Include cut0 and cutR endpoint controls and choose strongest static using calibration only |
| B5 | `Loop_Dynamic` | 1P+1D; **online** selection of `n` from observed P/D load and Decode SLO slack; D completes remaining Prefill loops | Not a retrospective regime composition; must have live admission and fallback |
| B6 | `Token_Static` | 1P+1D; P computes full depth for a prefix ending at a token-chunk boundary, D computes full depth for suffix | Transfer valid *full-depth prefix KV* and model-specific state |
| B7 | `Token_Dynamic` | 1P+1D; online choice of prefix-token migration boundary based on current P/D load and SLO budget | Real online action with validity/credit guard; same service objective as Loop_Dynamic |

**Transfer-protocol fairness:** The earlier cut-map full-cut control used a completed-prompt **snapshot**, while native `PD_G1` streamed full-depth token chunks. Single-start performance differed materially. Report the full-cut snapshot and native streaming separately, and include a **matched transport-policy control** when comparing cut effects. Avoid reporting a protocol change as a Loop-Cut benefit. If a matched protocol is impractical, label the comparison `TRANSPORT_CONFOUNDED` and do not claim causal speedup.

**Co-batching policy:** Same separate-phase policy on all seven arms. `DP_Chunked` interleaves smaller Prefill jobs with Decode in time, but does not put Prefill and Decode hidden rows into one recurrent-core GEMM. The original `DP` must remain a genuine distinct baseline; audit flags and kernel paths to verify this.

## 4. Partition correctness contracts

### Loop cut n

P computes recurrent depths `[1 ... n]` for **every prompt token**; D computes `[n+1 ... R]` and subsequently full-depth Decode. Transfer all consumer-valid prefix-depth KV plus **all prompt-token intermediate hidden/state rows** (Huginn's injection carrier included). Preserve per-depth causal progress, position encoding, RNG, phase ownership, and acknowledged state lifetimes. `n=0` means D-local full Prefill (no P compute), `n=R` means full P Prefill. Depth values remain R4/R32 for every generated token.

### Token prefix cut k

P computes **all R loops** for tokens `[0 ... k)`; D computes **all R loops** for suffix `[k ... L)`, then normal Decode. Transfer valid **full-depth** prefix KV, prefix validity/positions, and all model-specific state needed for suffix and first-token sampling; do not assume hidden transfer is always zero for all model implementations. Huginn's prelude/coda KV and injection/RNG transitions require explicit tests. Correct causal recurrence is a hard invariant; do not clone a loop connector and merely rename its cut variable.

The two axes assign different cells of a token-by-depth work grid, but **do not reduce total model work**. Do not compare `k/L` and `n/R` as exactly equal service time without measuring the resulting GPU work and attention costs.

### Online handoff safety

The old cut is committed before P starts and cannot be arbitrarily revised. For a *new online* method, choose a provisional boundary from **currently observable** measurements, reserve D-side KV/staging capacity and credit, and recheck D admission at the actual transfer frontier. If D refuses before export, safely continue locally to a later valid cut or full P completion; do not submit incompatible NIXL fragments, double-execute layers, drop KV, or partially overwrite an owned destination. All arrival and work scheduling decisions must be recorded. **No future request arrivals or actual future output length** may be used by the online decision.

## 5. W1–W4 workloads: primary 2-GPU matrix

Source preference: real ShareGPT prompt text. Add publicly reproducible code/document text to approximate Azure-Code-style *length/arrival properties*; do not claim to possess original Azure prompt content unless actually present. Use the two official tokenizers, frozen prompt IDs, disjoint calibration/test hash lists, reproducible token-target choices, and preserve source provenance. A previously prepared 200+200 W1 calibration/test manifest may be subselected without replacing or modifying the original.

**Common-model context rule:** Each request must satisfy `prompt_tokens + maximum_output_tokens <= model_limit`. For Huginn: `<=4096`; when output target is 512, prompt `<=3584`. Common-model comparisons use the valid shared support, primarily **800–3,584 prompt tokens**. An Ouro-only 3,585–4,096 prompt stratum is optional and is not reported as matched Huginn data. Do not change Huginn's context or RoPE limit.

| Workload | Primary measured requests **per arm/model/start** | Input token targets | Controlled output target | Arrival / target behavior |
|---|---:|---|---|---|
| **W1 Long-generation mixed** | **32** | 35% 0.8–1.5K; 40% 1.5–2.8K; 25% 2.8–3.584K | 8 × 384, 16 × 448, 8 × 512 (mean 448) | Fixed-seed open-loop Poisson; representative long-generation mixture |
| **W2 Prefill-heavy** | **24** | 2–3.5K, respect each target context | 128/192/256 balanced (mean 192) | Higher P pressure; avoid conflating longer input with D load |
| **W3 Bursty mixed** | **32** | 0.8–2K background + 3–3.5K burst | 256/384/512 in 25/50/25% (mean 384) | Explicit burst amid ongoing Decode; target P/D imbalance |
| **W4 Decode-heavy** | **24** | 0.8–1.5K | 400/448/512 approximately evenly split (mean ~453) | Many active decoders; force test of safe full-PD fallback |

**Controlled outputs:** Use target output lengths and `ignore_eos=True` **only for explicitly controlled long-generation/stress**; report cap-stops vs natural EOS separately. `max_tokens=512` is a cap, not proof that all answers naturally last 512 tokens. W1 Natural-EOS (cap 512) is an optional **clearly separate** follow-up, not a replacement for the required W1–W4 matrix.

**Primary one-start matrix size:** 7 arms × 4 workloads × 2 models = **56 measured configurations**, with `32+24+32+24=112` requests per model/arm, **1,568 total measured request instances** across arms. Given the controlled output targets, expect roughly **590K output tokens** before warmups, EOS deviations, and retries. This is already substantial for Huginn R32. Do not promise full statistical precision or all phases within 12 h before timing calibration.

**Time fallback, declared before measuring:** If preflight estimates the full 56-cell matrix cannot fit before hard closeout, reduce each main cell **uniformly and before looking at candidate performance** to W1=24, W2=16, W3=24, W4=16 (80 requests per model/arm; 1,120 total). Mark `BUDGET_REDUCED_MATRIX` and retain identical prompt IDs, targets, and arrival schedules across arms. Never shrink a single unfavorable arm selectively. If even that cannot fit, prioritize all four workloads for both models with `PD_G1`, `Loop_Static`, `Loop_Dynamic`, `Token_Static`, `Token_Dynamic`, then `DP` and `DP_Chunked` as time permits, and mark any missing mandatory cell `NOT_RUN_TIME_BUDGET`. Do not call a partially completed matrix complete.

## 6. Phases and 12-hour wall-clock budget

All durations include launch/setup overhead. The hard limit is **12 hours from the recorded campaign start**; target normal completion is **10–11 hours**. This is a planning allocation, not a guaranteed execution rate. Protect the last 45–60 minutes for analysis, burner restoration, and safe commit/push.

| Phase | Time target | Execution / acceptance |
|---|---|---|
| **0A: Existing runtime audit, provenance, dataset and isolated calibration** | ~0–1.0 h | Freeze source/branch, investigate DP vs chunked distinctness, prepare W1–W4, estimate actual runtimes; no candidate SLO tuning |
| **0B: Implement missing Token migration and Online Token/Loop selector; correctness gates** | ~1.0–3.5 h | Use existing multi-chunk Loop implementation; exact token/KV/hidden/depth tests on 8/16 prompts then 32/64 request independent seeds; if incomplete, report blockers, do not run invalid arm |
| **1: One-start 2-GPU W1–W4 7-arm comparison** | ~3.5–8.5 h | Full 56 cells if projected feasible, else frozen budget-reduced 56 cells; 1 representative pre-calibrated offered rate per model/workload |
| **2: Targeted rate/seed repeats near SLO knee** | ~8.5–11.0 h | Prioritize W1 and W3, `PD_G1`, best static Loop, `Loop_Dynamic`, `Token_Static` / `Token_Dynamic`; >=2 additional independent starts/rate points only if time allows; include W2/W4 repeat if headroom |
| **Closeout** | by 12.0 h | Save report+raw, classify unrun cells, restore GPU burners, commit/push safely, verify remote SHA |

**Hard stop and safety:** At 11 h, stop launching new long experiments; 11–12 h is reserved for draining active jobs, analysis, artifacts, cleanup and publish. A job that cannot finish within the remaining allowance is not started. At 12 h, do not launch or continue new GPU tests; safely terminate only owned experiment workers if required, retain logs, and restore approved resource policy. Do not claim the original deadline was met if publishing finishes late.

**Reality check:** Implementing correct Token Split and robust live renegotiation may exceed 3.5 h. If this happens, partial completion plus a reproducible blocker report takes precedence over fabricated 7-arm results, weakened correctness, or extending the budget without user approval.

## 7. Required Phases 0–2 in detail

### Phase 0: Operator and correctness qualification (mandatory)

1. Audit existing `adaptive_recurrence_cut` Loop connector and snapshot/full/native streaming controls.
2. Implement Token prefix/suffix migration and validate causal KV and Huginn prelude/coda/injection ownership; independent request IDs and transfer acknowledgements.
3. Implement **real online** policies for Loop and Token, collecting only observable queue, occupancy, Decode slack, predicted P/D work, KV credits, and transfer costs. No offline profile labels or completed-request minima.
4. Implement stable/admission-safe fallback (Loop→later valid Cut or R; Token→later valid prefix boundary or P-complete/full-PD) with explicit state machine and retries; preserve exact prescribed work.
5. Test eight to sixteen requests, multiple seeds, chunk 128/256, lengths 128/512/1536, first-token + bounded decode, then ≥32–64 requests under changing batch order; compare to qualified same-runtime fixed-reference. Record per-request prompt KV/hidden and actual generated token parity, stop reasons, and full-depth counters. A passing microtest is not official HF fidelity.
6. Assert `0 <= cut <= R` or `0 <= prefix_k <= L` and legal boundaries; ensure transfer completion before read. For Huginn, check context and prelude/coda/injection contracts. A timeout, deadlock or unverified transfer means the affected arm is **BLOCKED_CORRECTNESS**.

### Phase 1: representative serving landscape (all W1–W4)

- One **disjoint-calibration-set** rate discovery per model/workload from strong baseline(s), at modest size; freeze requested TTFT and request-average TPOT thresholds and offered rate before held-out candidate data. Include secondary p95/p99 ITL, not only average TPOT.
- Execute the seven-arm W1–W4 matrix on held-out prompt IDs with identical arrival offsets, run seeds, precision, workload and measured output-length targets.
- For `Loop_Static` tune over reasonable frozen cuts using only calibration (Ouro 0–4; Huginn 0/8/16/24/28/32 as initial sweep). For `Token_Static`, tune legal token-prefix fractions 0/25/50/75/100% aligned to chunk boundaries. Record pre-tuning sample size and cost.
- For dynamic policies, log timestamped decision (`cut`, queue, active Decode rows, KV credit, predicted additional D delay, observed SLO slack, fallback), **without looking ahead**. Compare against strongest calibration-tuned static within each axis.
- Keep native `PD_G1` and matched snapshot-full-cut results distinguishable; avoid transfer-protocol confounds.
- Record `TTFT/TPOT/ITL/E2E`, rate/pass counts, `actual output length`, token throughput, queue/backpressure, transport bytes/fragments, HBM, GPU kernel service vs host wall phases, CPU/launch overhead, and actual P/D GPU overlap.
- **One 24–32-request run is diagnostic:** per-run p99 or 90%-pass estimates are noisy, and a passing point is **not** demonstrated maximum stable goodput.

### Phase 2: bounded SLO-goodput and headroom (targeted)

- Use selected W1/W3 knees first, then W2/W4 with remaining time. Freeze the choice on calibration data or explicitly state it is exploratory, not an unbiased final gain.
- Run ≥2 additional independent starts for prioritized comparisons if feasible (ideally 3 total). Evaluate 2–3 arrival rates near 90% joint TTFT+mean-TPOT attainment; report backlog/drain, timeout and all failure counts. Treat timed-out requests as failures in SLO accounting, not silently excluded.
- Compare `Loop_Dynamic` against `Loop_Static`, `Token_Dynamic`, `Token_Static`, `PD_G1`, `DP`, and `DP_Chunked` on equal 2-GPU budgets. If actual repeats are incomplete, report first-start exploratory results only.
- **Offline dynamic composition != Online Dynamic Serving.** Do not claim a prediction model or dynamic oracle unless validated against independent coupled serving traces and tested requests.
- Exit with a transparent verdict for both models independently: `GO_MEASURED_ONLINE`, `CONDITIONAL`, `NO_GO_PERFORMANCE`, `BLOCKED_CORRECTNESS`, `NOT_IMPLEMENTED`, or `NOT_RUN_BUDGET`.

## 8. Fairness, statistics, and controls

- **7-arm total:** `DP`, `DP_Chunked`, `PD_G1`, `Loop_Static`, `Loop_Dynamic`, `Token_Static`, `Token_Dynamic`. No Layer arm. If a baseline cannot be executed correctly within budget, do not silently replace it with a different baseline.
- Each 2-GPU pair runs independently. GPUs 0–7 can serve isolated *correctness* pairs; clean latency on a shared host/fabric must not overlap with other experiments/burners or user processes. Record physical GPU UUIDs and device mapping.
- Separate method *and* protocol labels. Keep original P/D and numerical fixes equal; do not tune one method with privileged token budgets or caching.
- Report exact denominators, number of distinct prompt IDs, trial starts, cohort sizes, aborted requests, EOS/cap termination, and confidence intervals based on independent starts when meaningful. Treat 24/32-request p95/p99 as exploratory, and **do not** bootstrap individual tokens as independent trials.
- No future-arrival knowledge or output-length oracle in online methods. No model weight, BF16, KV precision, R4/R32, or attention semantics changed.
- **Novelty limit:** A Loop-Cut at a matching unfolded logical-layer boundary is representable by a generic layer cut. The Layer baseline is out of this campaign scope, so **do not claim Loop-exclusive novelty**; report the open structural limitation and focus on measured Token-versus-Loop advantages.
- No addition of overlapping GPU kernel/transfer times to create E2E speedup claims. Host-observed phase shares are not GPU critical-path shares.

## 9. Adaptive campaign controls

The controller should maintain a running `budget.json` with: initial monotonic timestamp, elapsed hours, executed/failed/not-run cells, last observed per-model/workload throughput, expected time-to-complete pending cells, reserve-to-closeout time, current GPU ownership, selected fallback matrix, and frozen SLOs/arrival seeds. Recompute estimates from actual durations **before** every new run; skip tests whose minimum estimated time threatens closeout. No retries on completed run names; create a new namespace for every new experiment. Preserve all failures.

Maintain initial scientific priority (no favorable-arm cherry picking): correctness > one comparable W1–W4 pass for all 7 arms/both models > paired repeats at the SLO knee > additional rate points > optional natural-EOS/profiler expansion. Always retain failed and unmeasured matrix entries.

## 10. Deliverables

Write all reports under `/home/esjung/MLLM-EP-github/poc_flashvep/reports/` with descriptive `snake_case` names, e.g.:

- `adaptive_loop_token_pd_2gpu_w1_w4_12h_poc_spec.md` (this specification)
- `adaptive_loop_token_pd_2gpu_correctness_and_protocol.md`
- `adaptive_loop_token_pd_2gpu_w1_w4_matrix.md`
- `adaptive_loop_token_pd_2gpu_online_policy_and_slo.md`
- `adaptive_loop_token_pd_2gpu_final_summary.md`

Required machine-readable evidence: `experiment_matrix.csv` (all 56 cells), `per_request.jsonl`, `per_run_metrics.csv`, `slo_calibration.json`, `online_decisions.jsonl`, `transfer_and_kv_checks.csv`, `gpu_execution_timeline` if obtained, `run_config/seed/source hashes`, `status.json`, `commands.jsonl`, `budget.json`, and full logs including failed startups.

Make a new branch or worktree without touching existing dirty/staged files. Preserve raw data and existing source patches. At completion, commit/push only relevant artifacts, no force push, verify remote SHA. Report actual elapsed GPU time and total wall time separately. Return concise Korean final verdict per model and arm, with explicit missing cells and budget overrun if any.

## 11. Exit criteria

This campaign does **not** require a positive speedup to be scientifically successful. A trustworthy result answers:

1. Are Token and Loop migration both correct across the same multi-chunk, realistic long-generation requests?
2. At the same workload and arrival rate, does the optimal Loop Cut vary with P/D pressure?
3. Can live SLO-safe `Loop_Dynamic` improve over tuned `Loop_Static` and `Token_Dynamic` with **equal 2-GPU resources**?
4. Is any benefit from split granularity rather than mismatched native-vs-snapshot transport, altered compute, or output differences?
5. How does that answer change across W1, W2, W3, W4, and between Ouro R4 and Huginn R32?

If the answer is negative, record the actual negative results and implementation costs rather than relaxing the protocol or SLO. If the full matrix cannot be completed by 12 h, report the exact partial scope and stop safely.
