# Loop-Frontier KV Handoff (B): Fixed-Depth, SLO-Goodput Serving PoC

**Document ID:** `looped_pd_b_slo_serving_poc_spec.md`  
**Proposed in-repository path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_b_slo_serving_poc_spec.md`  
**Source repository:** `jyssys/MLLM-EP`  
**Frozen prior evidence:** commit `b82eabd20a4ec168a5ad9a119c8401f97a565e78`, branch `flashvep/looped-pd-abc-disaggregation-poc`  
**Date:** 2026-10-29  
**Target total wall-clock budget:** **10–15 hours**, target ~13 hours, hard stop 15 hours including startup, correctness, measurement, diagnosis and reporting.  
**Available GPUs:** CUDA GPU IDs **0–7**, single node; no external GPUs.  
**Models:** **Dense Ouro Base 1.4B, fixed R=4**; **Dense Huginn-0125, fixed R=32** (the latter is the *actual pinned configuration of the prior PoC*, not an assumed R=16). BF16 and the original checkpoint, tokenizers, cache semantics, schedules, and generation parameters must be pinned and recorded. **Dynamic depth/early exit are explicitly out of scope**; optional future extension only.

## 0. Decision and non-goals

This is a **focused confirmation and mechanism test of existing Track B only**, not a rerun of all A/B/C, a new LoopKV residency mechanism, a new distributed engine, a model training experiment, or a dynamic-depth PoC. Preserve all previous reports/raw evidence, and avoid changing A/C behavior.

**Main research question:** At equal GPU budget and original model computation, does early transfer of **immutable depth-indexed KV frontiers inside a prefill token chunk** improve the **maximum sustainable serving arrival rate satisfying predeclared TTFT+TPOT SLOs**, beyond existing chunk-overlapped PD and a generic immutable-layerwise streaming comparator?

There are two **separate** verdicts:

1. **Method-level:** B vs **matched PD_G1** and **generic immutable-layerwise streaming**, keeping P:D partition, token chunk size, wavefront/scheduler/graphs, transports, KV layout and workload identical.
2. **System-level:** best **B-based PD** vs strongest **colocated DP/DP_Mixed** at exactly the same 2/4/8 total GPU count. PD-only success is not evidence that disaggregation is the best deployment choice.

A positive fixed-concurrency p95 E2E point, a transfer-only oracle, or a favorable retrospectively chosen SLO threshold **cannot** constitute a GO.

## 1. Exact As-Is vs To-Be dependency diagram

```
PREFILL CHUNK C1, Ouro R4 (illustrative; not equal-duration boxes)

Existing overlapped PD / PD_G1:
    P GPU:    [C1 L1] -> [C1 L2] -> [C1 L3] -> [C1 L4] [C2 computation ...]
                                                      |         ^
                 C1's full-depth KV is valid only HERE+       |
    P->D KV:                                          [C1 KV transfer]  (overlap with C2)
    D GPU:                                        wait for required KV + final hidden/commit
                                                    -> resume first token -> decode

B: earlier immutable depth-frontier streaming *within C1*:
    P GPU:    [C1 L1] -> [C1 L2] -> [C1 L3] -> [C1 L4] [C2 ...]
                  |           |          |          |
    P->D KV:      +-> KV1     +-> KV2    +-> KV3    +-> KV4
                   (subject to CUDA completion, readiness, NIXL bandwidth and
                    immutable KV semantics; overlap with later loops when legal)
    D GPU:       no decode admission until all REQUIRED state/commit is ready;
                 do NOT claim that the first token begins after KV1 only.
```

**Critical correction:** In PD_G1, *C1* KV cannot be transmitted before *C1* completes all required recurrent loops; it **can** overlap with the computation of a later chunk *C2*. B may transfer C1's completed, immutable KV depth plane before C1 finishes all loops, potentially overlapping with later loops of **C1**. It does not remove KV bytes or recurrent FLOPs; it can create more NIXL fragments/event/poll overhead.

Huginn Prelude/Coda planes need independent correctness rules. The shared/overwrite cache layout must **never** expose mutable intermediate KV as immutable. Inspect the pinned `frontier.py`, model-specific `requires_boundary_kv` behavior and RLT patch before acting.

## 2. Previous evidence and exact gaps

From `09_final_summary.md` at frozen commit:

- Ouro P512/concurrency4/output-cap32: B vs strongest equal-2GPU p95 E2E baseline DP_Mixed = **981.602 vs 1070.138 ms**, **1.090x**, conditional paired 95% CI **[1.012,1.153]**. Its moderate-SLO bounded goodput vs DP_Mixed is only **1.034x [0.975,1.143]**.
- Huginn same configuration: B vs DP p95 E2E **4065.456 vs 3295.142 ms**, **0.811x** (B loses); but exploratory *strict ITL<=150ms, TTFT<=5s* bounded goodput B vs PD_G1 **1.310x [1.125,1.580]** (70/72 vs 56/72 qualified). This is **not** open-loop sustainable capacity and not proof against DP's SLO capacity.
- Original-reference exact sequence parity: Ouro **88/100**, Huginn **89/100**. Original-reference *answer accuracy not measured*. Within the fixed B/PD control, bounded output and transferred KV byte identity passed; **do not claim official-model losslessness** until resolved.
- B sends **same total KV bytes** as PD (Ouro 9.664 GB per comparable cohort; Huginn 34.256 GB), but more transfer fragments (Ouro 168->216; Huginn 528->840). More fragmentation can outweigh better early readiness.
- 4/8GPU candidate scaling was one-start discovery, not accepted. Old foreground synchronous-logging comparisons invalid. Only use registered nonblocking evidence protocol.
- Previous Huginn depth is **R32**. Prior output caps often caused 0 EOS completions. Distinguish capped workloads from true natural complete-answer serving.

**Do not reuse prior best-case numbers as acceptance thresholds or tune the new SLO on B's held-out results.** Keep prior work immutable and cite the frozen snapshot.

## 3. Baselines (implemented/verified before measurement)

| ID | GPU placement | Exact conceptual configuration | Role |
|---|---|---|---|
| `DP` | N independent colocated replicas on N GPUs | Existing default scheduling, same pinned model/cache/precision | Basic deployment control |
| `DP_Mixed` | N independent colocated replicas | Existing mixed recurrent P/D scheduling and 256-token chunk option; **not merely 'DP with chunked prefill'**, as ordinary DP also has chunking | Strong colocated control |
| `PD` | P pool + D pool; P+D=N | Existing ordinary PD with completed chunk KV and possible overlap with subsequent chunks | Basic PD |
| `PD_G1` | matched P/D | `--wavefront --chunk 256` with current boundary adapter as required; completed full-depth chunk transfer may overlap following-chunk compute | **Strong causal B control** |
| `GenericLayer` | matched P/D | Generic immutable logical-layer/plane streaming with same legal CUDA event handling, payload, transport pacing and scheduler, **without privileged loop-specific assumptions** | Required novelty falsification control, if feasible |
| `B` | matched P/D | `PD_G1` + `--enable-frontier-handoff` | Candidate |

**GenericLayer caveat:** Recurrent depth planes can be interpreted as unrolled logical layers. If a correctly implemented generic layerwise streaming algorithm **reduces to the same transfer dependency graph as B**, record them as algorithmically equivalent; do not invent a contrived weak comparator or claim loop-exclusive novelty. Independently check whether an unchanged generic algorithm offers the same behavior. A 60–90-minute implementation budget cap applies; if not feasible, mark `NOT_MEASURED` and **no novelty GO**, while preserving PD_G1/B performance evidence.

**'Static PD' label precision:** The PoC label `PD` is **not strictly a no-overlap static copy**: it can overlap *completed-chunk* KV transfer with future chunk prefill. If including a fully synchronous/no-overlap PD for illustration, call it `PD_SYNC` and distinguish it from the existing `PD`, without making it the primary challenger.

**Fairness controls:** same model revision; tokenizer; R; BF16; physical GPU topology and budgets; per-request prompt/output treatment; KV layout/precision; attention backend; graph capture; max batch and KV capacity; initial cache state/prefix cache; network transfer transport and tuning. P:D ratio and tuning must be selected on an **independent calibration workload** for each PD method family, then frozen for all final methods; **the direct PD_G1 vs B comparison must use exactly the same P:D ratio and 256-token chunk/G1 settings**. Never cherry-pick per-run hardware or best seeded result for B.

## 4. Workloads: four fixed, non-cherry-picked families

Use actual **text** prompts and report tokenizer-computed input and actual output lengths. Honor license/access rules; cache or pin source datasets where allowed. If downloading datasets fails, use a clearly labeled, reproducible local real-text corpus—not repetitive synthetic filler—and disclose substitutions. Verify each requested context/output length is supported by both runtime and GPU HBM; gracefully shorten unsupported cells and report `NOT_SUPPORTED` rather than silently truncating.

| Workload | Purpose | Small bounded setup for 10–15h budget |
|---|---|---|
| **W1: Realistic Mixed (PRIMARY)** | Mixed length/role; real serving viability | 100–250 *distinct* ShareGPT/Azure-conversation-like prompts if available, preserving an empirical length histogram; Poisson open-loop arrivals; same frozen trace across arms |
| **W2: Prefill-Heavy** | Larger KV and prefill cost | 80–150 *distinct* code/document prompts; target 1–4K tokens (or valid supported bins), shorter output; arrival rates selected from W1 calibration |
| **W3: Bursty Prefill Interference** | Active decode interleaved with unexpected long prefills | 80–150 prompts, mix of active decode-bearing requests and 10–20% long inputs; alternating base-rate windows and **predeclared** 2x–4x transient bursts; identical arrival timestamps across arms |
| **W4: Decode-Heavy (negative control)** | Expose fragmentation/dispatch overhead | 60–100 distinct requests, longer outputs (target 128/256 where feasible), moderate inputs; terminate at EOS when possible; cap is an explicitly labeled safety ceiling |

**Dataset provenance:** candidate public traces and papers: DistServe (ShareGPT/LongBench, Poisson arrivals, joint TTFT/TPOT SLO), DuetServe (Azure Code, Azure Conversation, Mooncake; Poisson arrivals), and the original B report. Mooncake's average 12K input is **not** a mandatory test for this limited single-node PoC; do not assume Huginn R32 can host it.

**Fair workload rules:** No selection by B's speed. All baselines replay **identical request IDs, prompts, requested output cap, arrival timestamps**, seeds and model settings. Control prefix-cache leakage (reset/disable if possible; record exact policy). Do not create special token patterns that favor early frontier streaming in primary W1; W2/W3 are predeclared mechanism stress tests, W4 a counterexample. Report EOS stops and cap stops separately. If few outputs terminate naturally, primary can be truthfully described as *real-text, bounded-generation serving*, not full-answer completion.

## 5. Metrics and prespecified SLO contract

### 5.1. Primary endpoint — serving capacity, not selected p95 E2E

For each model `m`, workload `w` and **frozen SLO** `(F_m, D_m)`:

- `TTFT_q = timestamp(first emitted token) - timestamp(request arrival)`.
- Per-request `TPOT_q = (timestamp(last emitted token) - timestamp(first emitted token))/(actual_output_tokens-1)` for `actual_output_tokens>=2`; one-token successful requests have TPOT=0 by explicit convention. Do **not** derive internal token timestamps from coalesced output callbacks.
- Joint success indicator `J_q = 1{successful valid completion AND TTFT_q <= F_m AND TPOT_q <= D_m}`; exceptions, timed-out/aborted/OOM, invalid/truncated-by-unexpected-error requests count as failures. Intentional output cap is permitted for bounded-generation workloads and **must be reported as capped**.
- `SLO attainment = sum(J_q)/all arrivals in evaluation window`; report TTFT and TPOT individual attainment too.
- `G_qualified(λ) = (number of SLO-qualified requests arriving in measured window) / measured window seconds` in req/s; separately report achieved completed throughput and offered rate. Drain requests before analysis; do not manipulate denominator by ignoring drain or stuck requests.
- **Primary** `λ*_SLO(m,w,N)`: largest tested or refined **open-loop arrival rate** `λ` with **joint attainment >= 90%**, acceptable error/timeout rate, stable queue/backlog (no persistent upward trend, or clearly label as uncertain), and achieved rate approximating offered rate. If the rate ceiling is not bracketed or runs too short, report **bounded λ* estimate/interval**, not a proven steady-state maximum. Optionally `per-GPU goodput = λ*/N` with N fixed across comparisons.

**Predeclare one primary W1 result for each model and GPU budget.** Use W2/W3/W4 as predeclared supplementary conditions, not post hoc replacements for W1.

### 5.2. Model-specific SLO calibration without peeking at B

Use a small, **disjoint** W1 calibration set, low-load, on `DP_Mixed` and `PD_G1` (without B). Estimate per-model `p90(TTFT)` and `p90(per-request TPOT)`; predeclare e.g.:

`F_m = round_up_ms(1.25 * max_calibration_baseline_p90_TTFT)`  
`D_m = round_up_ms(1.25 * max_calibration_baseline_p90_TPOT)`.

This multiplier is a **precommitted operational choice**, not a paper-established universal constant. Document calibration sample size, observed values, hardware, derived absolute ms thresholds, and freeze them in immutable config **before executing B on held-out W1**. If the calibration produces nonsensical/unreachable thresholds or extremely loose SLOs, annotate and use a separately preregistered application-level threshold; do not tune against B outcomes. Different absolute thresholds for Ouro R4 and Huginn R32 are acceptable; use the **same formula and SLO attainment rule** for both.

Secondary **Strict Streaming** (separate, not used to select the primary winner): `TTFT<=F_m AND max(observed ITL_q)<=I_m` with preregistered `I_m` using baseline-only calibration; lengths must be stratified because max-ITL failure probability increases with output length. Historical Huginn 150ms ITL is **not** automatically the new main threshold.

### 5.3. Mandatory diagnostic metrics for every arm

- `TTFT p50/p95/p99` in ms (per request, arrival-based).
- `TPOT p50/p95/p99` in ms (**per-request averages**; do not pool all token gaps then call it TPOT).
- `observed ITL p50/p95/p99/max` in ms (actual emission-to-emission gaps; if aggregated callback outputs multiple tokens, never invent intermediate timestamps).
- `E2E p50/p95/p99`, mean, optional p99 CI, and timeout/failure counts; **E2E is secondary** and not the acceptance selector.
- Achieved req/s, output tok/s, SLO-qualified req/s, offered λ, joint attainment; offered-vs-achieved and backlog trend.
- Exact prompt/output token lengths, EOS vs cap, P/D queue wait and residence times, KV transfer ready/submitted/completed timestamps, final **exposed handoff stall**, transfer total bytes/count/descriptor overhead, maximum measured HBM per GPU and OOM events, optional GPU kernel timeline (diagnostic only).
- Show p50 and p99 for **both** Ouro and Huginn. A p99 with <100 requests is unstable: label exploratory and supply independent-run uncertainty; target >=200 independent distinct arrivals for key W1 comparison when budget permits, otherwise disclose the limitation instead of manufacturing precision.

**Statistics:** at least two independent run starts on shortlist rates and preferably three for the held-out W1 verification; paired arrivals/prompts across arms with disjoint seeds between independent runs. 95% confidence intervals should respect run/trace clustering and small-N limits. Do not call intervals simultaneous or selection-adjusted if they are not. Report all attempted rates, failed runs, and unselected SLO cells.

## 6. Time-budgeted execution stages (hard wall-clock cap)

**Clock starts before setup, not after first GPU benchmark.** Target **~13.0 hours**, do not exceed **15.0 hours**. Leave at least 45 min for final evidence and safe cleanup. Each stage has a stop condition and writes a checkpoint; optional stages drop automatically if the remaining budget is insufficient.

| Stage | Target budget | Actions | Mandatory gate |
|---|---:|---|---|
| 0. Read-only audit and safe prep | 0.75h | Confirm pinned branch, frozen results, source patches, GPU UUID/topology and free memory; create separate branch/output tree; register budget clock | Correct model pins/R4/R32; isolated run directory |
| 1. Short correctness qualification | 1.50h | RLT `PD_G1` vs B greedy tokens, KV bytes/immutable frontiers, boundary adapter, 20–50 held-out prompts per model; official reference discrepancy triage | If B differs from matched PD_G1, stop B performance and report failure; if HF/RLT discrepancy remains, mark global `FIDELITY_UNRESOLVED` |
| 2. Baseline-only calibration | 0.75h | W1 calibration, SLO freeze; quick λ bracket for both models, DP_Mixed+PD_G1 only | Freeze SLO and workload before B held-out runs |
| 3. W1 primary **2GPU** serving | 4.00h | Adaptive Poisson λ sweep (typically 3–4 bracketing points, prioritize DP_Mixed, PD_G1, B; DP/PD baseline anchor; GenericLayer when feasible), repeat borderline rates, identical arrivals | Primary estimates for both models; stop any arm with parity failure |
| 4. W2/W3/W4 reduced tests | 2.00h | **Focused** two-GPU comparisons at 1–2 preregistered rates using DP_Mixed, PD_G1, B (+GenericLayer if ready); W3 includes same burst schedule across arms | Include negative W4 outcomes and unsupported context records |
| 5. 4/8GPU focused discovery | 1.75h | W1 and one W3 point, matched total budget, default 2P2D/4P4D (optionally limited independent ratio calibration); DP_N versus same-P:D PD_G1 and B | No GO claims on one-start scale points |
| 6. Held-out independent confirm | 1.50h | If time, 2–3 fresh starts near λ* for W1 and strong controls; investigate root mechanism and generic streaming equivalence | Held-out, no post-hoc SLO tuning |
| 7. Analysis, documentation, safe closeout | 0.75h | Results tables, failures, confidence caveats, commits and push | Full 15h hard-stop respected |
| **Total planned** | **13.00h** | Reserve ~2h for loading, transport issues and retries | Hard cap **15h** |

**Adaptive prioritization:** Drop 4/8GPU extras first, then W4 extra rates, then redundant weaker DP/PD points; **never** drop primary W1, matched PD_G1, DP_Mixed, output parity, or the frozen SLO contract. If GenericLayer cannot be implemented safely within 60–90 minutes, defer that comparator explicitly and mark `NOVELTY_UNRESOLVED`, rather than consuming the whole budget.

**Do not hard-code 60–120s per benchmark as proof of steady state.** Start with short calibration runs, then allocate longer measured intervals at the actual SLO boundary. Measure warmup and test windows separately, inspect queue/backlog first/second halves and the drain. If λ* cannot be validated within budget, report `BOUNDED_CAPACITY_ONLY`.

## 7. Physical GPU allocation and isolation

- Available IDs **0,1,2,3,4,5,6,7**; first inspect `nvidia-smi -L`, `nvidia-smi topo -m`, actual per-GPU HBM, and other users' processes. Never kill unrelated jobs or reset a shared GPU.
- **2GPU:** one contiguous verified pair for each run (e.g. 0,1; rotate 2,3 / 4,5 / 6,7 when compatible), DP2 uses 2 independent replicas; PD uses P1:D1. Use paired same-device comparisons or balanced alternating hardware to suppress placement bias.
- **4GPU:** DP4 (four colocated replicas) versus PD P2:D2 by default; optionally P1:D3 or P3:D1 **only via independent calibration**, freeze before confirmation. Same 4 physical IDs for every candidate/control.
- **8GPU:** DP8 versus PD P4:D4 by default; optional quick separate ratio calibration. Use all GPUs 0–7 for the 8GPU comparison.
- Reuse all eight devices productively across campaign stages **without concurrently measuring methods whose GPU, PCIe or host-memory contention would confound comparisons**. Independent side tests can run in parallel only after verifying no cross-traffic impact.
- B/PD_G1/GenericLayer use identical P:D ratio, flags and transport budget. Log NIXL backend, path, bytes, GPU PCIe/NVLink topology and link contention. A stronger tuned PD_G1 is not to be compared against an untuned B (and vice versa).
- Poll every 0.2s or a documented interval for peak HBM (sampling misses subinterval peaks); capture peak allocated/reserved when reliable.

## 8. Fidelity, correctness and quality gates

1. Use official checkpoint revisions and original tokenization / chat template / fixed R / precision. Record source hash of RLT and PoC runtime, HF reference version, model generation settings, seed and attention backend.
2. Check **B vs matched PD_G1** for **exact greedy output IDs and lengths** on all qualifying workload requests when possible; compare transferred KV and final hidden state from a smaller instrumentation test; preserve warmup, reversed-fragment test and coda/boundary readiness checks. Do not apply a low-cost but different cache mode to B only.
3. Reproduce existing **official HF vs RLT** gap separately: past Ouro 88/100, Huginn 89/100 sequences. Investigate mismatches with first-divergence token/logit errors/precision where possible; **do not silently modify model math** to force equality. If unresolved, keep system results explicitly **RLT-internal conditional**, mark no original-model-lossless GO.
4. Natural request correctness: input IDs, output token IDs, EOS/cap distribution and any task accuracy results (GSM8K/HumanEval if measured). Do not invent benchmark accuracy or conflate sequence parity with downstream accuracy.
5. If method-dependent output length differs, **DO NOT** compare performance as equivalent: investigate discrepancy or use a separately labeled fixed-output-count microbenchmark. If B parity fails, stop accepting its serving speedup immediately.
6. `C` track is out of scope. A/C output failures must not contaminate the B study.

## 9. Mechanism and novelty falsification

**Measured chain of causality expected for B:**

`immutable depth KV ready earlier` -> `NIXL transfer submitted/completed earlier` -> `less exposed handoff wait (if critical)` -> `lower TTFT / fewer decode stalls under load` -> `higher predeclared SLO capacity`.

Explicitly measure/plot an event timeline for one successful and one unsuccessful regime. Report if the observed win instead appears to arise from batching perturbation or timing noise. Compare fragment count and CPU polling/submission overhead. Existing `PD_G1` already overlaps completed-chunk transfer with *later chunks*, so this is **not** a comparison against a deliberately synchronous baseline.

**Novelty counterexample:** generic immutable logical-layer KV streaming might implement the exact same idea without using recurrence beyond an indexing convention. If its dependencies, transferred bytes, and timing are equivalent to B, make the honest determination **`LOOP_SPECIFIC_NOVELTY_NOT_ESTABLISHED`**. No claim of a new Looped-specific principle follows just from using a recurrent model.

## 10. Go / No-Go decision rules

A **full GO** requires all of the following:

- B vs PD_G1 parity/correct KV under the same fixed model math; and official-reference fidelity issue resolved or claim explicitly restricted with **no full lossless GO**.
- At a **preregistered W1 SLO** and paired GPU budget, B provides a **reproducible, practically relevant improvement** vs both tuned PD_G1 and generic immutable-layerwise (if measured). Suggested engineering threshold **>=10%** on held-out `λ*_SLO` with uncertainty excluding parity, **not a statistical law**. No post-hoc cherry-picking.
- W1 remains stable under open-loop queueing; tail TTFT/TPOT not traded for hidden failures; achieved throughput and all error/timeout fractions disclosed.
- The best B-based PD is compared to DP_Mixed under **the same total GPU budget**. State explicitly whether PD+ B beats or loses to colocated. PD-only improvement without DP win may be `METHOD_ONLY_CONDITIONAL`, not system superiority.
- A distinct loop-specific mechanism survives GenericLayer comparison, or accept `NOVELTY_UNRESOLVED` and postpone paper-level claims.

Outcomes: `GO_SYSTEM_AND_METHOD`, `GO_METHOD_ONLY_CONDITIONAL`, `CONDITIONAL_FIDELITY`, `CONDITIONAL_CAPACITY`, `NO_GO_PERFORMANCE`, `NO_GO_CORRECTNESS`, `NO_GO_NOVELTY`, `NOT_MEASURED_TIME_BUDGET`. Record per model and GPU count; never convert missing data to GO.

## 11. Required artifacts, implementation practices and closeout

Create a **new** subdirectory under `poc_flashvep/reports/looped_pd_b_slo_serving/` and source module (as needed), retaining prior `looped_pd_abc` reports and raw files **unchanged**. Name files descriptively in snake_case, not `spec.md`.

Required outputs:

1. `00_scope_baselines_and_slo_lock.md` — flags, same-GPU matrix, frozen model-specific SLO numbers and selection protocol.
2. `01_dataset_trace_manifest.md` + machine-readable trace manifest with prompt IDs, tokenizer lengths, arrival schedule, seed, provenance, EOS/cap policy, hash.
3. `02_correctness_and_reference_fidelity.md` — paired B/G1 output parity, HF mismatch evidence, no fabricated accuracy.
4. `03_primary_w1_capacity.csv` — model × GPU × arm × offered λ × repeats × attainment × qualified throughput × queue trend × validity.
5. `04_latency_percentiles.csv` — **TTFT/TPOT/ITL/E2E p50/p95/p99** plus errors and lengths.
6. `05_workload_w2_w3_w4.csv` — prefill-heavy, bursts, decode-heavy results including negative cases.
7. `06_transfer_and_critical_path.csv` — ready/submit/done, exposed stall, bytes, fragments, HBM, uncertainty.
8. `07_scale_2_4_8gpu.csv` — equal-GPU comparisons and P:D ratio with strong colocated controls.
9. `08_novelty_generic_streaming.md` — implemented comparison or explicit reason `NOT_MEASURED` / equivalence.
10. `09_final_decision.md` — independent per-model, per-GPU method/system GO/NO-GO, evidence paths, budget consumed.
11. `reproduce.md`, `commands.jsonl`, `configs/`, `raw/`, `logs/`, evidence SHA manifest and plots (`capacity_vs_arrival_rate`, `TTFT_TPOT_p50_p99_vs_rate`, `frontier_timeline`).

**Engineering safety:** Work on a new feature branch based on frozen experiment branch (or repo-preserving equivalent); save source snapshots/config hashes; do not `git reset --hard`, `git clean`, `killall`, or delete/overwrite existing evidence. Check ownership and dirty/staged changes first. Keep all failures; fix logging only with explicit phase rerun records. No recursive full re-execution if a subset fails. Commit only intended new files, and push the new branch after tests if permissions allow; report exact commit/remote status. If repo is dirty, preserve unrelated state rather than including it.

**Prior art/reference links:**
- Prior locked results: `https://github.com/jyssys/MLLM-EP/blob/b82eabd20a4ec168a5ad9a119c8401f97a565e78/poc_flashvep/reports/looped_pd_abc/09_final_summary.md`.
- DistServe OSDI'24: `https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin`.
- DuetServe ICML'26: `https://proceedings.mlr.press/v306/gao26m.html`.
- SGLang serving goodput/ITL definitions: `https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx`.
- vLLM benchmark definitions (TPOT vs ITL): `https://docs.vllm.ai/en/latest/benchmarking/cli/`.

**Final instruction:** Respect the hard **15-hour budget** and communicate *partial, honest findings* if time expires. Do not claim a novel loop-specific SLO improvement without measured and correct evidence.
