# Looped MoE Expert Affinity and Communication-Aware EP — Discovery PoC Specification

**Spec ID:** `looped_moe_ep_affinity_placement_poc_spec`
**Date:** 2026-10-26
**Repository target:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_affinity_placement_poc_spec.md`
**Report root:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_affinity_placement/`
**Execution:** Single server, available GPUs 0–7, TRUE EP4 and TRUE EP8, BF16. Preserve exact model semantics.
**Decision objective:** Determine whether expert co-activation, recurrent expert-transition affinity, and communication-aware placement create implementable, quality-preserving EP latency headroom across looped MoE models, beyond strong generic EP baselines and previous NO-GOs.

## 0. Non-negotiable constraints

1. Use **exactly these three original pretrained checkpoints** for ALL stages, ALL primary metrics and ALL reports: (i) IFM Ouro MoE 8.78B 500B, (ii) IFM Huginn MoE 8.78B 500B, (iii) Foil-1 final 100B. Do not introduce proxy, mini, alternative checkpoint, quantized replacement, or training-derived model as a substitute.
2. Run TRUE EP4 **and** TRUE EP8 on all three; verify live multi-process/multi-rank expert sharding, dispatch, combine, correct weights and actual inter-rank communication. Trace-only/simulator results are NOT live E2E measurements.
3. Every common metric and primary workload must be reported as separate model × EP degree × batch-size rows. No cherry-picking of the highest-performing model or batch size. For an unsupported/failed configuration, report `NOT_RUN`, `OOM`, `INVALID`, or `UNAVAILABLE` with evidence and explanation. Do not silently omit.
4. Architect a **batch-size-agnostic** placement/dispatch mechanism that works correctly at B=1 and larger B. Optimization is allowed to offer small/zero speedup for some B, but must not depend on knowing an exact B in advance or use a hard-coded policy valid for one B only. Report performance as a curve across workload sizes.
5. Baselines and candidate must preserve the full native attention, router, Top-2 weighting, all Expert GEMMs, loop count and decoding semantics; no layer skipping, stale output, shortened loops, reduced precision, route freezing, unsupported early exit, training or modified network architecture.
6. No performance claim is valid unless official-quality native evaluation is reproduced reasonably well, or a discrepancy is explicitly diagnosed, documented, and the result is marked quality-unverified. 8-problem per-domain pilots are insufficient to establish accuracy.
7. Do not terminate another user's process, disturb existing dirty/staged changes, or assume 100% utilization from returned GPU burners reflects experimental efficiency.
8. Distinguish original static EP, **strong optimized generic EP**, loop-aware proposal and theoretical/simulated oracle. Prior NO-GO does not imply EP has no bottleneck.

## 1. Models, exact topology, and authoritative sources

| Model / checkpoint | Resident params | Expert count per physical MoE layer | Top-k | Physical MoE layers | Unique physical experts | Logical execution topology | Logical MoE invocations/token | Comments |
|---|---:|---|---:|---:|---:|---|---:|---|
| `IFM/LoopedLM-P1-moe-ouro-500b` | 8,780,889,600 | 25 in each of 28 | 2 | 28 | 28×25=700 | `R_28^4` | 28×4=112 | Entire physical stack repeated four times |
| `IFM/LoopedLM-P1-moe-huginn-500b` | 8,783,251,968 | 25 in each of 28 | 2 | 28 | 28×25=700 | `P_8 R_12^8 C_8` | 8+12×8+8=112 | Prelude and coda executed once, only middle 12 physical layers repeated 8× |
| `ShourenWSR/how-to-loop-moe` / `Foil-1` subfolder | **563.2M card; 553.7M paper counting** | 8 in prelude; 64 in recurrent core; 8 in coda | 2 | 3 | 8+64+8=80 | `P_1 R_1^16 C_1` | 1+16+1=18 | Untied attention across recurrent passes, tied expert/router weights in recurrent core |

Important distinctions:
- `E=25` for IFM is **per physical layer**, not one global set shared across all 28 physical layers. `700` counts distinct Expert parameter modules, not number activated at one layer.
- For Foil-1, `64` describes experts in the *single repeated core layer*, and `80` includes two independent 8-expert boundary layers. Its recurrent attention is **not** weight-tied across passes.
- All three route top-2 at each physical MoE invocation. Check precise scoring/normalization conventions in model code (IFM sigmoid top-2 versus Foil softmax/renormalization); do not rewrite scoring to a common version.
- For IFM, resident (total) parameter counts from checkpoint cards include accounting not identical to the paper's **8.0B non-embedding** table. Foil's 563.2M vs 553.7M card/paper discrepancy needs explicit code-based parameter inventory and named parameter-category accounting, **not an unsupported explanation**.
- Confirm actual loaded parameter counts and module paths from checkpoints and code before instrumentation; save architecture manifest and checksum/reference revisions.
- Confirm whether dynamically variable depth exists in these exact checkpoints; default to native documented fixed-depth inference (Ouro 4, Huginn recurrent 8, Foil 16). Do not introduce an adaptive depth policy.

**Sources**:
- IFM paper and benchmark protocol: https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/
- IFM released code: https://github.com/ifm-ai/xllm-loop
- Ouro official card: https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b
- Huginn official card: https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b
- Foil official card: https://huggingface.co/ShourenWSR/how-to-loop-moe
- Foil source: https://github.com/SR-A-W/how-to-loop-moe
- Foil paper: https://arxiv.org/abs/2609.35751

## 2. Research questions and explicitly tested hypotheses

**H1 — Generic EP bottleneck exists at B=1 and larger B.** Quantify true rank stragglers, non-local dispatch, unique destination ranks per token, collective fixed costs and expert kernel execution; determine whether generic communication-aware co-placement can beat static placement while respecting rank compute balance.

**H2 — Within-invocation Top-2 co-activation is structured.** For each physical layer, model, phase and B, test whether expert co-selection patterns exceed a null model preserving marginal popularity and whether low-destination-rank co-placement reduces **real** EP critical-path cost (not only a rank-count proxy).

**H3 — Inter-loop same-token expert transitions have predictive information.** For the SAME physical recurrent layer across adjacent/lagged recurrent passes, measure transition signal beyond popularity, with held-out traces and loop-conditional baselines. Quantify if this information is **actionable** for exact EP: proactive placement/replica choice without on-the-fly weight movement, or prediction used only to plan metadata/buffers while exact current-step router remains authoritative.

**H4 — Looped-specific incremental benefit exists** above a comparably optimized generic EP that has access to physical-layer co-activation but **no loop index or preceding-loop route**. Report a tied-weight/repeated-depth counterfactual as a clearly labeled analytical control, not a fourth model.

**H5 — One adaptive placement policy is valid across B.** Online weights may adapt to routing/load statistics, but offline trace-fit is not allowed to use future evaluation requests; runtime must handle any feasible batch size and maintain exact token/rank semantics.

Out of scope unless a GO gate is achieved: cross-loop coalescing reimplementation, loop-conditioned hot-expert replication revisited without new evidence, output caching/skipping, arbitrary token owner migration involving KV that is not explicitly costed, cross-loop exact future router knowledge, PP, precision reduction, retraining.

## 3. Prior experiment reuse and negative controls

Read, cite and reuse all grounded artifacts from:
- `poc_flashvep/reports/looped_moe_headroom/report.md` and `go_no_go.md` (single GPU expert skipping/reuse, NO_GO_HEADROOM).
- `poc_flashvep/reports/looped_moe_ep_coalescing_replication/report.md` and its supporting EP4/EP8 traces (cross-loop coalescing zero incremental modeled gain; loop-conditioned replication ≤0.75% estimated generic-relative increment, NOT measured E2E speedup).

Preserve and extend **correct working TRUE EP baseline** rather than implementing unrelated EP from scratch if reusable. Audit whether previous trace already contains per-token/per-physical-layer expert IDs, router probabilities, pass index and timing; do not duplicate expensive runs unnecessarily. Never infer exact code paths from memory: inspect repository.

**Controls:**
- S0: original TRUE EP placement and dispatcher.
- G1: optimized generic per-physical-layer co-activation placement using popularity, co-selection, rank balancing and equal HBM, with no recurrent features.
- G2: strongest existing generic load-balanced placement/re-routing/replica control feasible for this runtime (same HBM budget; independently tune generic and loop-aware; avoid a weak strawman).
- L1: recurrent transition-informed placement/rank selection with same constraints (not magic per-loop sharded weight changes).
- L2 (optional): transition-informed metadata-only preparation with exact router validation; all wrong predictions safe. No speculative activation transfer unless full correction and wasted work are included.
- Analytical ideal oracles: unconstrained co-activation placement; constrained fixed-placement and equal-memory; ideal token ownership; realistic movement-aware bounds. Clearly label impossible oracles.

## 4. Runtime readiness and environmental constraints

1. Inspect `nvidia-smi -L`, memory, compute capability, GPU-to-GPU NVLink/PCIe topology (`nvidia-smi topo -m`), available VRAM, CUDA/NCCL versions, source-code revisions, active processes, host RAM and I/O.
2. GPUs `0,1,2,3` and `4,5,6,7` can run separate EP4 cohorts concurrently **only** if resource contention and available memory allow; EP8 uses all `0..7` exclusively. Schedule resource-exclusive timing blocks for reliable comparisons; restore the preexisting burner state only as authorized/expected, and record changes.
3. IFM models are **native xLLM format**, not generic `AutoModel.from_pretrained`; use official `xllm.paper_part1.native_inference.load_native_model` and the published evaluation entry point `eval_paper_part1.py`. Respect required FlashAttention 3 support/architecture. Audit environment and reuse earlier TRUE EP adapter if already verified.
4. Foil uses official model code and `Foil-1` subfolder with `trust_remote_code=True`, independent tokenizer and right-padding restrictions; default to non-padded single-sequence or correct valid-token masking for batches.
5. BF16 weights; prohibit silent CPU offload, lossy casts, swap to TF32 as a replacement for original BF16, expert changes, fake collectives, or proxy fallback.
6. Verify EP4 and EP8 with rank-owner mapping, actual remote bytes, NCCL collectives or equivalent transport, expert GEMM on target GPUs, rank-synchronized timing and full output comparison. 25 experts partition unevenly across 4 or 8 ranks; account for rank counts (25 = 6+6+6+7 for EP4; 25 = 3×7 +4 for EP8 if unreplicated) and balancing choices. Foil boundary layers with 8 experts are structurally different from the 64-expert recurrent layer.
7. If a model cannot execute TRUE EP4 or TRUE EP8 on this hardware, record a hard blocker and precise cause; do not substitute TP or simulated EP without labeling that configuration unavailable.

## 5. Phase Q0 — Publication-quality model accuracy reproduction (MANDATORY, BEFORE optimization verdict)

### 5.1 Evaluation fidelity

- **IFM Ouro and Huginn:** use exact `500b` released artifacts with official evaluation protocol `eval_paper_part1.py --protocol paper-part1`, official tokenizer, prompts, shot counts, generation/length caps, answer extraction/scoring and datasets. Prioritize **MMLU 5-shot, ARC-Challenge 0-shot and HellaSwag 0-shot** as required broad reproducibility anchors, plus **GSM8K, MATH500, DROP** using the official protocol where available. Include exact benchmark configurations, sample counts, reference table values, reference source/table and measured scores. For evaluation tasks unavailable due to distribution/legal constraints, record missing source and missing comparison; do not invent a result.
- IFM paper (500B MoE) publicly reports: ARC-Challenge Ouro 51.33 / Huginn 50.47; HellaSwag 54.82 / 55.10; MMLU 64.03 / 66.81 (percent). Other official Table 5 values must be extracted faithfully from the paper (not guessed). Example protocol: ARC-Challenge 0-shot 1165 test, HellaSwag 0-shot 10042 validation, MMLU 5-shot 14042 test, per official appendices. Note ARC scoring may be token-based; reproduce exactly, do not replace with a generic evaluator without disclosure.
- **Foil-1:** use the *Foil paper's own* held-out LM NLL/perplexity and published downstream multiple-choice protocols (e.g., paper Tables 5–7, associated code), with exact data split, tokenization, shots and scoring. Note 100B final-window **training** loss = 2.499 nat as reported for Foil-1 in paper Table 4; **do not claim it is a held-out NLL target**. The official Foil card warns the `forward(labels=...)` loss includes router auxiliary losses and labels must be shifted; compute pure LM NLL from `logits[..., :-1, :]` and `input_ids[..., 1:]` over real tokens, with padding handling. Reproduction target must be a metric actually comparable with a published reference, not a cross-paper benchmark invented for Foil.
- Accuracy benchmarks need meaningful dataset sizes, not 8-question pilots. Attempt full official evaluations where available; record partial coverage, exact uncertainty (binomial/Wilson for accuracy where appropriate), sampling uncertainty and deviations. Do not assert accuracy is reproduced merely because 105/105 numerical correctness checks pass.

### 5.2 Correctness matrix

For **every model × EP4/EP8**:
- Compare TRUE EP outputs to official native/one-rank reference on deterministic fixed tokens, valid sequence lengths, Prefill and Decode; logits error distribution, top-1 token agreement, route indices, gate weights, recurrent pass identity and numerical tolerances. Use multiple contexts and multiple prompts, seed control, strict per-step KV correctness, and per-layer/loop sampling.
- Separately show native baseline task score, TRUE EP baseline score, candidate score when implemented, official paper score when protocol matches. The candidate must not degrade model quality; exact placement/rank changes ideally return the same expert set and nearly the same logits to BF16 roundoff.
- If native and EP backends have systematic discrepancies, halt any 'verified accuracy' or real speedup claim until investigated. Do not force paper target by changing decoding hyperparameters without documenting it.

Deliver `accuracy_reference.csv`, `accuracy_native.csv`, `accuracy_ep.csv`, `accuracy_reproduction.md`, `correctness.jsonl`; all **three models** and both EP degrees must appear.

## 6. Workload matrix and batch-agnostic criteria

- **Required batches:** B ∈ `{1,2,4,8,16}` for each of three models and EP4/EP8; do not optimize solely for one B. Add B32 (and beyond) only if memory and time permit; record capacity ceilings instead of eliminating model rows.
- **Required input contexts:** `{128,512,2048}` tokens (truncate as appropriate within each model's supported context); longer `{4096,8192}` as optional context-valid stress cases. Foil's documented context length is 4096.
- **Required phases:** Prefill and autoregressive Decode independently, plus full E2E request latency; use a standard output-token count (e.g., 32 and 128 tokens where feasible), record actual requested versus produced tokens. Keep exact forced-length timing separate from normal generated-output quality.
- Measure both per-request latency (TTFT, TPOT, E2E, P50/P90) and system behavior (makespan, throughput), including B1. Use multiple domains (knowledge, math, code, general text) and fixed-seed prompts, plus optional synthetic shape loads labeled as synthetic.
- Pre-register identical datasets, batch compositions, context lengths and sequences across methods. For variable batches, avoid policy-specific hard-coded thresholds; an adaptive implementation must accept varying B from one request to the next and a mix of sequence lengths without correctness failure.
- Main metric coverage: **all combinations** of 3 models × 2 EP sizes × 5 B sizes × 3 required lengths, measured with warm-up and **at least 3 isolated timed repetitions** per viable condition. For staged profiling cost, sample representative subsets but retain the full mandatory top-line matrix, with missing-data statuses explicit.
- If input/output cap or memory makes a combination infeasible, log capacity/status and still include it in the completeness table; never label an unavailable configuration as a zero speedup.

## 7. Phase Q1 — TRUE EP critical-path and rank-level instrumentation

Instrument same-revision native TRUE EP4 and EP8 across ALL three models. At each physical MoE invocation capture:

```
model, split, run_id, request_id, logical_token_id, phase, B,
physical_layer_id, recurrence_pass_id (or boundary_flag), logical_depth_id,
router_topk_ids, router_weights, local_owner_rank,
expert_owner_rank(s), unique_dest_ranks, remote_dispatch_bytes,
router_start/end, dispatch_start/end, expert_ready_start,
expert_gemm_start/end, combine_start/end, release_time,
rank_per_expert_token_counts, peak_hbm, cpu_scheduling_overhead
```

Time using synchronized and carefully calibrated CUDA events/profiler markers. Do not sum overlapping nested events as exclusive costs. Trace volume: trace-gate/sample if needed but guarantee unbiased sampled coverage and safe memory/overhead. Report instrumentation overhead versus no-hook runs.

**Required metrics (for every model/EP/B/phase, aggregated by loop and physical layer):**
- Count of routed tokens, valid masked/padded tokens, Top-2 distribution and entropy; per-expert load histogram, load CV, max/mean, Gini and imbalance (load and measured *time*).
- Rank idle fractions, straggler waiting in microseconds, max-rank minus mean-rank *measured runtime*, critical-path share, dispatch/combine latency, communication payload bytes and messages, active ranks/participating ranks.
- Per-token distinct EP destinations, remote fraction, duplicate destination fraction for Top-2, communication latency distribution, BF16 Expert GEMM utilization/occupancy where profiler exposes it, peak HBM, per-rank allocation fairness.
- EP4 vs EP8 real scaling vs EP1/native and equal total GPU-budget comparator when feasible; equal GPU budget and shape must be reported, not conflated with identical per-rank batch.
- Cross-physical-layer vs within-physical-layer recurrent comparisons; distinguish boundary layers (Foil and Huginn) from repeated body.

## 8. Phase Q2 — Within-layer Expert co-activation discovery

For each **model, physical layer, pass, phase, B, domain**:
1. Build weighted/unweighted expert co-selection matrix `C_l(i,j)` from coactivated top-2 expert IDs of the SAME token invocation, counted once per token.
2. Build per-expert marginals and proper independence/surrogate nulls that preserve layer, loop/pass, batch and input-domain marginal frequencies and number of choices; use feasible permutation/randomized pairing while respecting distinct top-2 choices. Report confidence/CI and variation across prompts.
3. Metrics: pair lift, PMI (bounded/smoothed), coactivation strength, average unique destination ranks at existing placement, remote bytes/token, destination fanout, rank-level bottleneck, and gain from ideal same-rank pair co-location.
4. Split calibration and evaluation **by request/prompt** (not by token from same prompt) and report held-out coactivation stability; run an independent evaluation workload, avoid oracle access to live future choices.
5. Compare static round robin, popularity-only, generic graph partition/balanced partition, per-loop oracle placements (counterfactual) and exact feasible fixed placement; common expert memory and compute/rank constraints. Re-mapping expert placement changes GPU memory layout; count startup/migration cost and amortization; do NOT claim per-loop re-partitioning for free.

A decrease in unique destination ranks is a mechanism metric, NOT a wall-time speedup guarantee. Co-location can increase one rank's expert GEMM critical path even as bytes decrease.

## 9. Phase Q3 — Inter-loop expert transition / recurrent affinity

Only pair routes for **same request, token identity, same physical recurrent layer** across valid successive passes. Do not compare unrelated physical layers or mix prelude/coda with repeated core.

For `r -> r+1` and multiple lags calculate:
- Conditional Top-k set transitions `P(S_{r+1}|S_r)` and per-expert `P(j∈S_{r+1}|i∈S_r)` (multi-label, not incorrect scalar single-expert model).
- Exact overlap, intersection-over-union, repeated-expert rate, transition entropy, normalized conditional mutual information, excess predictive accuracy versus per-pass marginal/pattern baselines, and per-loop confidence calibration.
- Hold-out next-expert Recall@2/@4/@8, top-2 set prediction precision/recall, negative log likelihood or calibrated probability if predictors used; compare previous-loop, same-pass popularity, domain-only and shuffled controls.
- Recompute for B=1,2,4,8,16 and across domains; report variation by pass index, physical layer, phase, context length, long-distance transitions.
- Assess actionable lead time: correct next-loop token hidden state does not exist until previous loop and intervening layers execute. Predictability ≠ ability to dispatch activation early. Prediction may prepare rank selection, metadata, buffers or resident replica choice; exact router must run and validate, and mistakes use correct fallbacks.
- Demonstrate optional runtime command/data schedule feasibility with actual dependency DAG and contention, not speculative perfect pre-gate claims.
- H3 GO requires both meaningful **out-of-sample increment over popularity** and corresponding realizable critical-path reduction, not high route repeat rate alone.

## 10. Phase Q4 — Communication-aware placement and re-routing headroom

Objective: minimize measured EP critical-path and communication overhead subject to exact routing, pinned weights / explicit replica budget, per-rank memory cap, rank GEMM congestion, topology, and live-batch variation.

Candidate classes:
- **P-generic:** global/per-physical-layer Top-2 coactivation placement with one static resident weight for each expert, rank-balanced and topology-aware.
- **P-recurrent:** same physical-layer coactivation + recurrence transition signal, but only choices executable under **actual** fixed/replicated expert residency; no magic shifting of one expert to a different rank per pass.
- **R-exact:** optional safe rank selection among pre-existing replicas, preserving route IDs and weights, with same physical memory budget as generic replica control. No large-scale replication solely to win the oracle.
- **Owner-aware:** optional token-owner migration/affinity analysis. Owner migration includes Attention/KV movement, residual state, combine return routing, synchronization and correctness; never infer EP benefits from uncosted token-owner changes.

**Oracles, separately reported**:
- O0 unconstrained ideal co-location (non-implementable lower bound).
- O1 fixed-placement HBM and topology constrained, trained on calibration traces / validated on held-out requests; generic.
- O2 same budget and constraints with pass-aware information; recurrent.
- O3 optional perfect routing foresight, labeled unattainable, to bound predictive headroom.
- O4 actionable/wall-time calibrated, including placement setup/migration amortization, dispatch, GEMM queuing, combine, synchronization, invalid predictions and latency crossover.

Verify measured per-configuration placement using actual EP implementations only after headroom threshold is met. Prefer microbench to calibrate `O4`, but NEVER report microbenchmark or model/simulator oracle as 'measured E2E speedup'.

Critical decomposition:

```
S_static_to_generic = T_static / T_generic
S_generic_to_loop  = T_generic / T_loop_aware
S_static_to_loop   = T_static / T_loop_aware
```

Show all three in every valid model × EP × B row, for Prefill, Decode and E2E as supported. Never present strong generic gain as proof of loop-aware novelty.

## 11. Batch-agnostic design validation

- Use feature-driven choices (measured load, rank occupancy, co-activation matrix, communication costs, calibrated latency) instead of a fixed Batch threshold. No specialization requiring exactly B1 or B4.
- Ensure fallback to original EP for any B when estimated gain is non-positive, when loads change, or when a route has no local/replicated destination; fallback correctness and decision overhead must be measured.
- Offline placement may be static but must be evaluated under multiple B; if online changes are needed, show zero-weight-migration rank choices among pre-resident replicas or explicitly cost weight migration.
- Test a mixed sequence of batch sizes and lengths, not just independent fixed-B sweeps. Report throughput–latency Pareto curve and policy overhead/memory.

## 12. Accuracy gates and release criteria

**A0 — Checkpoint and official reproduction:** source hashes and exact model shapes validated; primary publication benchmarks match within predeclared statistically meaningful tolerance and protocol differences logged; not all models have the same appropriate paper benchmark; give source-matched references per model.

**A1 — TRUE EP fidelity:** each model × EP4/EP8 passes native-vs-EP logits/tokens/route checks within justified tolerance across comprehensive examples; failures block speedup credibility.

**A2 — Exact candidate:** modified placement/rank selection retains selected experts and token outputs relative to matched TRUE EP baseline; report NLL difference and sample-level downstream paired changes with CI, not just top-1 token fidelity.

**B — Oracle headroom:** realistic E2E actionable oracle >=1.15× vs **optimized generic EP** in an identifiable nontrivial workload regime, plus strong held-out recurrent-specific predictive/placement signal, qualifies as GO candidate. 1.10–1.15× = CONDITIONAL requiring compelling actual-cost evidence; <1.10× = NO_GO for proposed recurrent-specific approach. Existing generic EP speedup independently reportable without calling it recurrent novelty.

**C — Actual implementation (only if B meets GO):** isolated repeated live E2E improvement with uncertainty on both EP4/EP8 where feasible; ideally >=1.10× in the claimed workload regime, no quality regression, without unacceptable regressions at other batch sizes. A candidate may still NO-GO if measured E2E/latency/accuracy underperforms the oracle.

**D — Integrity:** all three models, all primary metrics, all viable B/EP configurations present; all blockers listed and no synthetic results masquerade as measurements. If native quality differs materially from paper, label `QUALITY_REPRO_BLOCKED` even when traces exist.

## 13. Reporting and artifacts (MANDATORY completeness)

Create a reproducible run tree under `poc_flashvep/reports/looped_moe_ep_affinity_placement/`:

```
report.md
summary.md
go_no_go.md
model_architecture_and_sources.md
accuracy_reference.csv
accuracy_native.csv
accuracy_ep.csv
accuracy_reproduction.md
correctness.jsonl
environment_and_topology.json
baseline_e2e.csv
baseline_stage_breakdown.csv
router_trace_manifest.json
router_trace_sample.parquet (or safe equivalent)
coactivation_matrix_summary.csv
transition_affinity.csv
rank_imbalance.csv
placement_oracles.csv
batch_scaling.csv
quality_vs_latency.csv
runs/ (configs, commands, commit SHAs, logs, profiler samples)
figures/ (coactivation/transition heatmaps, occupancy, timing, speedup-vs-B, throughput-latency)
```

**Required report tables:**
1. Fixed architecture comparison of all 3 models, physical vs logical depth, top-2, total experts, checkpoint SHA, native revision.
2. Native quality vs paper reference for each model and its OWN official comparable metrics; complete scores and sample counts; EP4/EP8 fidelity and scores; Candidate if implemented. Record unavailable references without invention.
3. **Full status matrix** for every model × EP4/EP8 × B1/B2/B4/B8/B16 × input length: accuracy status, baseline E2E, throughput, TTFT, TPOT, dispatch, combine, expert, idle/straggler, unique destinations, bytes, HBM, candidate/oracle status, relevant CI. Inapplicable stage metrics explicitly `N/A` with reason.
4. Generic vs recurrent signal on held-out prompts: co-activation lift, PMI, transition recall, conditional entropy/mutual information, shuffle controls, route-locality and error bands.
5. Static vs optimized generic vs loop-aware constrained oracles and ACTUAL (if implemented): three speedup decompositions and end-to-end cost attribution for each model EP and B.
6. **Conclusions separated per model and per method**, plus overall reproducibility and novelty summary. A pooled average is supplementary, NEVER the only answer.

Never fill missing values with zero, guess paper accuracy, silently cherry-pick Foil/IFM, or merge Foil's 8-expert boundary layers into its 64-expert recurrent Expert Pool. Name all experiments and attach raw commands/seeds/splits.

## 14. Execution workplan and limited resources

P0. Survey previous code/results, lock checkpoints and source revisions, inventory model modules, GPU topology, BF16 runtime, official eval compatibility.
P1. Reproduce quality (official native first; TRUE EP4/EP8 fidelity/quality) and issue early diagnostic report if paper quality cannot be reproduced. Perform proper full-sized/official evaluations rather than only toy pilots.
P2. Collect TRUE EP critical-path + route traces for all model×EP×B; benchmark baseline wall time without instrumentation.
P3. Coactivation and conditional transition analysis with prompt-held-out evaluation and neutral nulls.
P4. Equal-memory generic vs recurrent placement + rank-selection oracles; valid wall-time model.
P5. Implement a minimal exact runtime change only if strict GO and robust accuracy; otherwise NO-GO without overengineering.
P6. Produce report, tables, plots, correctness, raw data, GO/NO-GO. Audit completeness. Commit/push without destroying unrelated changes; verify remote SHA; disclose any blocked step.

Eight GPUs available; do not pretend EP4×2 equals one EP8 measurement; avoid simultaneous contention in headline benchmarks. Save provenance. Prioritize accuracy reproducibility and critical-path discovery over unwarranted optimization.

## 15. Interpretation guide

- If `S_static_to_generic` is large but `S_generic_to_loop≈1`, this is a **generic EP optimization opportunity**, not evidence for Looped MoE-specific architecture.
- If cross-loop transitions predict expert identities but no legal/cheap rank ownership or prefetch path exists, the finding is **structural correlation only**.
- If co-location reduces inter-rank bytes but increases per-rank GEMM queueing, declare that trade-off rather than reporting communication-only wins.
- If strong results occur only on Foil's 64-expert R16 core and not on Ouro/Huginn's 25-expert recurrent layers, say so; do not generalize across model families.
- If official accuracy cannot be reproduced, serving performance can be studied **as conditional**, but not presented as demonstrated quality-preserving acceleration.
- If there is no actionable oracle >15% incremental E2E gain over strong generic EP, conclude a narrowly scoped NO-GO; do not claim Looped MoE or generic EP has no optimization possibilities.
