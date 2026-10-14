# Looped MoE × Physical Expert Parallelism: Feasibility and Critical-Path PoC

**Status:** experimental specification / kill-test, **not** a proposed final optimization.  
**Date:** 2026-10-14  
**Suggested repository:** `/home/esjung/MLLM-EP-github`  
**Target specification path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_poc_spec.md`  
**PoC output folder:** `poc_looped_moe_ep/`  
**GPU constraint:** **physical GPUs 4, 5, 6, 7 only**. Always use `CUDA_VISIBLE_DEVICES=4,5,6,7`; verify the logical-to-physical map at runtime.  
**Preferred regime:** low-request/B1 inference, BF16 where supported; no expert approximation, no training.

## 0. Decision to make

**Research question:** In an *authentic pretrained Looped MoE*, do loop-dependent expert-routing distributions produce a **measurable EP critical-path penalty** that a globally optimized **single static expert placement** cannot avoid?

A successful **Stage A routing-divergence result does not constitute a GO**. Routing distribution shifts, different hottest ranks, more bytes, and a simulated zero-cost oracle are not sufficient. This PoC must establish (i) physical EP execution; (ii) actual bottleneck size; (iii) measurable, held-out headroom beyond a strong global static baseline; and (iv) a plausible improvement that is not eliminated by weight movement or coordination overhead.

### Main falsifiable hypothesis

- Shared expert tensors are reused through several logical loop passes, but hidden states and possibly loop-conditioned or untied routers cause different expert demand per pass.
- A static mapping `expert -> EP rank` faces conflicting loop-wise demand vectors.
- The resulting *per-loop* imbalance costs real latency, even when aggregate expert demand looks balanced.
- **Crucially, the best single static placement still incurs this cost**, and a meaningful portion can theoretically be recovered without impossible per-loop weight copies.

### Essential negative controls

- Strong globally optimized static placement, not merely contiguous expert IDs.
- Full critical-path timing (not routed-token counts alone).
- Reproducible output equality across **weight-preserving placement permutations**.
- Ordinary nonlooped/shared-pool MoE negative/control traces when feasible, to test whether observed placement issues are uniquely recurrent.
- Calibration/held-out separation to prevent oracle leakage.

---

## 1. Motivation and exact scope

Conventional MoE: distinct sparse layers commonly own different expert tensors, allowing separate placements by layer. Looped MoE: a shared physical expert pool is invoked in several logical passes (the exact sharing topology depends on the checkpoint). Thus a *single fixed physical owner mapping* serves different loop-index-conditioned workloads.

**Important:** expert sharing is not necessarily sharing of the router, attention, or layer norms. Inspect each selected checkpoint and explicitly report which weights are shared. For example, Foil checkpoints have looped/shared expert pools but attention is *untied across passes*. LOOM explores per-loop routers; do not assume all looped models have independent routers or that increased diversity automatically means worse EP latency.

The target is **MoE EP inference**, not generic loop early exit, generic EPLB, generic prefetch, routing modification, training, or weight replication. The earlier project's MultiBD × EP result was NO-GO because rising logical D+C bytes did not degrade actual EP time per useful token; apply the same evidence standard here.

---

## 2. Prior work and claim boundaries (read before implementation)

| Work | Known contribution | This PoC must not claim |
|---|---|---|
| Sparse Layers are Critical to Scaling Looped Language Models (2026) | Looped-MoE routing divergence and model-side scaling / early exits | Discovering that loops select different experts |
| How to Loop MoE: Flatten the Experts, Untie the Attention (2026) | Foil architecture: more experts in smaller tied pools, untied per-pass attention | Foil or architectural flattening |
| Tying the Loop (2026) | Shared expert tensors with layer-specific attention / routers in tied MoE variants | Generic expert-weight tying |
| LOOM (2026) | Loop-stabilizing recurrence and per-loop routers to diversify expert selection | Loop-specific routing itself |
| Existing EPLB / LLEP / expert placement | Generic MoE load-balancing and expert redistribution | Generic load balancing, migrating experts, oracle placement alone |

Novelty candidate **only if validated**: a *recurrent-shared-expert placement conflict that survives global best-static optimization* and consumes sufficient **physical EP critical-path** time to warrant a practical remedy.

References:
- https://arxiv.org/abs/2605.09165
- https://arxiv.org/abs/2609.35751
- https://huggingface.co/ShourenWSR/how-to-loop-moe
- https://github.com/SR-A-W/how-to-loop-moe
- https://arxiv.org/abs/2606.16825
- https://github.com/epfml/looped-moe
- https://arxiv.org/abs/2610.01153
- https://github.com/hed-ucas/LOOM
- https://proceedings.mlr.press/v306/nguyen26m.html (LLEP)
- https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md

Maintain `prior_overlap.md` and verify claimed source availability/revisions; do not assert an unverified absence of Looped-MoE × EP prior work.

---

## 3. Model candidates: verify instead of assuming

**Primary candidate for rapid authentic loop characterization:**

`ShourenWSR/how-to-loop-moe`, **`Foil-1` subfolder** (published final weights). Model card reports 563.2M parameters, 64 experts per repeated core expert pool, top-2 routing, 1-layer looped block executed 16 times, with untied attention across passes and 4096 context. Thus the same **shared core expert pool** is repeatedly invoked, giving many loop phases to characterize. Pre- and post-loop MoE layers, if present, must be excluded from *shared-pool loop-divergence metrics* and included in end-to-end timing. Validate code and checkpoint geometry directly before assuming these details.

**Secondary structural candidates:** `Foil-2` (32 experts, two-layer ×8 passes), `Foil-3` (16 experts, four-layer ×4 passes), and `Base` (8 experts, eight-layer ×2 passes), all **different trained checkpoints**, not interchangeable `H` settings for a single checkpoint. Do not use model-to-model differences to estimate the effect of changing loop count on fixed weights.

**Fallback fast smoke test:** `ml-ryanlee/looped-moe-16x2-e8k2-32L-d512-1e18-a100`, a small 206.7M-param pretrained Looped-MoE with 8 experts/top-2, 16 layers ×2 passes and `trust_remote_code`. Model card context limit is 1024. This is a proof-of-shared-loop/routing smoke test, **not** representative evidence of H100-EP4 performance.

**Optional more realistic candidate:** Check whether actual downloadable weights **and valid inference runtime** for a larger pretrained Looped-MoE (e.g., Loopie or LOOM) are now genuinely available. A paper, logs, README mention, or placeholder repository is NOT evidence of released model weights. Do not wait indefinitely or start training a model to solve checkpoint availability.

### Representativeness caveat — mandatory

Foil and the small 0.2B fallback fit comfortably on one H100. Forcing EP4 across four H100s can create **artificially communication-dominated inference** that would never be a reasonable deployment. Treat such runs as *mechanism tests only*. No publishable EP-efficiency GO should be awarded solely on these tiny checkpoints. Look for a realistically EP-relevant weight/memory/workload regime; otherwise return `SMALL_MODEL_ONLY` or `REPRESENTATIVENESS_BLOCKED` with the observed structural results.

### Checkpoint validation

Before heavy work, record:
- exact HF repository/subfolder/revision/license and actual model files;
- number of unique physical sparse pools, repeated logical sites, passes, experts/pool, top-k, expert tensor bytes, router sharing, attention sharing, KV-cache semantics;
- checkpoint state mapping and whether weights are tied by *storage alias* rather than copied per pass;
- official inference path, tokenizer, context and max valid pass count;
- single-GPU greedy generation or next-token logit parity under repeat runs;
- memory footprint versus four H100s.

**Do not change the trained loop count `H` as a primary control.** H=1/2/4/8 is valid as a *loop-position trace prefix or specifically supported official adaptive-loop inference* only; changing the effective architecture may change quality and attention/KV semantics. The main intervention varies **expert placement for a fixed checkpoint and its native H**.

---

## 4. Hardware, physical execution and layout

Use ONLY:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
nvidia-smi
nvidia-smi topo -m
```

Inspect occupancy first. Never touch users on other GPUs; preserve unrelated jobs. Record GPU UUIDs and the physical/logical rank mapping.

**Preferred truthful physical path:** TP1 (or the minimal correct dense parallelism) with four ranks in EP4, expert-weight sharding and actual token dispatch to owner ranks, while preserving each request's autoregressive/decode state and cache semantics. A valid alternative is TP4+EP4 if dictated by the integration. These layouts have different costs; record TP/DP/SP/EP/PP exactly and **do not assert `TP4+EP4=16 GPUs`**. One B1 request must actually traverse a four-rank EP group; four independent replicated single-GPU requests is not EP4.

Physical EP4 PASS requires all of:
1. one expected owner per shared expert (unless explicitly testing a separately labeled replication control);
2. verify owner tensors numerically match the pinned checkpoint;
3. actual Dispatch → owner-local expert GEMM → Combine across GPUs 4/5/6/7;
4. all ranks participate in the same correctly scoped group;
5. correct loop index and expert identities for every trace;
6. output/logit parity to the single-device reference (predefined tolerances) and identical greedy generation if numerical parity permits;
7. weight-preserving static placement permutations produce equivalent output and routing, up to numerical tolerance, so that comparisons are legitimate.

If only simulated placement is possible, label `STRUCTURAL_ONLY`, never a physical speedup. If authentic EP4 integration requires rewriting the core attention/loop runtime, return `RUNTIME_BLOCKED` after the bounded feasibility attempt.

**Never assume off-the-shelf vLLM/SGLang/Transformers supports a custom Looped-MoE architecture in EP4 merely because its model card displays generic serve commands.** Inspect real model registry, tied tensors, repeated forwards and dispatch kernels. Reuse proven EP instrumentation from `poc_dllm_ep_comm`, `poc_dllm_attn_moe_pipeline`, `poc_multibd_moe_ep` where compatible, not their model-specific assumptions.

---

## 5. Frozen workload and measurement lanes

Use a frozen, non-cherry-picked prompt set, with e.g. **8 calibration + 24 held-out** requests (or another explicitly fixed comparable cohort). Prefer natural short-/medium-context decode tasks within model-card limits. For tiny base models, use repeatable prompt/continuation/perplexity probes rather than claiming meaningful reasoning accuracy from a weak checkpoint.

Primary **B1 low-request**, optional B4/B8 only if independent-request batching is supported correctly and reported separately (B>1 may create a substantially different expert occupancy regime). Keep native trained loop count fixed.

Use distinct lanes:
- **clean:** wall-clock E2E / prefill / decode / per-token latency, no tracing hooks;
- **light event-only:** CUDA-event stage timing with deferred resolution, no per-loop global sync;
- **heavy trace:** per-rank/per-loop/per-layer expert route and owner mapping, not headline wall-clock timing.

Run-order and placement-order counterbalancing are required where practical. Validate observer overhead: target ≤5%; if >10% or confidence bound crosses 10%, mark fine-grained latency attribution untrusted, retain clean E2E only. Run a bounded number of warmups and repeats, report them.

**Quality/correctness:** Changing `expert -> rank` must not change the mathematical model or greedy outputs. A trained checkpoint's native loop count should be respected. No quantization or expert approximation, no pruning, no change to router scores. Compare output tokens, representative layer outputs, per-loop expert IDs, token counts, and NFE/forward counts where relevant.

---

## 6. Stage 0 — Feasibility and model/runtime gate

Produce `model_runtime_audit.md` before substantial EP integration.

- Verify live checkpoint files (Foil-1 first), architecture, native loop count and shared core expert tensor identity.
- Verify inference runs on one GPU and a route hook exposes `shared_pool_id`, `physical_layer_id`, `logical_loop_id`, `token/source_rank`, `expert_id`, `router_weight`.
- Verify a four-GPU physical EP4 path is available via a small bounded bridge or supported framework.
- Verify the selected checkpoint is meaningful for a four-H100 hardware finding; label small-model-only clearly.

If no runnable pretrained looped checkpoint or no bounded physical EP path exists, exit `CHECKPOINT_BLOCKED` or `RUNTIME_BLOCKED` and preserve the useful audit. Do **not** create a fake looped checkpoint by arbitrarily repeating pretrained Qwen layers and call it equivalent.

---

## 7. Stage A — Is there real loop-specific routing conflict?

At a **fixed native loop count** (e.g., Foil-1 H=16), trace the same **physical expert pool** at each logical pass `h`. For every independent request, token step, shared pool, h and source rank, collect:

- expert assignment histogram and normalized frequency `p_h(e)`;
- top-k sets and weighted routing probabilities (if accessible);
- expert co-activation, active expert union, tokens/active expert;
- owner rank load histogram for the initial placement;
- max/mean rank assignment load, rank CV, hot rank identity;
- source→destination edge count/fanout, logical D+C payload bytes;
- real prefill vs decode flags; input/output tokens and natural stopping;
- per-loop routed-work cost estimate and stage timing, if physical runtime already works.

**Divergence metrics:** JSD between `p_h` distributions, overlap/weighted overlap, loop-to-global JSD, and correlation/hot-rank transitions. Use a properly weighted pooling of *independent tokens/requests*; do not treat layers or loops as statistically independent samples. Normalize for live-token count and source rank.

Critical distinctions:
1. Router diversity ≠ rank-load imbalance.
2. Rank-load imbalance ≠ critical-path time loss.
3. Frequent hot-rank changes ≠ best-static placement conflict.
4. High KL/JSD on tiny or nearly empty loops may be sampling noise.

**Stage A early stop:** If routing differs but max-rank load and physical stage cost remain nearly stable, return `NO_LOOP_EP_SIGNAL`. If loop distributions do not differ, return `NO_ROUTING_DIVERGENCE` for that checkpoint (not a universal Looped MoE claim).

---

## 8. Stage B — Best-static vs per-loop oracle (with honest labels)

The key question is NOT whether an oracle can optimize each loop. It is whether **one good globally static placement cannot already achieve almost the same performance**.

For each shared physical expert pool, define `x_e ∈ {0,1,2,3}` (expert owner). Enforce identical static assignments across all logical passes of that shared pool, correct expert cardinality/bytes per rank, and memory constraints.

Compare at least:

**S0 Native static:** official contiguous/round-robin mapping on physical EP4.

**S1 Strong global best-static:** optimize a single placement for the **whole calibration request set across all loop positions**, considering per-loop critical-rank work and source→destination traffic. Test multiple starts/seeds. Include a simple, well-defined global-frequency/load-balanced static heuristic, and an exact or tighter bound if the expert pool is small enough. Fit only on calibration, freeze before held-out. Beware: minimizing aggregate rank assignments is **not** the same as minimizing the sum of per-loop maxima.

**S2 Per-loop zero-migration oracle:** for each loop position, allow an independently optimal mapping under the same per-rank capacity. This assumes **free instant remapping of expert weights** and is therefore a **non-executable optimistic upper bound**. It must not be reported as speedup.

**S3 Optional feasible alternative:** an offline constrained schedule that either (i) uses a fixed placement for all passes, (ii) changes only metadata when expert weights already have legitimately available replicas, or (iii) includes **actual** per-loop copy/replication cost, synchronization, memory headroom and ROI. Do not assume future expert demand is known at run time.

### Cost objective and accounting

For a physical or calibrated replay stage, use the **loop-wise critical path**, not an average of rank loads:

```text
T_static = sum_over_actual_loop_invocations T_EP_critical(loop, S1)
T_oracle_lower = sum_over_actual_loop_invocations min_legal_placement T_EP_critical(loop, placement)
Optimistic_EP_headroom = (T_static - T_oracle_lower) / T_static
```

- This bound is defined only when the cost model is validated or the placements are directly measured; otherwise call it **modeled structural headroom**, not timing headroom.
- Include Dispatch + owner Expert + Combine, topology, real source rank, shared expert/non-EP overhead where applicable. Consider per-layer overlap; a sum of rank maxima is not automatically the actual end-to-end critical path.
- Never generate a per-loop oracle speedup by relabeling rank IDs while keeping expert weights stationary.
- The oracle's migration cost is zero *by definition*; separately report copied expert-weight bytes and measured/interpreted migration cost for any deployable strategy.
- If there are multiple distinct shared physical expert pools, their static placement variables are independent; never force one mapping across unrelated tensors.

**Required comparison on held-out:** `S0 vs S1 vs S2 (optimistic)`, under identical requests, same checkpoint and native H. Favor physically measured S0 and S1; S2 can be validated modeled bounds clearly segregated from actual latency.

---

## 9. Stage C — Physical EP4 causality and practical headroom

Only enter Stage C if Stage A/B reveal sizable conflict and Stage 0 physical EP4 is ready.

- Measure `S0` and `S1` (two fully static weight-preserving mappings) with the same checkpoint and true four-rank dispatch/owner GEMM/combine.
- Check token/logit parity, hardware topology, memory allocations, expert tensor ownership, and run-order stability.
- Profile per-loop event timing only with bounded observer overhead. Report output TPS, TTFT (if meaningful), TPOT/decode latency, E2E separately; no mixing cold/warm prompt regimes.
- Where feasible add a **nonlooped/control distribution** with the same hardware/mapping/expert shapes and equal token workload to test whether the observed conflict is truly loop-conditioned rather than generic static EP imbalance. This may be a trace replay; it is **not** a native-model quality comparison.
- Source-rank load, max GPU compute time and collective API time must be tracked separately. In particular, with NV18, remote bytes may not determine latency.

### Migration / replication ROI gate

For each hypothetical per-loop placement change:

```text
copy_bytes = sum(bytes of experts whose physical ownership changes)
T_move = measured synchronizing/reconfiguration cost (or conservative calibrated lower bound)
remaining_savings = sum(projected EP-stage savings over the next compatible invocations)
ROI = remaining_savings / T_move
```

At B1, if independent loop passes are short and large weights must move every pass, an oracle may be physically unattainable. If ROI <1, no dynamic-migration method GO, even if optimistic placement oracle is strong.

---

## 10. Suggested pre-registered kill gates

These are **heuristic discovery thresholds**, not promises of acceleration.

| Gate | Conditional GO signal | No-Go / limitation |
|---|---|---|
| G0 Authentic and feasible | Real pretrained Looped MoE, native H, correct TRUE EP4 on 4/5/6/7 | `CHECKPOINT_BLOCKED` / `RUNTIME_BLOCKED` |
| G1 Loop-distinct work | Stable request-level loop-conditioned demand differences and per-loop rank criticality | Routing shifts only, no physical effect |
| G2 Best-static conflict | Held-out S1→S2 **modeled EP-stage optimistic headroom ≥20%**; repeatable, CIs not compatible with trivial gap | <10% best-static→oracle gap ⇒ `NO-GO` |
| G3 Physical materiality | EP critical work **≥25% of clean decoding wall** OR ≥10% projected E2E ceiling; physical evidence, not just bytes | tiny critical-path share; early `NO-GO` |
| G4 Practical opportunity | A concrete bounded path with **≥10% credible net E2E potential** after coordination/movement, not merely free oracle | ROI <1 / speedup depends on moving massive weights per loop ⇒ `STRUCTURAL_ONLY` or `NO-GO` |
| G5 Generality | Signal on meaningful held-out workload; ideally another valid Looped checkpoint/loop structure | tiny toy-only artifact ⇒ `SMALL_MODEL_ONLY` |

**Important interpretation:** A physically profiled 0.2–0.6B model across four H100s cannot by itself pass G5. Do not give `GO` merely because its EP fraction is large due to fixed communication overhead. If only small authentic weights are available, deliver a useful **CONDITIONAL / SMALL_MODEL_ONLY** structural characterization and close the *performance claim* rather than engineering a fake large model.

Default terminal verdicts:
- `NO_ROUTING_DIVERGENCE`: no loop-conditioned routing difference in tested authentic checkpoint;
- `NO_LOOP_EP_SIGNAL`: divergence without material physical EP/rank penalty;
- `NO-GO`: best static removes conflict; oracle <10% headroom; or practical net opportunity absent;
- `STRUCTURAL_ONLY`: modeled structural gap, no valid physical latency;
- `SMALL_MODEL_ONLY`: results exist but hardware/scale is unrepresentative;
- `RUNTIME_BLOCKED` / `CHECKPOINT_BLOCKED`;
- `CONDITIONAL`: promising measured signal but model-scale/causality/practical feasibility unresolved;
- `GO`: multiple credible physical gates pass with a practical non-oracle route to ≥10% E2E.

---

## 11. Statistics and plotting

Sampling unit = **independent request** (or prompt/continuation episode). At least 2,000 request-level paired bootstrap resamples for material comparisons when feasible; do not pretend that hundreds of loop invocations constitute independent confidence samples. Preserve `N` for each comparison. Report variation across source-rank placement, prompt length, decoding phase and batch size explicitly. Do not omit failed runs or cherry-pick hot loops.

Required figures:
1. **Loop index × expert ID** demand heatmap, explicitly for one shared physical pool.
2. **Loop index × EP rank** load heatmap; mark changing hot rank and loop sample counts.
3. Per-loop **dispatch / expert / combine** actual durations vs max/mean rank load, and full-request critical share.
4. Held-out **native static S0 vs global best-static S1 vs per-loop oracle S2**: modeled lower bound vs physically measured timing clearly distinguished.
5. Ratio `oracle gap / static physical EP cost`, and projected E2E ceiling, with uncertainty.
6. **Movement cost/ROI** vs loop reuse count; negative results visibly displayed.
7. Optional memory/weight footprint demonstrating small-model representativeness caveat.

Tables must always label **logical bytes vs measured wire bytes**, actual timing vs model projection, and include loop architecture, source rank, model, N, and inference regime.

---

## 12. Minimal reproducible deliverables

```text
poc_looped_moe_ep/
├── model_runtime_audit.md
├── prior_overlap.md
├── reproduce.md
├── report.md
├── scripts/
│   ├── audit_checkpoint.py
│   ├── verify_loop_semantics.py
│   ├── validate_true_ep4.py
│   ├── trace_routing.py
│   ├── analyze_loop_conflict.py
│   ├── optimize_best_static.py
│   ├── estimate_oracle_bound.py
│   ├── run_physical_static.py
│   ├── analyze_migration_roi.py
│   └── make_plots.py
├── results/
│   ├── environment.json
│   ├── model_manifest.json
│   ├── frozen_requests.json
│   ├── per_loop_routing.csv.gz
│   ├── per_loop_rank_metrics.csv.gz
│   ├── placement_comparison.json
│   ├── physical_timing.json
│   ├── validity.json
│   └── verdict.json
└── figures/
    ├── expert_routing_by_loop.png
    ├── rank_load_by_loop.png
    ├── physical_loop_cost.png
    ├── best_static_vs_oracle.png
    └── movement_roi.png
```

Files may be omitted **only** if a documented earlier gate blocks them. Avoid fabricated empty measurements; write `NOT_RUN`, `UNAVAILABLE` or `UNVALIDATED`.

**Git:** recommended branch `flashvep/looped-moe-ep-poc`. Do not `git reset --hard`, `git clean`, or touch other researchers' files/processes. Commit/push only PoC-owned files after validation; record commit SHA and link to `report.md`.

---

## 13. Final report required answers

1. Exact authentic pretrained checkpoint, repository revision, sharing topology and native loop count?
2. Are router / attention / expert tensors shared independently? Which physical pool repeats?
3. TRUE EP4 on 4/5/6/7: owner-map proof, actual D/C, reference parity?
4. Is this model realistically EP-relevant on H100 ×4, or an artificial tiny-model setup?
5. Are per-loop routing distributions measurably different beyond sample noise?
6. Do those differences induce measurable physical per-loop rank stragglers and EP-critical work?
7. Which rank is hot per loop, and is graph density already saturated?
8. Can S1 global best-static remove most problems that S0 native placement has?
9. What *held-out* S1→S2 zero-copy oracle gap remains, in which cost domain?
10. What is the clean physical E2E / decode latency attributable to the sparse EP path?
11. Does a nonlooped/equal-work control show the same generic EP behavior?
12. Could any practical method achieve meaningful net gain after expert movement and coordination costs?
13. Were all correctness, benchmark and observer-overhead checks satisfied?
14. Is the result robust on an independent prompt set, and on a representative checkpoint scale?
15. What is the honest final verdict and the next smallest experiment if and only if justified?

### Hard conclusion discipline

**Do not** turn routing changes, hot-rank heatmaps, free-migration oracles, or increased logical bytes into an invented EP speedup. The decisive objective is a **real, large, loop-specific critical-path loss that survives best-static placement and admits a feasible remedy**. If this is absent, close Looped MoE × EP rather than inventing another placement/replication trick.
