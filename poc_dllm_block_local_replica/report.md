# Layer-Aware Block-Local Expert Replication: Stage-A NO-GO

The structural oracle has large destination-closure headroom, but simple early-block causal plans recover too little of it and cold expert copies have very poor byte ROI. **Stage B was not promoted. No replica runtime, speedup, or benchmark-accuracy parity is claimed.**

## Contract and physical baseline

- Source of truth: `references/authoritative_spec.md`, SHA256 `feba9b2fbc87a6f503bf76ad22f5ea6d271de71abf567afd18a36254ba0b3cbd`.
- Model: inclusionAI/LLaDA2.0-mini, revision `dad945cac317da394b390f82c7b40691d8a881ed`, BF16, 19 sparse layers, 256 routed experts/layer, K8, hidden2048, intermediate512, routed scaling2.5. No model/router/precision changes.
- Actual topology: denseTP4 / source-SP4 / physicalEP4 / DP1 / PP1. This is the validated dLLM topology, not Qwen TP2-DP2. Linear64 experts/rank.
- Physical H10080GB GPUs2,3,6,7 -> logical0,1,2,3, NV18 links. UUIDs/topology/process/version audit: `results/environment.json`, `raw/physical_revalidation/p2p/clean_p2p_start.json`.
- Reused source commit `dc539d3222193ab0902e0a342679edc3610f54bb`. Exact loaded w13 `[64,1024,2048]` BF16 and w2 `[64,2048,512]` BF16: **6,291,456 bytes =6MiB per (layer,expert)**. All19 layers counted in the shared per-source budget; no multiplication of the budget per layer.
- Fresh physical revalidation: 2 bounded probes, same token IDs/acceptance/NFE/router IDs/weights and received owner-local expert paths as prior capture. TRUE DeepEP sparse dispatch/combine is directly checked. Current/prior routing arrays are compared, not assumed equal.
- GPU session uses scoped burner stop, competing-process check, watchdog and finally restoration. No other GPU process was modified.

## Cohort and causal validity

Prior frozen32 problems (16 HumanEval,16 GSM8K) are reused, plus8 block64 robustness captures. New replica calibration split:4/domain, held-out12/domain. Split seed8611, fixed before replica outcome evaluation. These are a newly held-out **replica-policy split**, not fresh unseen generations: prior amplification statistics were already known.

HumanEval is the declared communication-exposed primary regime; GSM8K is secondary, with balanced pooled continuity also reported. Block32 native threshold0.9, temperature0, generation budget64, prefix block-diffusion cache, EOS enabled are unchanged. No favorable-output selection and no new full-model capture beyond physical sanity checks.

Only masked output positions **before each actual forward** enter `B_live_only`. Context and already-accepted output positions are excluded. Native cross-block source partitions are preserved; source rank is not re-derived from token ID. Each request's original `B_one_shot` stays frozen.

Primary no-replica continuity (all32):5.773x. This24-problem held-out baseline:5.426x [4.363, 6.520]. Different cohorts must not be silently mixed. The original one-shot1x denominator does not itself have replicas; localized communication may fall below1x without violating accounting.

Global/static uses normalized calibration request popularity only and can be ready at block start. First-iter/first2 plans use only ordinal0 / ordinals0–1 respectively. Their profiling iteration traffic is charged unchanged; plan applies only afterward. Future-truth policies alone inspect future routes. Tests perturb future routing and check causal plans do not change. Static is an intentionally strong control; its cold-copy charge is also reported, while persistent static replicas could amortize better and are outside this block-local contract.

## Policy result (24 held-out problems, request-macro means)

|Policy|GiB/rank|Residual x|D+C byte reduction|Oracle recovery|Byte ROI|Max-rank cumulative load increase|
|---|---:|---:|---:|---:|---:|---:|
|global|0.5|4.872|9.64%|25.91%|0.00244|9.99%|
|first_frequency|0.5|4.821|10.31%|27.70%|0.00499|16.27%|
|first_destination|0.5|4.677|12.74%|34.15%|0.00619|11.16%|
|first2_destination|0.5|4.501|14.05%|37.94%|0.00521|5.40%|
|oracle|0.5|3.425|37.42%|100.00%|0.01078|19.72%|
|oracle_load|0.5|3.915|29.15%|77.69%|0.00820|13.84%|
|global|1|4.530|16.15%|28.88%|0.00197|20.55%|
|first_frequency|1|4.337|18.44%|33.00%|0.00453|29.09%|
|first_destination|1|4.276|19.52%|34.92%|0.00479|23.47%|
|first2_destination|1|3.996|21.77%|39.18%|0.00407|12.87%|
|oracle|1|2.421|56.08%|100.00%|0.00818|34.89%|
|oracle_load|1|3.371|39.98%|71.07%|0.00570|19.05%|
|global|2|3.997|26.19%|32.83%|0.00157|33.83%|
|first_frequency|2|3.764|27.65%|34.75%|0.00380|38.28%|
|first_destination|2|3.752|27.84%|34.98%|0.00383|38.82%|
|first2_destination|2|3.372|31.34%|39.46%|0.00314|25.17%|
|oracle|2|1.127|79.79%|100.00%|0.00599|55.40%|
|oracle_load|2|2.893|49.67%|62.08%|0.00409|21.21%|
|global|4|2.961|45.87%|46.53%|0.00136|49.93%|
|first_frequency|4|3.627|29.80%|30.28%|0.00351|37.49%|
|first_destination|4|3.627|29.80%|30.28%|0.00351|37.49%|
|first2_destination|4|3.177|34.30%|34.92%|0.00285|24.87%|
|oracle|4|0.100|98.57%|100.00%|0.00431|66.72%|
|oracle_load|4|2.812|51.10%|51.78%|0.00379|20.91%|

All mean/median/95% request-bootstrap CIs are in `results/bootstrap_ci.json` (2000 resamples). No layer or iteration is treated as an independent serving sample.

The future-truth oracle is a destination-bundle greedy **achievable reference**, not an exact combinatorial optimum. A fractional LP relaxation supplies an upper bound at0.5/4GiB (stored per request); no fractional plan is executed or counted as physical replication. Recovery relative to a stronger exact oracle could only be lower.

For the held-out pool the achievable greedy reduction versus LP upper bound is37.420% vs38.702% at0.5GiB, and98.573% vs98.691% at4GiB. Thus the large-budget oracle headroom is tightly bracketed even though integer optimality is not claimed. The maximum mean recovery across **every** tested first1/2-iteration causal budget is39.464%, still below50%.

Primary HumanEval held-out (12 problems) at the calibration-selected4GiB point: live-only3.591x -> future-truth oracle0.058x -> causal2.404x. Causal reduction23.434%, oracle recovery23.792%; global/static residual1.893x. The balanced pooled numbers should not be substituted for coding-primary results.

## Direct answers to the17 research questions

1. **Oracle reduction:** at4GiB/rank, 98.573% [98.141, 98.983] of live-only D+C bytes; large structural opportunity exists.
2. **Oracle residual:** 0.100x [0.065, 0.138]; load-capped future-truth residual 2.812x [2.165, 3.477].
3. **Memory:** swept0.5/1/2/4GiB/rank. Calibration-selected descriptive causal point `first2_destination@4` has mean peak extra memory3.382GiB/rank; no point is declared a runtime operating point. Burner-free audit and loaded baseline size leave these budgets feasible in80GB HBM.
4. **Layer dependence:** replicas are selected independently by(layer,source,expert) under one joint capacity. `figures/layer_replica_allocation.png` and `results/layer_allocation_matrix.csv` show19 separate sparse-layer profiles with source/owner identities. Expert17 at two layers is charged twice.
5. **Destination vs frequency:** at0.5GiB, first-iteration destination-aware adds 2.435pp [1.953, 2.932] reduction over raw frequency. At4GiB first-iteration choices saturate and are identical. Positive low-budget structural improvement is not sufficient for runtime promotion.
6. **Causal vs static:** selected `first2_destination@4` adds -11.566pp [-18.483, -4.767] reduction versus global; at large budget static wins. Smaller-budget causal advantages exist but do not satisfy all gates.
7. **Recovered oracle:** selected causal 34.919% [28.094, 41.402]; all first/first2 causal points in pooled held-out remain below50%. HumanEval at selected point: 23.792% [15.132, 32.777].
8. **Copy amortization:** selected causal byte ROI 0.003x [0.002, 0.004] versus2x gate. Median block byte break-even estimate 3350.7 future iterations at the observed avoidance rate, versus median7 remaining iterations. This is a byte-volume extrapolation, not a measured latency break-even. An optimistic policy-independent bound over all40 captures is1.029x, still below2x. P2P expert-sized median across edges:0.0504ms;384MiB bulk median BW:389.6GB/s.
9. **Compute hotspot:** selected causal cumulative max-rank assignment increase 24.867% [20.719, 28.947]; worst iteration/layer maximum increase 186.455% [163.501, 208.414]. Sum-of-stage maxima changes only 1.155% [-1.237, 3.390]; this aggregate can hide large instantaneous hotspots and is not a measured expert latency model.
10. **Pareto:** load-capped oracle at4GiB cuts 51.099% [47.520, 54.827] while each iteration/layer max is capped at+10%. It still has cold-copy cost and needs future truth. At2GiB it nearly saturates (49.67% reduction). No causal point passes the full recovery/copy/load chain. See `figures/comm_vs_load_pareto.png`.
11. **Stage B:** NOT_IMPLEMENTED, as required by failed Stage-A gates. No dynamic placement runtime or learned predictor was added.
12. **Measured D+C latency gain:** NOT_MEASURED for replicas; offline byte reduction is not speedup.
13. **Measured E2E latency gain:** NOT_MEASURED for replicas. P2P times do not establish an equivalent measured dispatch/combine timing argument.
14. **Correctness:** physical baseline parity PASS on2 probes; structural accounting preserves fresh K8 work and weights conceptually. Replica token/acceptance/router/NFE/tensor/combine parity NOT_TESTED because no runtime exists. Do not call offline invariance measured runtime correctness.
15. **Accuracy:** GSM8K exact match/HumanEval pass@1 replica comparisons NOT_RUN. Bounded prior generations are not full benchmark completions; no fabricated zero-delta score.
16. **Verdict:** **NO-GO** for simple first1/2-iteration, cold block-local exactBF16 replicas in this low-request source-SP4 regime. Oracle headroom passes, but causal recovery fails, chosen large-budget causal policy loses to static, byte ROI fails, and substantial instantaneous compute hotspots appear. Negative conclusion does not rule out persistent cross-block warm replicas, much larger batch or another EP topology.
17. **Smallest next experiment:** before any runtime, measure cross-block/request persistence under one small persistent replica cache and its cumulative destination-closure byte/copy ROI. Only if that amortizes across much longer lifetimes should this exact-placement direction be revisited.

## Copy-cost estimates and limitations

Gate-audit clarification: `results/summary_v2.json` is the authoritative final verdict artifact. The first summary checked the summed-stage-max proxy for the preferred compute gate; the audit corrected it to require both cumulative max-rank and worst-stage maxima within+10%. That gate is **FAIL**, not PASS. All policy numbers, CIs, calibration-selected point and NO-GO verdict are unchanged. The first calculation is retained append-only for transparency.

Every selected(layer,expert,source) incurs one full6MiB copy each block. Selected causal request estimates: serial two-tensor copy165.806ms [143.092, 187.334]; endpoint-parallel estimate68.943ms [59.604, 77.622]; optimistic bulk estimates serial51.190ms [44.175, 57.845], parallel21.052ms [18.168, 23.730]. Parallel/bulk costs are estimates from measured single-edge P2P, not measured concurrent scattered-expert copies.

**Byte ROI is not timing ROI.** DeepEP has fixed/protocol costs and P2P copy bandwidth differs from DeepEP hidden-row traffic. Actual avoided D+C time, time ROI and runtime break-even are UNAVAILABLE. Do not conclude the measured timing can never amortize merely from byte ROI. Stage A still fails independently: early causal oracle recovery is below50% at every budget, especially in the coding primary regime. No byte-to-ms speedup conversion is used to rescue that failed causal premise.

Baseline primary continuity D+C/request wall17.974%, D+C/denoising span19.255% remain **measured prior physical facts**. Structural dead-token elimination5.773x is **not measured Epoch**; subtracting those dead-token bytes is not this PoC's contribution.

## Main table (matched24-problem holdout; structural rows not runtime)

|Method|Quality|Norm. EP traffic|D+C ms|E2E ms|Speedup|Extra HBM/rank|
|---|---|---:|---:|---:|---:|---:|
|Vanilla physical baseline|bounded baseline; no full task score|13.232x|307.29|1984.73|1.00x|0|
|Perfect stable/dead-token elimination structural reference|n/a|5.426x|n/a|n/a|n/a|0|
|Global/static structural replicas (4GiB)|n/a; runtime not run|2.961x|n/a|n/a|n/a|<= 4GiB|
|Block-local causal structural replicas (4GiB)|n/a; runtime not run|3.177x|n/a|n/a|n/a|<= 4GiB|
|Oracle block-local structural replicas|n/a|0.100x|n/a|n/a|n/a|<= 4GiB|
|Original acceptance-route one-shot reference|n/a|1.000x|n/a|n/a|n/a|n/a|

Vanilla stage time uses one cumulative-critical rank per request (all components from that same rank), event-only timing lane. Old validated observer overhead is8.284%mean; heavy router-capture wall is not headline timing. No structural row reuses Vanilla timings as if it were measured candidate timing.

## Reproduction and artifacts

Run `bash poc_dllm_block_local_replica/scripts/reproduce.sh` to create a fresh append-only output directory; raw source paths are pinned to the parent evidence commit. Required GPU work is only the short physical revalidation/P2P lane; all policy evaluation reuses prior rich trace. Tests: `python -m pytest -q poc_dllm_block_local_replica/tests`.

`results/design_freeze.json`, `results/reused_trace_manifest.json`, `data/live_routes/*.npz`, `results/simulation/*`, `results/copy_cost/*`, `results/bootstrap_ci.json`, `results/summary.json` carry reproducible compact evidence. No unrelated dirty/staged work was included.

## Final fields

BASELINE_STATUS: TRUE_PHYSICAL_EP4_REVALIDATED_PASS
LIVE_ONLY_CONTINUITY: 5.773x all32; 5.426x this24 heldout
ORACLE_RESIDUAL: 0.100x at4GiB/rank
CAUSAL_RESIDUAL: 3.177x (first2_destination@4)
ORACLE_RECOVERY: 34.92%
MEMORY_BUDGET: 4GiB/rank descriptive; NO qualifying runtime point
COPY_AMORTIZATION: BYTE_ROI 0.00285x; TIME_ROI UNAVAILABLE
COMPUTE_IMBALANCE: cumulative max-rank +24.87%; worst-stage +186.46%
STAGE_B: NOT_PROMOTED
MEASURED_D_C_GAIN: NOT_MEASURED
MEASURED_E2E_GAIN: NOT_MEASURED
REPLICA_CORRECTNESS_ACCURACY: NOT_RUN; physical baseline parity PASS
VERDICT: NO-GO
NEXT_ACTION: CLOSE_COLD_BLOCK_LOCAL_DIRECTION; optional offline cross-block lifetime test only
