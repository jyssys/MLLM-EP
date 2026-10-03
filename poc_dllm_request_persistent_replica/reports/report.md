# Request-Persistent / Cross-Block Expert Residency — NO-GO

Stage A completed; Stage B NOT implemented. This is an offline placement characterization grounded in current physical EP4 traces, not a measured replica speedup.

## Contract and physical validation

Source of truth: [authoritative specification](../references/authoritative_spec.md). Parent evidence commit `90fcaa039d609e5c3f2c8e563448c54874673597`. Surgical-changes and simplicity-first skills kept all work inside this directory, reused validated instrumentation/accounting, and avoided runtime/predictor expansion after gate failure.

- Model `inclusionAI/LLaDA2.0-mini`, revision `dad945cac317da394b390f82c7b40691d8a881ed`; BF16, hidden2048,256routed experts,K8,19sparse layers1–19 (layer0 dense), shared expert preserved, routed scaling2.5.
- Actual topology **dense TP4 / source-SP4 / physical EP4 / DP1 / PP1**, NOT Qwen TP2-DP2. Source partition comes from recorded physical route metadata, not token-index EP projection.
- Physical `2,3,6,7` → logical `0,1,2,3`. H10080GB, NV18 fabric. Runtime versions/UUIDs/topology and exact loaded weight shapes are in [environment](../results/environment.json) and lossless raw ownership evidence.
- Runtime: existing dInfer upstream `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` **plus its previously validated bridge modifications**, PyTorch2.8.0+cu128, SGLang0.5.3.post1, DeepEP1.2.1+73b6ea4, Transformers4.57.0. Loaded source hashes are pinned in the inherited environment; no external file is edited by this PoC.
- Each layer/expert costs **6MiB** from loaded `w13` and `w2` BF16 tensors.64experts/rank. Memory budgets0.5/1/2/4GiB/source span all19layers jointly. Expert17 at layer3 and layer10 are distinct allocations.
- Tiny same-prompt bypass/trace tokens and NFE match across4ranks. All32long requests match tokens/NFE between bypass,timing,trace; acceptance is cross-rank equal, with two extra acceptance-only probes. **109,003** actual receive-row/path ownership checks pass. No silent EP fallback.

## Cohort and unchanged decoding

32fresh requests:24coding(6calibration/18held-out),8reasoning(2cal/6held-out). Source task IDs are seed9017-frozen public HumanEval/GSM8K tasks disjoint from prior communication prompts. Coding asks for a self-contained module with tests, not a benchmark pass@1 claim. Primary property was long-generation B1; no request was selected/dropped after routing, length or output inspection.

Native block32,threshold0.9,T0,EOS enabled,prefix cache,maximum_unroll1,expected_tpf15,early_stop unchanged. `gen_length=512` is a native block-aligned budget: observed outputs **298–543**, not an exact512-token truncation. Held-out mean output 505.1, mean blocks 16.38; every request retained. Existing32short B32 traces reused without GPU execution. Three warmups+32×3lanes+2acceptance probes =101 model runs; no unbounded retry.

Current long held-out live-only residual **5.6763×**, coding **4.9459×**, reasoning **7.8676×**. These replace, rather than assume, previous short5.773×.

## Traffic, lifetime and causal rules

All placement gains are above **perfect already-accepted/context traffic removal**. A remote destination contributes one4096-byte hidden row plus one4096-byte combine row per live source token. Multiple selected experts on one destination still require one row; a destination closes only if every selected expert there is resident. K8 paths all execute freshly; no hidden/output reuse,expert dropping,route/probability/scaling change.

Global Warm is calibrated only on8requests, starts resident, and pays zero per-request copy. Both plain global frequency and stronger global destination-bundle controls are reported. Current request policies profile one/two **complete** blocks, pay those blocks' full live-only traffic, then copy once and retain until request end. No per-block reload, bounded incremental update omitted, no learned model. First-block/first2 frequency ablations are included.

Load caps+10/15/20% apply to **observed history only** for causal policy and calibration only for Global; no test-future pruning. Future actual worst layer/iteration load is then measured independently. Oracle may use request truth. A cap label therefore is NOT a guarantee about future load. Native source-SP placement can change across blocks; history-only safety does not silently become future-truth safety.

The inherited eight-offer destination bundle heuristic is reused with an exact cached-layer implementation, verified against the original on random and rejection cases. Oracle is an achievable future-truth reference, not an exact integer optimum. Global can floor an oracle reference ONLY when it satisfies that oracle's actual future-load cap. P5/P6 are not deployable methods.

## Calibration freeze and primary held-out result

Before held-out policy summaries, [operating point](../results/frozen_operating_point.json) freezes **first2_blocks, 2GiB/rank, observed-history +10% cap**, stronger calibration-selected Global=`global_destination`. This is a descriptive point, not a valid deployable candidate. Selecting the best-looking held-out point is prohibited. Statistics use2000 request-bootstrap replicates; iterations,layers,blocks are not independent samples.

Pooled held-out24. **The following byte gains/recoveries are apparent, NOT matched-actual-load gains. Every selected request violates the preferred future load cap; no positive novelty/runtime promotion is inferred.**

- Request-vs-Global additional D+C byte reduction: **11.2231 [9.9438, 12.5785]%** (95%CI).
- Aggregate byte-ratio additional reduction: **11.0428 [9.9303, 12.1913]%**. The first statistic equally weights request-relative gains; this one divides summed gains by summed Global bytes.
- Global residual **5.3329 [4.6064, 6.1293]×**; causal residual **4.7381 [4.0915, 5.4448]×**.
- Persistent oracle recovery **beyond Global**: **87.5189 [75.5810, 98.6630]%**. Negative recovery means the causal plan is worse than Global.
- Coding-primary18 additional reduction vs Global: **11.6581 [10.0316, 13.4464]%**.
- Positive future-block mean advantage count: **15**, first4-future-block sustained gate=True.

Same budget / same observed guard, actual load explicitly exposed:

| Policy | residual / one-shot | reduction vs live-only | actual peak extra GiB/rank | mean / absolute-worst layer-iteration load increase |
|---|---:|---:|---:|---:|
| global_frequency | 5.4296× | 4.371% | 1.998 | 84.26% / 150.00% |
| global_destination | 5.3329× | 6.187% | 1.998 | 81.51% / 200.00% |
| first_block | 5.1071× | 10.633% | 1.417 | 125.84% / 200.00% |
| first2_blocks | 4.7381× | 16.713% | 1.996 | 146.55% / 200.00% |
| first_frequency | 5.1151× | 10.336% | 1.459 | 131.49% / 200.00% |
| first2_frequency | 4.8094× | 15.424% | 1.998 | 156.05% / 200.00% |
| oracle_load | 4.6504× | 18.752% | 1.998 | 10.00% / 10.00% |

Unconstrained4GiB future oracle residual **1.2200 [1.0087, 1.4347]×** is an upper-headroom reference, not a recommendation: inspect its hotspot/copy columns. All budget/guard curves and request rows are in `results/simulation_long_v2/`; the short continuity run is separate.

## Cross-block persistence versus Global

Expert cosine may be high because of global popularity. Future coverage and whole-destination closure against Global are decisive. A top-M diagnostic uses M remote experts **per layer/source**, not the production whole-rank memory budget.

| distance from block0 | request expert cosine | Global cosine | coverage@16 advantage (pp,95%CI) | destination-closure@16 advantage (pp,95%CI) |
|---:|---:|---:|---:|---:|
| 1 | 0.3779 | 0.4508 | -3.3174 [-8.4151, 1.9954] | -3.4372 [-6.8467, 0.1586] |
| 2 | 0.3839 | 0.4621 | -8.3726 [-13.2513, -3.7730] | -7.8005 [-11.7008, -4.1494] |
| 4 | 0.3352 | 0.5218 | -18.0873 [-22.1003, -14.0420] | -15.5090 [-19.2718, -12.1806] |
| 8 | 0.3412 | 0.5620 | -23.2542 [-28.2160, -18.2691] | -21.8936 [-26.5208, -17.5670] |
| 12 | 0.3334 | 0.5394 | -20.1361 [-24.7000, -15.4663] | -18.5801 [-22.2116, -14.9067] |
| 16 | 0.2733 | 0.5269 | -22.7312 [-27.7341, -17.4638] | -20.3007 [-24.8539, -15.7409] |

Full cosine,JSD,top-M overlap/coverage,rank cosine/JSD,destination closure for M4/8/16/32, adjacent and distance2/4/8, first2 profile, and fixed same-domain calibration other-request control are in [persistence CSV](../results/persistence_v2/block_locality.csv). Request-clustered paired CI is separate. `persistence/` retains the preliminary output; `persistence_v2/` fixes tiny negative-roundoff JSD square roots without changing hypotheses/features/selection.

## Copy economics and hotspot

Expert-sized and384MiB P2P measurements on the same12directed GPU edges are reused with hashes. Selected held-out causal mean copies **6.136GiB/request**, serial expert-copy estimate **54.874ms**, endpoint-parallel estimate **19.148ms** (not a measured concurrent transfer). Global warm per-request preparation=0.

Byte ROI **0.0267 [0.0227, 0.0309]×**. Observed byte break-even **0/24** requests; median extrapolated break-even / remaining refinement horizon **38.756×**. Byte break-even is NOT latency break-even.

**Time ROI and temporal break-even are UNRESOLVED:** replica D+C was not executed. Saved bytes are not converted into stage speedup. Existing timing gives only total baseline D+C/copy as an optimistic ceiling, not avoided communication time. Thus low byte ROI is a structural concern, not a proof that time ROI must be equally low.

Causal request-cumulative max-rank increase **14.5475 [11.0709, 17.9054]%**, mean request worst layer/iteration **146.5468 [135.9479, 157.7823]%**, **absolute worst 200.00%**. Both cumulative and per-stage counters are retained. Matched future-load validity=False; an apparent win with uncontrolled hotspot is not acceptable.

Worst ratios can occur when only a few live paths remain. Absolute per-rank counts are in [absolute_hotspot_counts.csv](../results/absolute_hotspot_counts.csv); summed per-stage critical load counts actually decrease **4.55%**. These counters fail the predeclared load gate but do not prove a measured latency regression. No runtime result is fabricated in either direction.

## Physical timing and correctness scope

Current baseline D+C / request wall held-out mean **13.163%**. Timing uses one cumulative-critical rank and all components from that rank, no mixing component maxima. Mean timing observer overhead all32 **7.920%**; heavy trace is not headline timing. This does not measure replica latency.

No Stage B replica tensors,route redirection,slot eviction,production scheduler,CUDA kernel,quantization or EPLB were implemented. Therefore replica final-token/acceptance/router/NFE parity, GSM8K EM/HumanEval pass@1, and measured D+C/E2E gains are **NOT RUN**, not zero or assumed parity. Baseline physical parity passes; offline accounting preserves unchanged path count and layer identity, which is not runtime correctness.

## Gates and direct answers

`summary_v3.json` is the authoritative audited summary. v2 enforces matched-actual-load prerequisites; v3 excludes future-load-unsafe Global sets from constrained-oracle feasible floors. The selected primary point, its CI and verdict are unchanged. Plans, features, thresholds and operating point never change; earlier summaries are preserved. Corrected complete CI is `bootstrap_ci_v3.json`.

| Gate | result |
|---|---|
| apparent_byte_increment | True |
| request_beats_global | False |
| sustained_four_future_blocks | True |
| copy_roi_validated | False |
| timing_roi | UNRESOLVED_NO_REPLICA_TIMING |
| early_break_even | False |
| actual_worst_load_le10 | False |
| matched_actual_load | False |
| recovery_beyond_global_ge50 | False |

1–3. Cross-block locality and decay are quantified above; request-vs-global advantage, not raw cosine, governs novelty.
4–6. Matched-budget additional reduction, horizon and calibration-selected budget are above; no feasible passing memory/load point is asserted.
7–9. Exact one-request copy byte ROI is measured structurally; copy milliseconds use measured P2P. Avoided D+C timing ROI/temporal break-even remain unresolved, byte break-even is shown separately.
10. Worst-stage hotspot is explicitly exposed and can invalidate a history-capped plan.
11–12. Global-after-oracle headroom and causal recovery are in paired CI; recovery denominator is Global minus persistent oracle, never Vanilla minus oracle.
13. Long generation supplies roughly16–17blocks instead of2–3. This alone does not establish request-specific advantage or amortization.
14. Optional AR control not run; no mandatory gate depends on it.
15–18. Stage B not promoted unless all Stage A premises pass; runtime parity/accuracy and measured gains cannot be claimed from offline accounting.
19. Final verdict **NO-GO**.
20. Close dLLM-specific expert-weight replication/movement family=True. Scope is this single-request BF16 source-SP4 regime and the tested causal lifetime policies, not all possible large-batch/cross-request generic replication.

## Next action

Close tested dLLM-specific expert-weight replication/movement family in this B1 BF16 source-SP4 regime: persistent byte advantage exists only with uncontrolled future hotspots and copy amortization is unvalidated. Investigate activation-side repeated live-token communication; do not build persistent runtime.

No adaptive or learned rescue. If a new direction is authorized, the smallest probe is a frozen-route activation-side experiment measuring what repeated live-token traffic can change without modifying denoising semantics, with the optimistic dead-token-elimination reference kept separate.

## Reproduction and evidence

See [README](../README.md) and [reproduce.sh](../scripts/reproduce.sh). Lossless content-deduplicated complete raw traces are committed under `raw/compact`; original1.1GiB capture remains untouched locally under `raw/long_v1`. Every named NPZ array and original JSON byte is round-trip validated; ZIP-container byte identity is not claimed. Planner-only ID duplicates rebuild from raw and are not redundantly committed. Original SHA and compact SHA are indexed.

Required audited figures in `figures_v2/`: predictor advantage/distance, residual/budget, gain/global/future block, copy ROI/break-even, communication/load Pareto, layer/source/destination allocation, persistent oracle recovery. Preliminary figures are retained in `figures/`. No runtime timing plot is fabricated.

## Final status

BASELINE_STATUS: PASS_TRUE_EP4
REQUEST_COHORT: fresh8calibration/24heldout,32requests,B1,nativeblock32,genbudget512
REQUEST_VS_GLOBAL_GAIN: apparent11.2231 [9.9438, 12.5785]%; NOT_LOAD_MATCHED
PERSISTENCE_HORIZON: positive mean at15future blocks; sustained4gate=True
BEST_RESIDUAL: selected causal4.7381×,Global5.3329×,unconstrained4GiBoracle1.2200×
BEST_BUDGET: descriptive2GiB/rank; no passing deployed budget
ORACLE_RECOVERY_BEYOND_GLOBAL: apparent87.5189 [75.5810, 98.6630]%; NOT_LOAD_MATCHED
COPY_BYTE_ROI: 0.0267×
TIME_ROI: UNRESOLVED
BREAK_EVEN: observed byte0/24; temporalUNRESOLVED
LOAD_IMBALANCE: mean worst146.55%,absolute worst200.00%
STAGE_B: NOT_IMPLEMENTED_STAGE_A_NO_GO
CORRECTNESS_ACCURACY: baselinePASS; replica runtimeNOT_RUN
MEASURED_D_C_E2E_GAIN: NOT_MEASURED
VERDICT: NO-GO
NEXT_ACTION: Close tested dLLM-specific expert-weight replication/movement family in this B1 BF16 source-SP4 regime: persistent byte advantage exists only with uncontrolled future hotspots and copy amortization is unvalidated. Investigate activation-side repeated live-token communication; do not build persistent runtime.
