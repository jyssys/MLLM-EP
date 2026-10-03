"""Evidence-backed report; measured physical baseline versus offline placement separated."""
import json
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_block_local_replica.metrics import bootstrap_mean


def interval(x,scale=1):
    return f"{x['mean']*scale:.4f} [{x['ci95'][0]*scale:.4f}, {x['ci95'][1]*scale:.4f}]"


def main():
    s=json.loads(summary_path().read_text());v=json.loads((ROOT/'results/baseline_validation.json').read_text())
    chosen=s['selected'];policy,budget,cap=chosen['policy'],chosen['budget_gib'],chosen['load_guard']
    target=s['selected_heldout'];contrast=s['selected_contrast'];control=chosen['global_control']
    g=s['stats']['long_heldout_all'][f'{control}@{budget:g}@{cap:g}']
    df=pd.read_csv(ROOT/'results/simulation_long_v2/policy_budget.csv');held=df[df.split=='heldout']
    lengths=pd.read_csv(ROOT/'results/long_requests.csv');h=lengths[lengths.split=='heldout']
    costs=pd.read_csv(ROOT/'results/copy_cost/request_copy_roi.csv')
    c=costs[(costs.lane=='long')&(costs.split=='heldout')&(costs.policy==policy)&(costs.budget_gib==budget)&(costs.load_guard==cap)]
    longlocal=pd.read_csv(ROOT/'results/persistence_v2/block_locality.csv')
    local=longlocal[(longlocal.lane=='long')&(longlocal.split=='heldout')]
    table=[];localstats=[]
    for distance in (1,2,4,8,12,16):
        p=local[(local.policy=='first_block')&(local.distance==distance)]
        a=p.merge(local[local.policy=='global'],on=['request','target_block'],suffixes=('_r','_g'),validate='one_to_one')
        if not len(a):continue
        adv=bootstrap_mean((a.coverage_at_16_r-a.coverage_at_16_g).tolist())
        closure=bootstrap_mean((a.destination_closure_at_16_r-a.destination_closure_at_16_g).tolist())
        localstats.append(dict(distance=distance,coverage_advantage=adv,closure_advantage=closure))
        table.append(f"| {distance} | {a.cosine_r.mean():.4f} | {a.cosine_g.mean():.4f} | {interval(adv,100)} | {interval(closure,100)} |")
    new_json(ROOT/'results/locality_headline.json',dict(unit='request',repeats=2000,first_block_vs_global=localstats))
    best_oracle=held[(held.policy=='oracle')&(held.load_guard==-1)&(held.budget_gib==4)]
    oracle_stats=bootstrap_mean(best_oracle.residual_amp.tolist())
    main=[]
    for p in ('global_frequency','global_destination','first_block','first2_blocks','first_frequency','first2_frequency','oracle_load'):
        r=held[(held.policy==p)&(held.load_guard==cap)&(held.budget_gib==budget)]
        main.append(f"| {p} | {r.residual_amp.mean():.4f}× | {100*r.dc_reduction.mean():.3f}% | {r.peak_memory_gib.mean():.3f} | {100*r.worst_stage_load_increase.mean():.2f}% / {100*r.worst_stage_load_increase.max():.2f}% |")
    directory=ROOT/'reports';directory.mkdir(exist_ok=True)
    text=f'''# Request-Persistent / Cross-Block Expert Residency — {s['verdict']}

Stage A completed; Stage B {'promoted' if s['stage_b_promoted'] else 'NOT implemented'}. This is an offline placement characterization grounded in current physical EP4 traces, not a measured replica speedup.

## Contract and physical validation

Source of truth: [authoritative specification](../references/authoritative_spec.md). Parent evidence commit `{PARENT}`. Surgical-changes and simplicity-first skills kept all work inside this directory, reused validated instrumentation/accounting, and avoided runtime/predictor expansion after gate failure.

- Model `inclusionAI/LLaDA2.0-mini`, revision `{REVISION}`; BF16, hidden2048,256routed experts,K8,19sparse layers1–19 (layer0 dense), shared expert preserved, routed scaling2.5.
- Actual topology **dense TP4 / source-SP4 / physical EP4 / DP1 / PP1**, NOT Qwen TP2-DP2. Source partition comes from recorded physical route metadata, not token-index EP projection.
- Physical `2,3,6,7` → logical `0,1,2,3`. H10080GB, NV18 fabric. Runtime versions/UUIDs/topology and exact loaded weight shapes are in [environment](../results/environment.json) and lossless raw ownership evidence.
- Runtime: existing dInfer upstream `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` plus previously validated bridge modifications, PyTorch2.8.0+cu128,SGLang0.5.3.post1,DeepEP1.2.1+73b6ea4,Transformers4.57.0. Loaded source hashes are pinned; this PoC edits no external file.
- Each layer/expert costs **6MiB** from loaded `w13` and `w2` BF16 tensors.64experts/rank. Memory budgets0.5/1/2/4GiB/source span all19layers jointly. Expert17 at layer3 and layer10 are distinct allocations.
- Tiny same-prompt bypass/trace tokens and NFE match across4ranks. All32long requests match tokens/NFE between bypass,timing,trace; acceptance is cross-rank equal, with two extra acceptance-only probes. **{v['actual_receive_buffer_destination_dedup_checks']:,}** actual receive-row/path ownership checks pass. No silent EP fallback.

## Cohort and unchanged decoding

32fresh requests:24coding(6calibration/18held-out),8reasoning(2cal/6held-out). Source task IDs are seed9017-frozen public HumanEval/GSM8K tasks disjoint from prior communication prompts. Coding asks for a self-contained module with tests, not a benchmark pass@1 claim. Primary property was long-generation B1; no request was selected/dropped after routing, length or output inspection.

Native block32,threshold0.9,T0,EOS enabled,prefix cache,maximum_unroll1,expected_tpf15,early_stop unchanged. `gen_length=512` is a native block-aligned budget: observed outputs **{int(lengths.output_tokens.min())}–{int(lengths.output_tokens.max())}**, not an exact512-token truncation. Held-out mean output {h.output_tokens.mean():.1f}, mean blocks {h.blocks.mean():.2f}; every request retained. Existing32short B32 traces reused without GPU execution. Three warmups+32×3lanes+2acceptance probes =101 model runs; no unbounded retry.

Current long held-out live-only residual **{h.Residual_Amp.mean():.4f}×**, coding **{h[h.domain=='coding'].Residual_Amp.mean():.4f}×**, reasoning **{h[h.domain=='reasoning'].Residual_Amp.mean():.4f}×**. These replace, rather than assume, previous short5.773×.

## Traffic, lifetime and causal rules

All placement gains are above **perfect already-accepted/context traffic removal**. A remote destination contributes one4096-byte hidden row plus one4096-byte combine row per live source token. Multiple selected experts on one destination still require one row; a destination closes only if every selected expert there is resident. K8 paths all execute freshly; no hidden/output reuse,expert dropping,route/probability/scaling change.

Global Warm is calibrated only on8requests, starts resident, and pays zero per-request copy. Both plain global frequency and stronger global destination-bundle controls are reported. Current request policies profile one/two **complete** blocks, pay those blocks' full live-only traffic, then copy once and retain until request end. No per-block reload, bounded incremental update omitted, no learned model. First-block/first2 frequency ablations are included.

Load caps+10/15/20% apply to **observed history only** for causal policy and calibration only for Global; no test-future pruning. Future actual worst layer/iteration load is then measured independently. Oracle may use request truth. A cap label therefore is NOT a guarantee about future load. Native source-SP placement can change across blocks; history-only safety does not silently become future-truth safety.

The inherited eight-offer destination bundle heuristic is reused with an exact cached-layer implementation, verified against the original on random and rejection cases. Oracle is an achievable future-truth reference, not an exact integer optimum. Global can floor an oracle reference ONLY when Global satisfies that oracle's actual future-load cap. P5/P6 are not deployable methods.

## Calibration freeze and primary held-out result

Before held-out policy summaries, [operating point](../results/frozen_operating_point.json) freezes **{policy}, {budget:g}GiB/rank, observed-history +10% cap**, stronger calibration-selected Global=`{control}`. This is a descriptive point, not a valid deployable candidate. Selecting the best-looking held-out point is prohibited. Statistics use2000 request-bootstrap replicates; iterations,layers,blocks are not independent samples.

Pooled held-out24. **The following byte gains/recoveries are apparent, NOT matched-actual-load gains. Every selected request violates the preferred future load cap; no positive novelty/runtime promotion is inferred.**

- Request-vs-Global additional D+C byte reduction: **{interval(contrast['relative_gain'],100)}%** (95%CI).
- Aggregate byte-ratio additional reduction: **{interval(contrast['aggregate_relative_gain'],100)}%**. The first statistic equally weights request-relative gains; this one divides summed gains by summed Global bytes.
- Global residual **{interval(g['residual_amp'])}×**; causal residual **{interval(target['residual_amp'])}×**.
- Persistent oracle recovery **beyond Global**: **{interval(contrast['persistent_oracle_recovery'],100) if contrast['persistent_oracle_recovery'] else 'undefined'}%**. Negative recovery means the causal plan is worse than Global.
- Coding-primary18 additional reduction vs Global: **{interval(s['selected_coding_contrast']['relative_gain'],100)}%**.
- Positive future-block mean advantage count: **{s['positive_future_blocks']}**, first4-future-block sustained gate={s['gates']['sustained_four_future_blocks']}.

Same budget / same observed guard, actual load explicitly exposed:

| Policy | residual / one-shot | reduction vs live-only | actual peak extra GiB/rank | mean / absolute-worst layer-iteration load increase |
|---|---:|---:|---:|---:|
{chr(10).join(main)}

Unconstrained4GiB future oracle residual **{interval(oracle_stats)}×** is an upper-headroom reference, not a recommendation: inspect its hotspot/copy columns. All budget/guard curves and request rows are in `results/simulation_long_v2/`; the short continuity run is separate.

## Cross-block persistence versus Global

Expert cosine may be high because of global popularity. Future coverage and whole-destination closure against Global are decisive. A top-M diagnostic uses M remote experts **per layer/source**, not the production whole-rank memory budget.

| distance from block0 | request expert cosine | Global cosine | coverage@16 advantage (pp,95%CI) | destination-closure@16 advantage (pp,95%CI) |
|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Full cosine,JSD,top-M overlap/coverage,rank cosine/JSD,destination closure for M4/8/16/32, adjacent and distance2/4/8, first2 profile, and fixed same-domain calibration other-request control are in [persistence CSV](../results/persistence_v2/block_locality.csv). Request-clustered paired CI is separate. `persistence/` retains the preliminary output; `persistence_v2/` fixes tiny negative-roundoff JSD square roots without changing hypotheses/features/selection.

## Copy economics and hotspot

Expert-sized and384MiB P2P measurements on the same12directed GPU edges are reused with hashes. Selected held-out causal mean copies **{c.total_copy_bytes.mean()/1024**3:.3f}GiB/request**, serial expert-copy estimate **{c.copy_serial_expert_ms.mean():.3f}ms**, endpoint-parallel estimate **{c.copy_endpoint_parallel_est_ms.mean():.3f}ms** (not a measured concurrent transfer). Global warm per-request preparation=0.

Byte ROI **{interval(target['byte_roi'])}×**. Observed byte break-even **{s['observed_byte_break_even_requests']}/{s['heldout_requests']}** requests; median extrapolated break-even / remaining refinement horizon **{c.projected_byte_break_even_remaining_fraction.median():.3f}×**. Byte break-even is NOT latency break-even.

**Time ROI and temporal break-even are UNRESOLVED:** replica D+C was not executed. Saved bytes are not converted into stage speedup. Existing timing gives only total baseline D+C/copy as an optimistic ceiling, not avoided communication time. Thus low byte ROI is a structural concern, not a proof that time ROI must be equally low.

Causal request-cumulative max-rank increase **{interval(target['max_rank_load_increase'],100)}%**, mean request worst layer/iteration **{interval(target['worst_stage_load_increase'],100)}%**, **absolute worst {100*target['absolute_worst_stage_load_increase']:.2f}%**. Both cumulative and per-stage counters are retained. Matched future-load validity={contrast['global_and_candidate_actual_load_matched']}; an apparent win with uncontrolled hotspot is not acceptable.

Worst ratios can occur when only a few live paths remain. Absolute per-rank counts are in `absolute_hotspot_counts.csv`; summed per-stage critical path counts actually decrease{100*target['stage_critical_load_increase']['mean']:.2f}%. These counters are a failed predeclared load gate, not a measured latency regression. No runtime result is fabricated in either direction.

## Physical timing and correctness scope

Current baseline D+C / request wall held-out mean **{100*h.comm_share_request_wall.mean():.3f}%**. Timing uses one cumulative-critical rank and all components from that rank, no mixing component maxima. Mean timing observer overhead all32 **{100*v['timing_observer_mean']:.3f}%**; heavy trace is not headline timing. This does not measure replica latency.

No Stage B replica tensors,route redirection,slot eviction,production scheduler,CUDA kernel,quantization or EPLB were implemented. Therefore replica final-token/acceptance/router/NFE parity, GSM8K EM/HumanEval pass@1, and measured D+C/E2E gains are **NOT RUN**, not zero or assumed parity. Baseline physical parity passes; offline accounting preserves unchanged path count and layer identity, which is not runtime correctness.

## Gates and direct answers

`summary_v3.json` is the authoritative audited summary where present. v2 enforces matched-actual-load prerequisites; v3 excludes future-load-unsafe Global sets from constrained-oracle feasible floors. Selected primary point, its CI and verdict are unchanged. Plans, features, thresholds and operating point never change; earlier summaries are preserved. Corrected complete CI is `bootstrap_ci_v3.json`.

| Gate | result |
|---|---|
{chr(10).join('| '+str(k)+' | '+str(x)+' |' for k,x in s['gates'].items())}

1–3. Cross-block locality and decay are quantified above; request-vs-global advantage, not raw cosine, governs novelty.
4–6. Matched-budget additional reduction, horizon and calibration-selected budget are above; no feasible passing memory/load point is asserted.
7–9. Exact one-request copy byte ROI is measured structurally; copy milliseconds use measured P2P. Avoided D+C timing ROI/temporal break-even remain unresolved, byte break-even is shown separately.
10. Worst-stage hotspot is explicitly exposed and can invalidate a history-capped plan.
11–12. Global-after-oracle headroom and causal recovery are in paired CI; recovery denominator is Global minus persistent oracle, never Vanilla minus oracle.
13. Long generation supplies roughly16–17blocks instead of2–3. This alone does not establish request-specific advantage or amortization.
14. Optional AR control not run; no mandatory gate depends on it.
15–18. Stage B not promoted unless all Stage A premises pass; runtime parity/accuracy and measured gains cannot be claimed from offline accounting.
19. Final verdict **{s['verdict']}**.
20. Close dLLM-specific expert-weight replication/movement family={s['close_replication_family']}. Scope is this single-request BF16 source-SP4 regime and the tested causal lifetime policies, not all possible large-batch/cross-request generic replication.

## Next action

{s['next_action']}

No adaptive or learned rescue. If a new direction is authorized, the smallest probe is a frozen-route activation-side experiment measuring what repeated live-token traffic can change without modifying denoising semantics, with the optimistic dead-token-elimination reference kept separate.

## Reproduction and evidence

See [README](../README.md) and [reproduce.sh](../scripts/reproduce.sh). Lossless content-deduplicated complete raw traces are committed under `raw/compact`; original1.1GiB capture remains untouched locally under `raw/long_v1`. Every named NPZ array and original JSON byte is round-trip validated; ZIP-container byte identity is not claimed. Planner-only ID duplicates rebuild from raw and are not redundantly committed. Original SHA and compact SHA are indexed.

Required audited figures in `figures_v2/`: predictor advantage/distance, residual/budget, gain/global/future block, copy ROI/break-even, communication/load Pareto, layer/source/destination allocation, persistent oracle recovery. Preliminary figures are retained in `figures/`. No runtime timing plot is fabricated.

## Final status

BASELINE_STATUS: PASS_TRUE_EP4
REQUEST_COHORT: fresh8calibration/24heldout,32requests,B1,nativeblock32,genbudget512
REQUEST_VS_GLOBAL_GAIN: apparent{interval(contrast['relative_gain'],100)}%; NOT_LOAD_MATCHED
PERSISTENCE_HORIZON: positive mean at{s['positive_future_blocks']}future blocks; sustained4gate={s['gates']['sustained_four_future_blocks']}
BEST_RESIDUAL: selected causal{target['residual_amp']['mean']:.4f}×,Global{g['residual_amp']['mean']:.4f}×,unconstrained4GiBoracle{oracle_stats['mean']:.4f}×
BEST_BUDGET: descriptive{budget:g}GiB/rank; no passing deployed budget
ORACLE_RECOVERY_BEYOND_GLOBAL: apparent{interval(contrast['persistent_oracle_recovery'],100) if contrast['persistent_oracle_recovery'] else 'undefined'}%; NOT_LOAD_MATCHED
COPY_BYTE_ROI: {target['byte_roi']['mean']:.4f}×
TIME_ROI: UNRESOLVED
BREAK_EVEN: observed byte{s['observed_byte_break_even_requests']}/{s['heldout_requests']}; temporalUNRESOLVED
LOAD_IMBALANCE: mean worst{100*target['worst_stage_load_increase']['mean']:.2f}%,absolute worst{100*target['absolute_worst_stage_load_increase']:.2f}%
STAGE_B: {'PROMOTED' if s['stage_b_promoted'] else 'NOT_IMPLEMENTED_STAGE_A_NO_GO'}
CORRECTNESS_ACCURACY: baselinePASS; replica runtimeNOT_RUN
MEASURED_D_C_E2E_GAIN: NOT_MEASURED
VERDICT: {s['verdict']}
NEXT_ACTION: {s['next_action']}
'''
    file=directory/'report.md'
    if file.exists():raise FileExistsError(file)
    file.write_text(text)
    (directory/'baseline_readiness.md').write_text('# Baseline readiness\n\nSee `../results/baseline_validation.json`, `../results/environment.json` and the full report. Physical EP4 sharding/DeepEP ownership is verified on current GPUs; topology is TP4/source-SP4, not Qwen TP2-DP2.\n')
    new_json(ROOT/'results/final_verdict.json',dict(verdict=s['verdict'],stage_b=s['stage_b_promoted'],selected=chosen,
        authoritative_summary=str(summary_path().relative_to(ROOT)),primary_gain_validity='APPARENT_NOT_ACTUAL_LOAD_MATCHED',
        request_vs_global=contrast['relative_gain'],causal_residual=target['residual_amp'],global_residual=g['residual_amp'],
        persistent_oracle_recovery=contrast['persistent_oracle_recovery'],copy_byte_roi=target['byte_roi'],time_roi=None,
        load_worst=target['absolute_worst_stage_load_increase'],measured_D_C_E2E_gain=None,
        close_replication_family=s['close_replication_family'],next_action=s['next_action']))


if __name__=='__main__':main()
