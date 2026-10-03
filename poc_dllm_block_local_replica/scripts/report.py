"""Render explicit answers, not a runtime-success report after an offline gate."""
import json
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *


def fmt(x,scale=1,suffix=''):
    m=x['mean']*scale;lo,hi=[v*scale for v in x['ci95']]
    return f'{m:.3f}{suffix} [{lo:.3f}, {hi:.3f}]'


def main():
    s=json.loads(final_summary_path().read_text());p=s['descriptive_candidate']
    policy,budget=p.split('@');st=s['stats']['heldout_b32_all'];primary=s['stats']['heldout_b32_humaneval']
    c=st[p];o=st[f'oracle@{budget}'];load=st[f'oracle_load@{budget}']
    f=json.loads((ROOT/'results/design_freeze.json').read_text());baseline=json.loads((ROOT/'results/baseline_validation.json').read_text())
    p2p=pd.read_csv(ROOT/'results/copy_cost/p2p_summary.csv')
    copy=pd.read_csv(ROOT/'results/copy_cost/block_copy_roi.csv');subset=copy[(copy.policy==policy)&(copy.budget_gib==float(budget))&copy.request.str.endswith('_b32')&copy.request_id.isin(f['heldout'])]
    old=pd.read_csv(SOURCE/'results/main_v2/requests.csv');hold=old[(old.block_length==32)&old.request_id.isin(f['heldout'])]
    table=['|Policy|GiB/rank|Residual x|D+C byte reduction|Oracle recovery|Byte ROI|Max-rank cumulative load increase|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for b in BUDGETS:
        for method in ('global','first_frequency','first_destination','first2_destination','oracle','oracle_load'):
            r=st[f'{method}@{b:g}']
            table.append(f"|{method}|{b:g}|{r['residual_amp']['mean']:.3f}|{r['dc_reduction']['mean']*100:.2f}%|{r['oracle_recovery']['mean']*100:.2f}%|{r['byte_roi']['mean']:.5f}|{r['max_rank_load_increase']['mean']*100:.2f}%|")
    expert=p2p[p2p['mode']=='actual_two_tensor_P2P'];bulk=p2p[(p2p['mode']=='contiguous_P2P')&(p2p.bytes==384*1024**2)]
    text=f'''# Layer-Aware Block-Local Expert Replication: Stage-A NO-GO

The structural oracle has large destination-closure headroom, but simple early-block causal plans recover too little of it and cold expert copies have very poor byte ROI. **Stage B was not promoted. No replica runtime, speedup, or benchmark-accuracy parity is claimed.**

## Contract and physical baseline

- Source of truth: `references/authoritative_spec.md`, SHA256 `{sha(SPEC)}`.
- Model: inclusionAI/LLaDA2.0-mini, revision `{REVISION}`, BF16, 19 sparse layers, 256 routed experts/layer, K8, hidden2048, intermediate512, routed scaling2.5. No model/router/precision changes.
- Actual topology: denseTP4 / source-SP4 / physicalEP4 / DP1 / PP1. This is the validated dLLM topology, not Qwen TP2-DP2. Linear64 experts/rank.
- Physical H10080GB GPUs2,3,6,7 -> logical0,1,2,3, NV18 links. UUIDs/topology/process/version audit: `results/environment.json`, `raw/physical_revalidation/p2p/clean_p2p_start.json`.
- Reused source commit `{SOURCE_COMMIT}`. Exact loaded w13 `[64,1024,2048]` BF16 and w2 `[64,2048,512]` BF16: **6,291,456 bytes =6MiB per (layer,expert)**. All19 layers counted in the shared per-source budget; no multiplication of the budget per layer.
- Fresh physical revalidation: {len(baseline['probes'])} bounded probes, same token IDs/acceptance/NFE/router IDs/weights and received owner-local expert paths as prior capture. TRUE DeepEP sparse dispatch/combine is directly checked. Current/prior routing arrays are compared, not assumed equal.
- GPU session uses scoped burner stop, competing-process check, watchdog and finally restoration. No other GPU process was modified.

## Cohort and causal validity

Prior frozen32 problems (16 HumanEval,16 GSM8K) are reused, plus8 block64 robustness captures. New replica calibration split:4/domain, held-out12/domain. Split seed8611, fixed before replica outcome evaluation. These are a newly held-out **replica-policy split**, not fresh unseen generations: prior amplification statistics were already known.

HumanEval is the declared communication-exposed primary regime; GSM8K is secondary, with balanced pooled continuity also reported. Block32 native threshold0.9, temperature0, generation budget64, prefix block-diffusion cache, EOS enabled are unchanged. No favorable-output selection and no new full-model capture beyond physical sanity checks.

Only masked output positions **before each actual forward** enter `B_live_only`. Context and already-accepted output positions are excluded. Native cross-block source partitions are preserved; source rank is not re-derived from token ID. Each request's original `B_one_shot` stays frozen.

Primary no-replica continuity (all32):5.773x. This24-problem held-out baseline:{fmt(c['baseline_amp'],suffix='x')}. Different cohorts must not be silently mixed. The original one-shot1x denominator does not itself have replicas; localized communication may fall below1x without violating accounting.

Global/static uses normalized calibration request popularity only and can be ready at block start. First-iter/first2 plans use only ordinal0 / ordinals0–1 respectively. Their profiling iteration traffic is charged unchanged; plan applies only afterward. Future-truth policies alone inspect future routes. Tests perturb future routing and check causal plans do not change. Static is an intentionally strong control; its cold-copy charge is also reported, while persistent static replicas could amortize better and are outside this block-local contract.

## Policy result (24 held-out problems, request-macro means)

{chr(10).join(table)}

All mean/median/95% request-bootstrap CIs are in `results/bootstrap_ci.json` (2000 resamples). No layer or iteration is treated as an independent serving sample.

The future-truth oracle is a destination-bundle greedy **achievable reference**, not an exact combinatorial optimum. A fractional LP relaxation supplies an upper bound at0.5/4GiB (stored per request); no fractional plan is executed or counted as physical replication. Recovery relative to a stronger exact oracle could only be lower.

Primary HumanEval held-out (12 problems), at the same calibration-selected descriptive point: live-only {primary[p]['baseline_amp']['mean']:.3f}x -> oracle {primary[f'oracle@{budget}']['residual_amp']['mean']:.3f}x -> causal {primary[p]['residual_amp']['mean']:.3f}x. Causal reduction {primary[p]['dc_reduction']['mean']*100:.3f}%, oracle recovery {primary[p]['oracle_recovery']['mean']*100:.3f}%; global/static residual {primary[f'global@{budget}']['residual_amp']['mean']:.3f}x. Coding-primary and pooled numbers are separate estimands.

## Direct answers to the17 research questions

1. **Oracle reduction:** at{budget}GiB/rank, {fmt(o['dc_reduction'],100,'%')} of live-only D+C bytes; large structural opportunity exists.
2. **Oracle residual:** {fmt(o['residual_amp'],suffix='x')}; load-capped future-truth residual {fmt(load['residual_amp'],suffix='x')}.
3. **Memory:** swept0.5/1/2/4GiB/rank. Calibration-selected descriptive causal point `{p}` has mean peak extra memory{c['peak_memory_gib']['mean']:.3f}GiB/rank; no point is declared a runtime operating point. Burner-free audit and loaded baseline size leave these budgets feasible in80GB HBM.
4. **Layer dependence:** replicas are selected independently by(layer,source,expert) under one joint capacity. `figures/layer_replica_allocation.png` and `results/layer_allocation_matrix.csv` show19 separate sparse-layer profiles with source/owner identities. Expert17 at two layers is charged twice.
5. **Destination vs frequency:** at0.5GiB, first-iteration destination-aware adds {fmt(s['paired_contrasts']['heldout_b32_all']['first_destination-first_frequency@0.5'],100,'pp')} reduction over raw frequency. At4GiB first-iteration choices saturate and are identical. Positive low-budget structural improvement is not sufficient for runtime promotion.
6. **Causal vs static:** selected `{p}` adds {fmt(s['paired_contrasts']['heldout_b32_all'][f'{policy}-global@{budget}'],100,'pp')} reduction versus global; at large budget static wins. Smaller-budget causal advantages exist but do not satisfy all gates.
7. **Recovered oracle:** selected causal {fmt(c['oracle_recovery'],100,'%')}; all first/first2 causal points in pooled held-out remain below50%. HumanEval at selected point: {fmt(primary[p]['oracle_recovery'],100,'%')}.
8. **Copy amortization:** selected causal byte ROI {fmt(c['byte_roi'],suffix='x')} versus2x gate. Median block byte break-even estimate {subset.byte_break_even_iterations.median():.1f} future iterations at the observed avoidance rate, versus median{int(subset.remaining_iterations.median())} remaining iterations. This is a byte-volume extrapolation, not a measured latency break-even. An optimistic policy-independent bound over all40 captures is{s['copy_policy_independent_bound']['maximum_any_policy_byte_roi_bound']:.3f}x, still below2x. P2P expert-sized median across edges:{expert.median_ms.median():.4f}ms;384MiB bulk median BW:{bulk.GB_per_s.median():.1f}GB/s.
9. **Compute hotspot:** selected causal cumulative max-rank assignment increase {fmt(c['max_rank_load_increase'],100,'%')}; worst iteration/layer maximum increase {fmt(c['worst_stage_load_increase'],100,'%')}. Sum-of-stage maxima changes only {fmt(c['stage_critical_load_increase'],100,'%')}; this aggregate can hide large instantaneous hotspots and is not a measured expert latency model.
10. **Pareto:** load-capped oracle at4GiB cuts {fmt(load['dc_reduction'],100,'%')} while each iteration/layer max is capped at+10%. It still has cold-copy cost and needs future truth. At2GiB it nearly saturates ({st['oracle_load@2']['dc_reduction']['mean']*100:.2f}% reduction). No causal point passes the full recovery/copy/load chain. See `figures/comm_vs_load_pareto.png`.
11. **Stage B:** NOT_IMPLEMENTED, as required by failed Stage-A gates. No dynamic placement runtime or learned predictor was added.
12. **Measured D+C latency gain:** NOT_MEASURED for replicas; offline byte reduction is not speedup.
13. **Measured E2E latency gain:** NOT_MEASURED for replicas. P2P times do not establish an equivalent measured dispatch/combine timing argument.
14. **Correctness:** physical baseline parity PASS on2 probes; structural accounting preserves fresh K8 work and weights conceptually. Replica token/acceptance/router/NFE/tensor/combine parity NOT_TESTED because no runtime exists. Do not call offline invariance measured runtime correctness.
15. **Accuracy:** GSM8K exact match/HumanEval pass@1 replica comparisons NOT_RUN. Bounded prior generations are not full benchmark completions; no fabricated zero-delta score.
16. **Verdict:** **NO-GO** for simple first1/2-iteration, cold block-local exactBF16 replicas in this low-request source-SP4 regime. Oracle headroom passes, but causal recovery fails, chosen large-budget causal policy loses to static, byte ROI fails, and substantial instantaneous compute hotspots appear. Negative conclusion does not rule out persistent cross-block warm replicas, much larger batch or another EP topology.
17. **Smallest next experiment:** before any runtime, measure cross-block/request persistence under one small persistent replica cache and its cumulative destination-closure byte/copy ROI. Only if that amortizes across much longer lifetimes should this exact-placement direction be revisited.

## Copy-cost estimates and limitations

Every selected(layer,expert,source) incurs one full6MiB copy each block. Selected causal request estimates: serial two-tensor copy{fmt(c['copy_serial_expert_ms'],suffix='ms')}; endpoint-parallel estimate{fmt(c['copy_parallel_expert_est_ms'],suffix='ms')}; optimistic bulk estimates serial{fmt(c['copy_serial_bulk_est_ms'],suffix='ms')}, parallel{fmt(c['copy_parallel_bulk_est_ms'],suffix='ms')}. Parallel/bulk costs are estimates from measured single-edge P2P, not measured concurrent scattered-expert copies.

**Byte ROI is not timing ROI.** DeepEP has fixed/protocol costs and P2P copy bandwidth differs from DeepEP hidden-row traffic. Actual avoided D+C time, time ROI and runtime break-even are UNAVAILABLE. Do not conclude the measured timing can never amortize merely from byte ROI. Stage A still fails independently: early causal oracle recovery is below50% at every budget, especially in the coding primary regime. No byte-to-ms speedup conversion is used to rescue that failed causal premise.

Baseline primary continuity D+C/request wall17.974%, D+C/denoising span19.255% remain **measured prior physical facts**. Structural dead-token elimination5.773x is **not measured Epoch**; subtracting those dead-token bytes is not this PoC's contribution.

## Main table (matched24-problem holdout; structural rows not runtime)

|Method|Quality|Norm. EP traffic|D+C ms|E2E ms|Speedup|Extra HBM/rank|
|---|---|---:|---:|---:|---:|---:|
|Vanilla physical baseline|bounded baseline; no full task score|{hold.Raw_Amp.mean():.3f}x|{hold.comm_ms.mean():.2f}|{hold.timing_wall_ms.mean():.2f}|1.00x|0|
|Perfect stable/dead-token elimination structural reference|n/a|{c['baseline_amp']['mean']:.3f}x|n/a|n/a|n/a|0|
|Global/static structural replicas ({budget}GiB)|n/a; runtime not run|{st[f'global@{budget}']['residual_amp']['mean']:.3f}x|n/a|n/a|n/a|<= {budget}GiB|
|Block-local causal structural replicas ({budget}GiB)|n/a; runtime not run|{c['residual_amp']['mean']:.3f}x|n/a|n/a|n/a|<= {budget}GiB|
|Oracle block-local structural replicas|n/a|{o['residual_amp']['mean']:.3f}x|n/a|n/a|n/a|<= {budget}GiB|
|Original acceptance-route one-shot reference|n/a|1.000x|n/a|n/a|n/a|n/a|

Vanilla stage time uses one cumulative-critical rank per request (all components from that same rank), event-only timing lane. Old validated observer overhead is8.284%mean; heavy router-capture wall is not headline timing. No structural row reuses Vanilla timings as if it were measured candidate timing.

## Reproduction and artifacts

Run `bash poc_dllm_block_local_replica/scripts/reproduce.sh` to create a fresh append-only output directory; raw source paths are pinned to the parent evidence commit. Required GPU work is only the short physical revalidation/P2P lane; all policy evaluation reuses prior rich trace. Tests: `python -m pytest -q poc_dllm_block_local_replica/tests`.

`results/design_freeze.json`, `results/reused_trace_manifest.json`, `data/live_routes/*.npz`, `results/simulation/*`, `results/copy_cost/*`, `results/bootstrap_ci.json`, `results/summary.json` carry reproducible compact evidence. No unrelated dirty/staged work was included.

## Final fields

BASELINE_STATUS: TRUE_PHYSICAL_EP4_REVALIDATED_PASS
LIVE_ONLY_CONTINUITY: 5.773x all32; {c['baseline_amp']['mean']:.3f}x this24 heldout
ORACLE_RESIDUAL: {o['residual_amp']['mean']:.3f}x at{budget}GiB/rank
CAUSAL_RESIDUAL: {c['residual_amp']['mean']:.3f}x ({p})
ORACLE_RECOVERY: {c['oracle_recovery']['mean']*100:.2f}%
MEMORY_BUDGET: {budget}GiB/rank descriptive; NO qualifying runtime point
COPY_AMORTIZATION: BYTE_ROI {c['byte_roi']['mean']:.5f}x; TIME_ROI UNAVAILABLE
COMPUTE_IMBALANCE: cumulative max-rank +{c['max_rank_load_increase']['mean']*100:.2f}%; worst-stage +{c['worst_stage_load_increase']['mean']*100:.2f}%
STAGE_B: NOT_PROMOTED
MEASURED_D_C_GAIN: NOT_MEASURED
MEASURED_E2E_GAIN: NOT_MEASURED
REPLICA_CORRECTNESS_ACCURACY: NOT_RUN; physical baseline parity PASS
VERDICT: NO-GO
NEXT_ACTION: CLOSE_COLD_BLOCK_LOCAL_DIRECTION; optional offline cross-block lifetime test only
'''
    path=ROOT/'report.md'
    if path.exists():raise FileExistsError(path)
    path.write_text(text)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    notes={
        'baseline_readiness.md':'TRUE EP4 physical sanity, exact tensor ownership and current/prior output+router parity PASS. See ../results/baseline_validation.json and physical_revalidation/ep4 raw. GPUs2,3,6,7 only.',
        'oracle_budget.md':'Destination-closure future-truth greedy is an achievable reference, with fractional LP upper bounds at0.5/4GiB. Memory is joint across19 layers/source, never a per-layer budget. Main budget table and CIs: ../report.md and ../results/bootstrap_ci.json.',
        'causal_planner.md':'Global uses4 calibration problems/domain only; first1/2 policies use only completed current-block iterations. Profiling traffic is unchanged. Future-perturbation test passes. Every causal policy recovers less than50% of future-truth greedy oracle across pooled primary budgets.',
        'copy_amortization.md':'Physical6MiB two-tensor and96/384MiB bulk P2P measurements on all12 directed authorized-GPU pairs. Cold per-block byte ROI fails2x. Parallel/bulk costs estimated, not observed all-edge/runtime copy. No actual D+C time saving inferred from bytes.',
        'load_balance.md':'Cumulative rank load and worst iteration/layer maxima are both reported. Aggregation can hide hotspots. Load-aware oracle caps each stage max to1.10x, not accumulated per-rank totals. Assignment counts are a compute-load proxy, not measured execution time.',
        'stage_b_gate.md':'NO-GO: causal recovery, copy byte ROI and large-budget causal>static conditions fail. Runtime replication, replica accuracy, and speedup measurements intentionally NOT_RUN. No complex predictor rescues the direction.',
    }
    for name,note in notes.items():(reports/name).write_text('# '+name.replace('_',' ').replace('.md','')+'\n\n'+note+'\n')
    rows=[dict(method='Vanilla',quality='bounded baseline only',Norm_EP_traffic=hold.Raw_Amp.mean(),D_C_ms=hold.comm_ms.mean(),E2E_ms=hold.timing_wall_ms.mean(),speedup=1.,extra_mem_gib=0.),
        dict(method='Perfect stable/dead-token elimination structural reference',quality='n/a',Norm_EP_traffic=c['baseline_amp']['mean']),
        dict(method='Global structural',quality='runtime_not_run',Norm_EP_traffic=st[f'global@{budget}']['residual_amp']['mean'],extra_mem_gib=float(budget)),
        dict(method='Block-local structural',quality='runtime_not_run',Norm_EP_traffic=c['residual_amp']['mean'],extra_mem_gib=float(budget)),
        dict(method='Oracle structural',quality='n/a',Norm_EP_traffic=o['residual_amp']['mean'],extra_mem_gib=float(budget)),
        dict(method='One-shot structural reference',quality='n/a',Norm_EP_traffic=1.)]
    pd.DataFrame(rows).to_csv(ROOT/'results/main_table.csv',index=False)


if __name__=='__main__':main()
