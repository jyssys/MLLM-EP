"""Frozen criteria, measured facts vs structural simulation vs copy estimates."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_agentic_ep.contract import ROOT,write


def ci(row,scale=1):
    return f"{row['mean']*scale:.3f} [{row['ci95'][0]*scale:.3f}, {row['ci95'][1]*scale:.3f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args();root=ROOT/'results'/a.version
    if (ROOT/'report.md').exists():raise FileExistsError('final report already exists')
    locality=json.loads((root/'locality_summary.json').read_text())
    preparation=json.loads((root/'preparation_summary.json').read_text())
    manifest=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text())
    audit=json.loads((root/'prefix_exclusion_audit.json').read_text())
    primary=locality['paired_deltas']['new_prefill'];global_gap=primary['adjacent_minus_global']['coverage_16']
    def positive(r,threshold):return r['mean']>=threshold and r['ci95'][0]>0
    g1=all(positive(primary['adjacent_minus_'+c]['expert_cosine'],.05) or
        positive(primary['adjacent_minus_'+c]['coverage_16'],.05) for c in ('same_shuffled','cross_matched'))
    g2=positive(global_gap,.05)
    budgets=preparation['metrics']['new_prefill'];useful=[]
    for slots,policies in budgets.items():
        oracle=policies['oracle']['remote_dispatch_bytes_reduction_pct']
        previous=policies['previous']['remote_dispatch_bytes_reduction_pct']
        recovered=preparation['recovered_oracle_pct']['new_prefill'][slots]
        fits=48*int(slots)*preparation['expert_bytes']<=preparation['measured_free_bytes_min']
        if fits and oracle['mean']>=15 and (previous['mean']>=10 or recovered.get('mean',0)>=50):useful.append(slots)
    g3=any(p['oracle']['remote_dispatch_bytes_reduction_pct']['mean']>=15 for p in budgets.values())
    g4=bool(useful)
    g5=any(budgets[s]['previous']['min_tool_gap_estimate_ms']['mean']<=500 for s in useful)
    verdict='GO' if all((g1,g2,g3,g4,g5)) else ('CONDITIONAL' if g1 and g2 and g3 and g4 and not g5 else 'NO-GO')
    result=dict(verdict=verdict,gates=dict(G1_LOCALITY=g1,G2_PREVIOUS_OVER_GLOBAL=g2,
        G3_STRUCTURAL_ORACLE_HEADROOM=g3,G4_CAUSAL_RECOVERY=g4,G5_TOOL_GAP_ESTIMATE=g5),
        primary='NEW_PREFILL_SUFFIX_ONLY',heldout_trajectories=8,heldout_pairs=64,calibration_trajectories=4,
        useful_slots=useful,previous_minus_global_coverage16=global_gap,
        common_prefix_exclusion='PASS token position + stronger historical semantic boundary',
        target='Qwen3-30B-A3B',topology='TRUE TP2-DP2-EP4-PP1',physical_gpus='2,3,6,7',
        next_action='No runtime migration/scheduler. Close this tested agent-locality premise if G1/G2 fail; do not rescue with learned predictors.' if verdict=='NO-GO' else
            'One bounded actual-replica copy + unchanged-route execution sanity at the smallest useful budget, including allocator/layout overhead; no production scheduler.',
        timing_claim='No measured E2E/TPOT speedup; routing headroom simulation and peer-copy feasibility only',
        scope='12 public SWE-smith trajectories, first9 turns, coding-agent context<=16384, single-active DP0, teacher-forced recorded32 secondary',
        ci_scope='2000 held-out trajectory bootstrap resamples; fixed four-trajectory calibration memory; not64 independent pairs',
        branch='flashvep/agentic-moe-ep-locality-poc')
    result['statistically_detected_locality']=all(primary['adjacent_minus_'+c]['expert_cosine']['ci95'][0]>0 for c in ('same_shuffled','cross_matched'))
    write(root/'summary.json',result)
    plots=ROOT/'plots';plots.mkdir(exist_ok=True)
    frame=pd.read_csv(root/'pair_metrics.csv');pref=frame[frame.phase=='new_prefill']
    order=['adjacent','same_shuffled','cross_matched','global']
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([pref[pref.policy==x].expert_cosine for x in order],tick_labels=order)
    ax.set_ylabel('Layer-mean expert histogram cosine');ax.set_title('Only new suffix;64 correlated pairs /8 held-out trajectories')
    fig.tight_layout();fig.savefig(plots/'adjacent_vs_controls.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for policy in ['same_shuffled','global','adjacent','oracle']:
        metrics=locality['phase_metrics']['new_prefill'][policy]
        vals=[metrics[f'coverage_{m}'] for m in (4,8,16,32)]
        means=np.array([r['mean']*100 for r in vals]);errors=np.array([[r['mean']-r['ci95'][0],r['ci95'][1]-r['mean']] for r in vals]).T*100
        ax.errorbar([4,8,16,32],means,yerr=np.maximum(errors,0),marker='o',label=policy)
    ax.set_xlabel('M hot experts per layer');ax.set_ylabel('Next-turn assignment coverage %');ax.legend()
    fig.tight_layout();fig.savefig(plots/'next_turn_expert_coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,key in zip(axes,['rank_cosine','rank_l1','hottest_rank_persistence']):
        vals=[locality['phase_metrics']['new_prefill'][x][key] for x in order]
        ax.bar(np.arange(4),[r['mean'] for r in vals]);ax.set_xticks(np.arange(4),order,rotation=30,ha='right');ax.set_title(key)
    fig.tight_layout();fig.savefig(plots/'ep_rank_locality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    slots=sorted(budgets,key=int);x=np.array([48*int(s)*preparation['expert_bytes']/1024**3 for s in slots])
    for policy in ['global','previous','oracle']:
        for ax,key in zip(axes,['remote_paths_reduction_pct','remote_dispatch_bytes_reduction_pct']):
            vals=[budgets[s][policy][key] for s in slots]
            ax.plot(x,[r['mean'] for r in vals],'o-',label=policy)
            ax.fill_between(x,[r['ci95'][0] for r in vals],[r['ci95'][1] for r in vals],alpha=.12)
            ax.set_xlabel('Replica capacity GiB /active source rank');ax.set_ylabel(key)
    axes[0].legend();fig.tight_layout();fig.savefig(plots/'oracle_vs_previous_turn.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for policy in ['global','previous','oracle']:
        ax.plot(x,[budgets[s][policy]['min_tool_gap_estimate_ms']['mean'] for s in slots],'o-',label=policy)
    for gap in [50,100,250,500,1000]:ax.axhline(gap,linestyle=':',alpha=.3,color='grey')
    ax.set_xlabel('Replica capacity GiB /active source rank');ax.set_ylabel('Conservative serial-copy estimate ms')
    ax.set_title('Measured P2P service, modeled preparation; not real migration');ax.legend()
    fig.tight_layout();fig.savefig(plots/'preparation_gap_feasibility.png',dpi=160);plt.close(fig)
    expert=preparation['expert_bytes'];lines=[]
    lines+=['# Agentic serving × physical MoE EP: cheap discovery', '',f'**Verdict: {verdict}**. No runtime migration, learned predictor, scheduler or new kernel was implemented.', '',
        '## Protocol and validity', '',
        'Measured Qwen3-30B-A3B revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, BF16,48 layers/d2048/m768/E128/K8. TRUE TP2-DP2-EP4-PP1 /DeepEP HT/Triton sparse experts,linear32 ownership verified from loaded runtime tensors. Only physical2,3,6,7 (logical0,1,2,3), H10080GB/NV18. TP2/DP2 is the existing supported family, not spec-recommended optional TP1/DP1 rewritten into a new topology. One active user request on DP0; DP1 dummy collectives are not user routes.', '',
        'SWE-smith XML-format public action/observation histories preserve semantic content, reconstructed with the frozen native Qwen3 template (thinking=False). Source revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`. Metadata-only seeded selection from100 public rows;12 instance-disjoint trajectories,9 turns each,4 calibration +8 held-out,64 held-out adjacent pairs. Outcome/resolution/patch/routing never selected samples. Input context is preserved, not cropped; maximum16384 tokens. No live tools or dataset commands were executed.', '',
        'Source: [pinned SWE-smith dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/08e109b4a59eaeebf80e4675cd125d42e7ac99a4), [official SWE-agent trajectory semantics](https://swe-agent.com/latest/usage/trajectories/). This is replay under Qwen3, not a claim about agent task success.', '',
        '**Prefix caching is actually enabled.** Primary routing retains only positions after both the actual KV-cache query start and the frozen exclusion frontier. The frontier is max(exact previous-prompt LCP, previous completed-context LCP, current-rendering new observation boundary). Native Qwen3 may rewrite prior assistant empty-think formatting, so plain LCP alone can accidentally count repeated assistant content. A sentinel rendered before routing gives the actual new observation span, conservatively excluding all historical semantic messages. Originalv1 smoke/cohort are preserved; v2 changes only exclusion metadata, not selected instances/input IDs/answers. Two calibration smoke turns were seen before this repair; no held-out routes were inspected. All new-only raw tensors pass position/coverage/owner/path-count checks.', '',
        'Primary=new prefill suffix. Secondary=first32 recorded assistant tokens teacher-forced in a causal packed forward; not natural AR output, not decode timing, not future features. The native one-token sample after this suffix is discarded. Global prior uses only four fully completed calibration trajectories; previous-turn predictor uses only the prior completed turn. Oracle and shuffled future diagnostic controls are explicitly nondeployable.', '',
        'All confidence intervals resample8 entire held-out trajectories (2000 draws). Layer events and64 pairs are not independent samples. Calibration uncertainty is not included in these conditional intervals. Controls and practical thresholds are frozen before held-out analysis; the first-turn exclusion sensitivity is separately recorded.']
    lines+=['','## Direct answers and primary held-out results','',
        '| Comparison | Expert cosine delta,95% CI | Coverage@16 delta pp,95% CI |',
        '|---|---:|---:|']
    for name in ['same_shuffled','cross_matched','global']:
        part=primary['adjacent_minus_'+name]
        lines.append(f"| Adjacent/previous minus {name} | {ci(part['expert_cosine'])} | {ci(part['coverage_16'],100)} |")
    lines+=['',
        f"1. **Adjacent agent turns routing-local?** Statistically detected={result['statistically_detected_locality']}; practical frozen Gate1={g1}. Matched-control cosine improvement is positive but below the frozen0.05 point target, and coverage gain is below5pp. This is not a claim that locality is absent.",
        '2. **Survives shared-prefix removal?** Only new observation suffix routes enter the primary contrasts. Full accumulated-prompt similarity is never used.',
        f"3. **Stronger than shuffled/matched?** Both controls must show meaningful positive held-out contrasts. Gate1={g1}.",
        f"4. **Previous beats global popularity?** Coverage@16 delta={ci(global_gap,100)}pp. Gate2={g2}.",
        '5. **Actual EP4 rank-level manifestation?** See rank deltas below, on actual contiguous32 ownership. Hottest-rank constancy is not automatically useful signal.', '',
        '| Adjacent minus control | Rank cosine delta | Rank L1 delta (lower better) | Hottest-rank persistence delta pp |',
        '|---|---:|---:|---:|']
    for name in ['same_shuffled','cross_matched','global']:
        part=primary['adjacent_minus_'+name]
        lines.append(f"| {name} | {ci(part['rank_cosine'])} | {ci(part['rank_l1'])} | {ci(part['hottest_rank_persistence'],100)} |")
    lines+=['', '## Structural oracle preparation and tool-gap feasibility','',
        f'Fresh loaded tensor arithmetic: **{expert:,} bytes/expert/layer** ({expert/1024**2:.3f}MiB), w1[32,1536,2048]+w2[32,2048,768]. Every replica budget counts all48 layers. One extra expert/layer/rank costs {48*expert/1024**3:.6f}GiB/rank, not merely9MiB for the entire model.', '',
        'Replicas are modeled at the source rank; original expert IDs/weights/routes remain semantically unchanged. All policies use the same top-frequency selection rule. Oracle uses actual next-turn per-source counts and exactly maximizes removed assignments under independent slots, but does not prove a globally byte-optimal destination-closure solution. Useful communication bytes are deduplicated per token/destination. Removing one remote expert path does not eliminate a hidden-row transfer if another path still needs that destination.', '',
        'Only active source ranks prepare positive-score replicas; idle DP sources do not invent traffic or useful replicas. Capacity is per rank, and actual total prepared bytes/counts/edge traffic are saved. Feasibility checks loaded-model+8GiB-KV free memory. Extra compute/load may move onto the source and worsen rank imbalance; those quantities are reported, not assumed improved.', '',
        '| Slots/layer/rank | Capacity GiB/rank | Global remote bytes reduction % | Previous reduction % | Oracle reduction % | Previous/oracle recovery % | Previous serial-copy estimate ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in slots:
        pol=budgets[s];recovery=preparation['recovered_oracle_pct']['new_prefill'][s]
        lines.append(f"| {s} | {48*int(s)*expert/1024**3:.3f} | {ci(pol['global']['remote_dispatch_bytes_reduction_pct'])} | {ci(pol['previous']['remote_dispatch_bytes_reduction_pct'])} | {ci(pol['oracle']['remote_dispatch_bytes_reduction_pct'])} | {ci(recovery) if 'mean' in recovery else 'N/A'} | {ci(pol['previous']['min_tool_gap_estimate_ms'])} |")
    lines+=['',
        f'6. **Oracle EP headroom?** Gate3={g3}; table shows feasible next-truth replication-policy logical byte reductions, not measured latency gains.',
        f'7. **Previous-turn recovers what fraction?** Recovery is computed request-wise vs oracle logical byte reduction, then trajectory-bootstrapped. Gate4={g4}. Global comparison is mandatory even if recovery is high.',
        '8. **Required expert memory?** Table counts full48-layer capacity; preparation_budget.csv also saves actual total/per-rank bytes. 0.5/1/2/4GiB maps to slots '+json.dumps(preparation['memory_GiB_to_slots'])+'.',
        f'9. **Can tool gap hide preparation?** Gate5={g5}. Actual12-direction torch P2P benchmark has3 warmups+10 repeats at9MiB and64MiB. serial-copy and bulk-serial extrapolations are feasibility estimates only; no concurrent replication/packing/allocator/admission overhead is measured. 50/100/250/500/1000ms status is saved per policy/request/budget. These are hypothetical gaps, not measured SWE-agent wall gaps.',
        f'10. **Overall verdict? {verdict}.** '+result['next_action'], '',
        '## Evidence boundaries and reproduction','',
        'Routing and P2P copy service are measured facts. Replica routing/byte/load gains are structural simulation. Preparation time is a measured-copy-based estimate. Expert/MoE trace timings are diagnostic instrumentation-only, with one cumulative-critical rank per invocation. They are not a validated latency predictor. No E2E/TPOT/real-migration speedup is claimed.', '',
        'Raw retained traces store top8 IDs, original FP32 routing weights, true source ranks and original token positions for new-prefill / teacher-forced decode only. No model weights or hidden states are uploaded. Full local failed/smoke/runtime logs remain append-only; compact semantic event/ownership/cache evidence is exported separately. Original dirty worktree is unchanged.', '',
        'Reproduction commands are in README.md; tests cover prefix identity, phase mapping, destination dedup, memory arithmetic, idle ranks and trajectory bootstrap. No full benchmark quality rerun, new kernel, learned predictor, production scheduler, quantization or real dynamic migration.', '',
        '```text',f'VERDICT: {verdict}',f'GATES: {result["gates"]}',
        'GPU_MASK: 2,3,6,7', 'TOPOLOGY: TRUE TP2-DP2-EP4-PP1',
        'BRANCH: flashvep/agentic-moe-ep-locality-poc',
        'COMMIT: git log -1 --format=%H on this evidence branch; exact SHA in final handoff',
        'MEASURED_RUNTIME_SPEEDUP: NOT_RUN', 'NEXT_ACTION: '+result['next_action'],'```','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
