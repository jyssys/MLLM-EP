"""Join independent clean timings to canonical traced inputs, not to time bins.

Trace route features are *not* measured again by the clean pass. Alignment
requires identical request output, prefix length, draft IDs, verified rows and
progress. This controls inputs; it does not assert bit-exact repeated routing.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,write
from poc_speculative_moe_ep.scripts.analyze import extract,cost_models,matched_pairs,depth_disagreement,bootstrap_mean


KEY=['request_id','gamma','committed_position','verified_rows','committed_progress']


def traced_table(version):
    root=ROOT/'artifacts'/version
    table=pd.read_parquet(root/'verification_metrics.parquet')
    layers=pd.read_parquet(root/'route_metrics.parquet')
    layers['actual_max_active_experts_rank']=layers.actual_active_experts_per_rank.map(max)
    layers['actual_expert_histogram_skew']=layers.actual_expert_histogram.map(lambda x:max(x)/(sum(x)/128))
    extra=layers.groupby(['run','wave','forward'])[['actual_max_active_experts_rank','actual_expert_histogram_skew']].mean().reset_index()
    table=table.merge(extra,on=['run','wave','forward'],validate='one_to_one')
    drafts={}
    for run in table.run.unique():
        for line in (ROOT/'raw'/run/'frames.rank0.jsonl').read_text().splitlines():
            f=json.loads(line)
            drafts[(run,f['wave'],f['forward'])]=list(next(iter((f.get('scheduler') or {}).get('draft_ids',{}).values()),[]))
    table['draft_token_ids']=[drafts[(r.run,r.wave,r.forward)] for r in table.itertuples()]
    assert not table.duplicated(KEY).any(), 'one immutable trace realization per canonical invocation required'
    return table,json.loads((root/'parity.json').read_text())


def same_prefix_oracle(table):
    """Available common-prefix alternatives only; no invented missing costs."""
    # Median across repeat is a paired opportunity cost, not a simulated
    # adaptively generated trajectory. gamma0 denotes AR, not SD gamma1.
    t=table.groupby(['request_id','gamma','committed_position']).agg(
        ep=('ep_comm_ms','median'),target=('target_forward_ms','median'),
        progress=('committed_progress','first'),experts=('expert_union','first'),
        accepted=('raw_accepted_draft_tokens','first'),
        drafts=('draft_token_ids','first')).reset_index()
    opportunities=[]
    for (rid,pos),g in t.groupby(['request_id','committed_position']):
        if set(g.gamma)!={0,2,4,8}:continue
        rows={int(r.gamma):r for r in g.itertuples()}
        # Widths can be nonnested in native drafting; retain the distinction.
        nested=all(list(rows[k].drafts)==list(rows[8].drafts)[:len(rows[k].drafts)] for k in (2,4))
        cost={k:dict(ep=r.ep/r.progress,target=r.target/r.progress,experts=r.experts/r.progress,
                    progress=int(r.progress),accepted=int(r.accepted)) for k,r in rows.items()}
        ep=min(cost,key=lambda k:cost[k]['ep']);expert=min(cost,key=lambda k:cost[k]['experts'])
        acceptance=max(cost,key=lambda k:cost[k]['progress']);aggregate=min(cost,key=lambda k:cost[k]['target'])
        opportunities.append(dict(request_id=rid,committed_position=int(pos),nested_draft_prefix=nested,
            ep_gamma=ep,expert_gamma=expert,acceptance_gamma=acceptance,aggregate_gamma=aggregate,cost=cost))
    if not opportunities:return dict(status='INSUFFICIENT_ALIGNED_OPPORTUNITIES',opportunities=[])
    request_costs=[]
    for rid in sorted({o['request_id'] for o in opportunities}):
        rows=[o for o in opportunities if o['request_id']==rid]
        value={f'fixed_{k}':np.mean([o['cost'][k]['target'] for o in rows]) for k in (0,2,4,8)}
        for name in ('ep','expert','acceptance','aggregate'):
            value[name]=np.mean([o['cost'][o[f'{name}_gamma']]['target'] for o in rows])
        request_costs.append(dict(request_id=rid,opportunities=len(rows),**{k:float(v) for k,v in value.items()}))
    best_fixed=min((0,2,4,8),key=lambda k:np.mean([r[f'fixed_{k}'] for r in request_costs]))
    # Promotion compares to strongest fixed/expert/acceptance baseline. The
    # measured aggregate-cost oracle is an upper-bound diagnostic, not a
    # tautologically unbeatable method-promotion comparator.
    baseline=min((f'fixed_{best_fixed}','expert','acceptance'),key=lambda k:np.mean([r[k] for r in request_costs]))
    gain=[100*(r[baseline]-r['ep'])/r[baseline] for r in request_costs]
    return dict(status='OFFLINE_SAME_PREFIX_AVAILABLE_OPPORTUNITY_DIAGNOSTIC',
        gamma0='AR one target row, not one draft proposal',opportunities=opportunities,request_costs=request_costs,
        common_opportunity_count=len(opportunities),request_count=len(request_costs),best_fixed_gamma=best_fixed,
        strongest_target_side_baseline=baseline,target_side_gain_pct=bootstrap_mean(gain),
        target_gain_vs_fixed_pct=bootstrap_mean([100*(r[f'fixed_{best_fixed}']-r['ep'])/r[f'fixed_{best_fixed}'] for r in request_costs]),
        target_gain_vs_expert_pct=bootstrap_mean([100*(r['expert']-r['ep'])/r['expert'] for r in request_costs]),
        target_gain_vs_aggregate_oracle_pct=bootstrap_mean([100*(r['aggregate']-r['ep'])/r['aggregate'] for r in request_costs]),
        e2e_gain=None,scope='measured target-forward per progress on available common prefixes; not policy rollout; no E2E oracle',
        warning='trace features for clean pass; limited aligned opportunities; per-width native draft prefixes checked, not assumed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace-version',default='phase_a_v1')
    p.add_argument('--version',required=True);p.add_argument('runs',nargs='+');a=p.parse_args()
    root=ROOT/'artifacts'/a.version
    if root.exists():raise FileExistsError(root)
    trace,old_parity=traced_table(a.trace_version)
    reference={r['request_id']:r['tokens'] for r in old_parity if r['gamma']==0}
    records=[];parity=[]
    for name in a.runs:
        path=ROOT/'raw'/name
        assert json.loads((path/'status.json').read_text())['ok']
        r,_,q=extract(path);records+=r;parity+=q
    clean=pd.DataFrame(records)
    assert all(r['tokens']==reference[r['request_id']] for r in parity), 'clean output diverges from trace/AR'
    assert all(r['trace_sampler_reconstruction_equal'] for r in parity)
    route_cols=[k for k in trace.columns if k not in clean.columns]+['draft_token_ids']
    feature=trace[KEY+route_cols].rename(columns={'draft_token_ids':'trace_draft_token_ids'})
    joined=clean.merge(feature,on=KEY,how='left',validate='many_to_one',indicator=True)
    unmatched=joined[joined['_merge']!='both'].to_dict('records')
    if unmatched:raise ValueError(f'{len(unmatched)} clean invocations lack exact traced alternatives')
    assert all(list(a)==list(b) for a,b in zip(joined.draft_token_ids,joined.trace_draft_token_ids)), 'draft IDs differ'
    joined=joined.drop(columns=['_merge','trace_draft_token_ids'])
    root.mkdir(parents=True)
    joined.to_parquet(root/'verification_metrics.parquet',index=False)
    write(root/'parity.json',parity)
    write(root/'alignment.json',dict(output_equal=True,sampler_equal=True,keys=KEY,draft_ids_equal=True,
        clean_invocations=len(joined),unmatched=len(unmatched),route_features_source=a.trace_version,
        limitation='clean routes not captured; identical canonical inputs/drafts, route repeatability assessed separately'))
    models=cost_models(joined)
    write(root/'cost_model_results.json',models)
    write(root/'trace_recomputed_cost_model_results.json',cost_models(trace))
    write(root/'width_controlled_cost_model_results.json',cost_models(joined,width_fixed_effects=True))
    write(root/'matched_pair_analysis.json',matched_pairs(joined))
    write(root/'depth_disagreement.json',depth_disagreement(joined))
    gate=dict(mae=models['ep_mae_reduction_pct']>=15,r2=models['ep_r2_increment']>=.10)
    disagree=depth_disagreement(joined)['disagreement_fraction']
    gate['depth_disagreement']=disagree is not None and disagree>=.25
    pairs=matched_pairs(joined)
    gate['matched_pairs']=pairs['n']>=10 and pairs['median_ep_delta_pct']>=20
    write(root/'missing_cost_gate.json',dict(criteria=gate,pass_any=any(gate.values()),
        evidence='clean measured target labels + independently traced aligned routes; request-grouped CV'))
    if any(gate.values()):write(root/'oracle_policy_results.json',same_prefix_oracle(joined))
    else:write(root/'oracle_policy_results.json',dict(status='NOT_RUN_GATE_FAILED',e2e_gain=None))
    summaries=[];request_metrics=[]
    for (gamma,rid),g in joined.groupby(['gamma','request_id']):
        walls=[r for r in parity if r['gamma']==gamma and r['request_id']==rid]
        assert all(r['actual_request_completion_ms'] is not None for r in walls)
        request_metrics.append(dict(gamma=int(gamma),request_id=rid,
            ep_ms_per_committed=float(g.ep_comm_ms.sum()/g.committed_progress.sum()),
            target_ms_per_committed=float(g.target_forward_ms.sum()/g.committed_progress.sum()),
            e2e_ms_per_token=sum(r['actual_request_completion_ms'] for r in walls)/sum(len(r['tokens']) for r in walls)))
    for gamma,g in joined.groupby('gamma'):
        progress=g.committed_progress.sum();wall=[r for r in parity if r['gamma']==gamma]
        drafted=g.drafted_tokens.sum();rq=[r for r in request_metrics if r['gamma']==gamma]
        summary=dict(gamma=int(gamma),requests=len({r['request_id'] for r in wall}),repeated_waves=len(wall),verifications=len(g),
            acceptance=float(g.raw_accepted_draft_tokens.sum()/drafted) if drafted else None,
            committed_per_verify=float(progress/len(g)),verified_rows_per_verify=float(g.verified_rows.mean()),
            ep_ms_per_verify=float(g.ep_comm_ms.mean()),dispatch_ms_per_verify=float(g.dispatch_ms.mean()),
            expert_ms_per_verify=float(g.expert_ms.mean()),combine_ms_per_verify=float(g.combine_ms.mean()),
            router_selection_ms_per_verify=float(g.router_selection_ms.mean()),
            ep_ms_per_committed=float(g.ep_comm_ms.sum()/progress),target_ms_per_committed=float(g.target_forward_ms.sum()/progress),
            e2e_ms_per_token=sum(r['actual_request_completion_ms'] for r in wall)/sum(len(r['tokens']) for r in wall),
            ep_target_share=float(g.ep_comm_ms.sum()/g.target_forward_ms.sum()),
            mean_unique_experts=float(g.unique_experts.mean()),mean_remote_paths=float(g.remote_routed_paths.mean()),
            mean_remote_dedup_rows=float(g.remote_destination_dedup_rows.mean()),
            mean_fanout=float(g.rank_fanout.mean()),mean_max_rank_bytes=float(g.max_rank_bytes.mean()),
            rejected_logical_ep_byte_fraction=float(g.rejected_logical_dispatch_bytes.sum()/g.logical_dispatch_bytes.sum()),
            logical_ep_bytes_per_committed=float(48*(g.logical_dispatch_bytes+g.logical_combine_bytes).sum()/progress),
            logical_remote_ep_bytes_per_committed=float(96*g.logical_remote_dispatch_bytes.sum()/progress),
            target_invocations_per_committed=float(len(g)/progress),
            ep_ms_per_verified_row=float(g.ep_comm_ms.sum()/g.verified_rows.sum()),
            expert_ms_per_verified_row=float(g.expert_ms.sum()/g.verified_rows.sum()),
            ep_request_ci=bootstrap_mean([r['ep_ms_per_committed'] for r in rq]),
            target_request_ci=bootstrap_mean([r['target_ms_per_committed'] for r in rq]),
            e2e_request_ci=bootstrap_mean([r['e2e_ms_per_token'] for r in rq]),
            evidence='CLEAN_NATIVE_TRUE_EP4; independent trace features; repeat-aggregated request bootstrap')
        summaries.append(summary)
    write(root/'gamma_summary.json',summaries)
    write(root/'request_metrics.json',request_metrics)
    print(json.dumps(dict(summary=summaries,gate=gate,models={k:v for k,v in models.items() if k not in ('M0','M1','M2','folds')}),indent=2))


if __name__=='__main__':main()
