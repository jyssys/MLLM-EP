"""Freeze calibration-only operating point, then request-clustered held-out gates."""
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_block_local_replica.metrics import bootstrap_mean
from poc_dllm_request_persistent_replica.statistics import bootstrap_ratio

METRICS=('baseline_amp','residual_amp','dc_reduction','byte_roi','peak_memory_gib',
    'max_rank_load_increase','worst_stage_load_increase','stage_critical_load_increase',
    'copy_serial_expert_ms','copy_endpoint_parallel_est_ms','future_blocks','future_iterations')


def main():
    tables={lane:pd.read_csv(ROOT/'results'/('simulation_'+lane+'_v2')/'policy_budget.csv') for lane in ('short','long')}
    reference=[]
    for lane,d in tables.items():
        for row in d.drop_duplicates('request').to_dict('records'):
            reference.append(dict(lane=lane,request=row['request'],domain=row['domain'],split=row['split'],
                policy='no_replication_live_only',post_bytes=row['live_bytes'],B_one_shot=row['B_one_shot'],
                residual_amp=row['baseline_amp'],copy_bytes=0,extra_memory=0,
                status='PERFECT_DEAD_TOKEN_ELIMINATION_STRUCTURAL_REFERENCE_NOT_MEASURED_RUNTIME'))
    if not (ROOT/'results/no_replication_reference.csv').exists():pd.DataFrame(reference).to_csv(ROOT/'results/no_replication_reference.csv',index=False)
    cost=pd.read_csv(ROOT/'results/copy_cost/request_copy_roi.csv')
    for lane,d in list(tables.items()):
        lane_cost=cost[cost.lane==lane]
        tables[lane]=d.merge(lane_cost[['request','policy','budget_gib','load_guard','copy_serial_expert_ms','copy_endpoint_parallel_est_ms',
            'projected_byte_break_even_remaining_fraction']],on=['request','policy','budget_gib','load_guard'],validate='one_to_one')
    cal=tables['long'][tables['long'].split=='calibration']
    global_controls={};candidate_scores=[]
    for cap in (.10,.15,.20,-1.):
        for budget in BUDGETS:
            global_rows=cal[(cal.load_guard==cap)&(cal.budget_gib==budget)&cal.policy.str.startswith('global')]
            control=global_rows.groupby('policy').post_bytes.mean().idxmin()
            global_controls[f'{budget:g}@{cap:g}']=control
            g=global_rows[global_rows.policy==control].set_index('request').post_bytes
            if cap!=.10 or budget>2.:continue
            for policy in ('first_block','first2_blocks'):
                r=cal[(cal.load_guard==cap)&(cal.budget_gib==budget)&(cal.policy==policy)].set_index('request').post_bytes
                gains=((g-r)/g).dropna()
                candidate_scores.append(dict(policy=policy,budget_gib=budget,load_guard=cap,global_control=control,
                                             mean_relative_gain=float(gains.mean())))
    chosen=max(candidate_scores,key=lambda x:x['mean_relative_gain'])
    freeze=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),chosen=chosen,global_controls=global_controls,
        candidates=candidate_scores,selection='calibration8 only, best mean relative gain over stronger calibration-selected Global; cap10%,budget<=2GiB',
        heldout_results_not_used_for_selection=True,thresholds_unchanged=True,
        input_calibration_ids=cal.request.unique().tolist(),generation_contract_sha256=sha(ROOT/'results/frozen_manifest.json'))
    freeze_path=ROOT/'results/frozen_operating_point.json'
    if freeze_path.exists():
        old=json.loads(freeze_path.read_text())
        assert old['chosen']==chosen and old['global_controls']==global_controls
    else:new_json(freeze_path,freeze)
    # All subsequent statistics are untouched test evaluation, not retuning.
    stats={};contrasts={};summary_rows=[]
    for lane,df in tables.items():
        groups={'calibration':df[df.split=='calibration'],'heldout_all':df[df.split=='heldout']}
        if lane=='long':groups.update(heldout_coding=df[(df.split=='heldout')&(df.domain=='coding')],
                                      heldout_reasoning=df[(df.split=='heldout')&(df.domain=='reasoning')])
        for cohort,d in groups.items():
            key=lane+'_'+cohort;stats[key]={};contrasts[key]={}
            for (policy,budget,cap),r in d.groupby(['policy','budget_gib','load_guard']):
                name=f'{policy}@{budget:g}@{cap:g}'
                vals={m:bootstrap_mean(r[m].dropna().tolist()) for m in METRICS if len(r[m].dropna())}
                vals['absolute_worst_stage_load_increase']=float(r.worst_stage_load_increase.max())
                vals['fraction_requests_with_actual_worst_load_le10']=float((r.worst_stage_load_increase<=.10+1e-8).mean())
                stats[key][name]=vals
                summary_rows.append(dict(cohort=key,policy=policy,budget_gib=budget,load_guard=cap,
                    **{m:vals[m]['mean'] for m in METRICS if m in vals}))
            for cap in (.10,.15,.20,-1.):
                for budget in BUDGETS:
                    local=d[(d.load_guard==cap)&(d.budget_gib==budget)]
                    control=global_controls[f'{budget:g}@{cap:g}']
                    g=local[local.policy==control].set_index('request')
                    oracle_name='oracle' if cap==-1. else 'oracle_load'
                    oracle=local[local.policy==oracle_name].set_index('request')
                    # A Global set can floor an oracle reference ONLY when
                    # it satisfies that oracle's actual future-load constraint.
                    # A history-capped but future-unsafe set is not feasible.
                    global_feasible=(g.worst_stage_load_increase<=cap+1e-8) if cap>=0 else pd.Series(True,index=g.index)
                    oracle_bytes=oracle.post_bytes.copy()
                    oracle_bytes.loc[global_feasible]=np.minimum(oracle_bytes.loc[global_feasible],g.post_bytes.loc[global_feasible])
                    for policy in ('first_block','first2_blocks','first_frequency','first2_frequency'):
                        r=local[local.policy==policy].set_index('request').reindex(g.index)
                        relative=(g.post_bytes-r.post_bytes)/g.post_bytes
                        recovery=(g.post_bytes-r.post_bytes)/(g.post_bytes-oracle_bytes).replace(0,np.nan)
                        name=f'{policy}-global@{budget:g}@{cap:g}'
                        contrasts[key][name]=dict(global_control=control,
                            relative_gain=bootstrap_mean(relative.dropna().tolist()),
                            aggregate_relative_gain=bootstrap_ratio(g.post_bytes-r.post_bytes,g.post_bytes),
                            absolute_byte_gain=bootstrap_mean((g.post_bytes-r.post_bytes).dropna().tolist()),
                            persistent_oracle_recovery=bootstrap_ratio(g.post_bytes-r.post_bytes,g.post_bytes-oracle_bytes),
                            macro_persistent_oracle_recovery=bootstrap_mean(recovery.dropna().tolist()) if recovery.notna().any() else None,
                            oracle_gain_beyond_global=bootstrap_mean(((g.post_bytes-oracle_bytes)/g.post_bytes).dropna().tolist()),
                            global_and_candidate_actual_load_matched=bool(g.worst_stage_load_increase.max()<=cap+1e-8 and r.worst_stage_load_increase.max()<=cap+1e-8) if cap>=0 else False)
                        for direct_control in ('global_frequency','global_destination'):
                            direct=local[local.policy==direct_control].set_index('request').reindex(r.index)
                            direct_gain=(direct.post_bytes-r.post_bytes)/direct.post_bytes
                            contrasts[key][f'{policy}-{direct_control}@{budget:g}@{cap:g}']=dict(
                                global_control=direct_control,relative_gain=bootstrap_mean(direct_gain.dropna().tolist()))
    policy=chosen['policy'];budget=chosen['budget_gib'];cap=chosen['load_guard']
    name=f'{policy}@{budget:g}@{cap:g}';contrastname=f'{policy}-global@{budget:g}@{cap:g}'
    target=stats['long_heldout_all'][name];contrast=contrasts['long_heldout_all'][contrastname]
    blocks=pd.read_csv(ROOT/'results/simulation_long_v2/block_policy.csv')
    heldids=tables['long'].loc[tables['long'].split=='heldout','request'].unique()
    b=blocks[(blocks.request.isin(heldids))&(blocks.budget_gib==budget)&(blocks.load_guard==cap)]
    c=b[b.policy==policy].merge(b[b.policy==chosen['global_control']],on=['request','block'],suffixes=('_r','_g'),validate='one_to_one')
    c['relative_gain_over_global']=(c.post_bytes_g-c.post_bytes_r)/c.post_bytes_g
    profile=1 if policy=='first_block' else 2
    c['future_index']=c.block-profile+1
    horizon=[]
    for index,r in c[c.future_index>0].groupby('future_index'):
        horizon.append(dict(future_index=int(index),**bootstrap_mean(r.relative_gain_over_global.dropna().tolist())))
    positive=sum(x['mean']>0 for x in horizon)
    first4=horizon[:4]
    recovery=contrast['persistent_oracle_recovery']
    selected=tables['long'][(tables['long'].split=='heldout')&(tables['long'].policy==policy)&
         (tables['long'].budget_gib==budget)&(tables['long'].load_guard==cap)]
    observed_break=selected.byte_break_even_iteration.notna()
    apparent_increment=contrast['relative_gain']['mean']>=.10 and contrast['relative_gain']['ci95'][0]>0
    matched_load=contrast['global_and_candidate_actual_load_matched']
    gates=dict(apparent_byte_increment=apparent_increment,
        request_beats_global=apparent_increment and matched_load,
        sustained_four_future_blocks=len(first4)==4 and all(x['mean']>0 for x in first4),
        copy_roi_validated=target['byte_roi']['mean']>=2.,timing_roi='UNRESOLVED_NO_REPLICA_TIMING',
        early_break_even=bool(observed_break.all() and selected.projected_byte_break_even_remaining_fraction.median()<=.50),
        actual_worst_load_le10=target['absolute_worst_stage_load_increase']<=.10+1e-8,
        matched_actual_load=contrast['global_and_candidate_actual_load_matched'],
        recovery_beyond_global_ge50=bool(recovery and recovery['mean']>=.50 and matched_load))
    passed=all(v is True for k,v in gates.items() if k!='timing_roi')
    # Time ROI is not fabricated. Structural byte ROI is the predeclared
    # fallback only; no runtime promotion unless ALL central premises pass.
    verdict='GO' if passed else 'NO-GO'
    result=dict(verdict=verdict,stage_b_promoted=passed,selected=chosen,gates=gates,
        selected_heldout=target,selected_contrast=contrast,
        selected_coding=stats['long_heldout_coding'][name],selected_coding_contrast=contrasts['long_heldout_coding'][contrastname],
        positive_future_blocks=positive,horizon=horizon,
        observed_byte_break_even_requests=int(observed_break.sum()),heldout_requests=len(selected),
        stats=stats,paired_contrasts=contrasts,oracle='future-truth eight-offer destination bundle achievable reference, not exact integer optimum; Global floors oracle ONLY where it satisfies actual future-load cap',
        runtime_gain=None,replica_accuracy=None,replica_correctness='NOT_IMPLEMENTED_STAGE_A_NO_GO' if not passed else 'PENDING',
        copy_time_roi=None,incremental_updates=0,optional_AR='NOT_RUN, not mandatory',
        close_replication_family=(not gates['request_beats_global']) or target['absolute_worst_stage_load_increase']>.20,
        validity='Apparent byte gain and apparent oracle recovery are NOT load-matched. They do not establish the primary request-vs-global novelty gate.',
        next_action='Close tested dLLM-specific expert-weight replication/movement family in this B1 BF16 source-SP4 regime: persistent byte advantage exists only with uncontrolled future hotspots and copy amortization is unvalidated. Investigate activation-side repeated live-token communication; do not build persistent runtime.' if not gates['request_beats_global'] else
                    'Do not implement runtime while copy/load/recovery gates fail; no learned-predictor rescue.')
    destination=ROOT/'results/summary.json'
    if destination.exists():
        destination=ROOT/'results/summary_v2.json'
        result['audited_correction']='Enforce the specification matched-actual-load prerequisite on request-specificity and oracle recovery; no predictor/threshold/plan/data changes. Preliminary summary.json is retained.'
    if destination.exists():
        destination=ROOT/'results/summary_v3.json'
        result['audited_correction']='Additionally exclude future-load-unsafe Global sets from constrained-oracle feasible floors. Selected point values/verdict unchanged; all plans/thresholds and previous summaries retained.'
    new_json(destination,result)
    if destination.name=='summary_v3.json':new_json(ROOT/'results/bootstrap_ci_v3.json',dict(unit='request',repeats=2000,stats=stats,paired=contrasts,horizon=horizon,
        correction='oracle floors require actual-load-feasible Global; selected primary estimates unchanged'))
    if not (ROOT/'results/bootstrap_ci.json').exists():new_json(ROOT/'results/bootstrap_ci.json',dict(unit='request',repeats=2000,stats=stats,paired=contrasts,horizon=horizon))
    if not (ROOT/'results/selected_gain_by_block.csv').exists():c.to_csv(ROOT/'results/selected_gain_by_block.csv',index=False)
    if not (ROOT/'results/policy_summary.csv').exists():pd.DataFrame(summary_rows).to_csv(ROOT/'results/policy_summary.csv',index=False)
    if not (ROOT/'results/accuracy_correctness.json').exists():new_json(ROOT/'results/accuracy_correctness.json',dict(baseline='physical token/acceptance/NFE parity and all receive-buffer ownership checks PASS',
        runtime_replica='NOT_IMPLEMENTED_STAGE_A_NO_GO' if not passed else 'PENDING',
        final_token_parity=None,acceptance_parity=None,NFE_parity=None,router_ID_weight_parity=None,
        exact_replica_tensor=None,GSM8K_exact_match=None,HumanEval_pass_at_1=None,accuracy_delta=None,
        semantics='Offline relocation preserves paths/scaling algebraically; not a measured runtime correctness/accuracy claim'))
    print('PERSISTENT_STAGE_A',verdict,chosen,gates,flush=True)


if __name__=='__main__':main()
