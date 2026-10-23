"""Describe frozen feature surfaces and completed execution records.

This module does not tune a policy or modify a registered plan.
"""
import json
import math
from collections import Counter,defaultdict
import numpy as np
from .common import ROOT,PRIOR,write
from .objective_features import prepare_items,contextualize_offline
from .objective_families import choose,scores
from ..ouro_budget_joint_cg.optimizer import enumerate_candidates
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup,csv_file

def mi(pairs):
    """Empirical plug-in mutual information in bits, without causal meaning."""
    if not pairs:return None
    xy=Counter(pairs);x=Counter(a for a,b in pairs);y=Counter(b for a,b in pairs);n=len(pairs)
    return sum(count/n*math.log2(count*n/(x[a]*y[b])) for (a,b),count in xy.items())

def offline():
    lookup=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
    norm=json.loads((ROOT/'objective_normalization_v1.json').read_text())
    states={}
    for folder in sorted((PRIOR/'raw').glob('cb_p3_budget_*')):
        for file in sorted(folder.glob('J1_T*.json')):
            data=json.loads(file.read_text())
            for index,d in enumerate(data['optimizer_decisions']):
                s=d['state']
                if s['kind']!='NEW_CHUNK' or not s['request_event_ready']:continue
                key=(folder.name,data['trial'],s['id'],s['start'],d['budget_ms'])
                states.setdefault(key,(data,index,contextualize_offline(s,data,d['host_s']),d['budget_ms']))
    interactions=[];endpoints=[]
    for key,(cohort,index,state,B) in states.items():
        items=prepare_items(enumerate_candidates(state,B,lookup),state,norm,lookup)
        if not items:continue
        by={(c.C,c.G):(c,f) for c,f in items};cs=sorted({c.C for c,f in items})
        e,_=choose('O2',items,endpoint='E_ONLY');r,_=choose('O2',items,endpoint='R_ONLY')
        base=dict(run_id=key[0],trial=key[1],request=key[2],span_start=key[3],decision=index,
            P=cohort['P'],B_ms=B,decode_rows=len(state.get('issued_decode',[])),
            decode_age_ms=state.get('decode_age_max_ms',0),prompt_age_ms=state.get('prompt_age_ms',0),
            phase=state['phase'],legal_actual_C=len(cs),
            evidence='OFFLINE_ESTIMATE_CONSUMED_TRAIN_FEATURE_SURFACE_NOT_GPU_COUNTERFACTUAL')
        for family,weight in [('O0',1),('O2',8),('O4',1),('O6',2),('O10',8)]:
            dual,_=choose(family,items,weight=weight)
            endpoints.append(dict(base,objective=family,weight=weight,E_C=e.C,E_G=e.G,R_C=r.C,R_G=r.G,
                dual_C=dual.C,dual_G=dual.G,C_matches_E=dual.C==e.C,G_matches_R=dual.G==r.G,
                C_differs_E=dual.C!=e.C,G_differs_E=dual.G!=e.G,
                pareto_actions=sum(not any(g['E']>=f['E'] and g['R']<=f['R'] and
                    (g['E']>f['E'] or g['R']<f['R']) for _,g in items) for _,f in items)))
        if len(cs)<2:continue
        low,high=cs[0],cs[-1]
        gs=sorted(g for c,g in by if c==low and (high,g) in by)
        if len(gs)<2:continue
        gl,gh=gs[0],gs[-1]
        corners=[by[(low,gl)],by[(high,gl)],by[(low,gh)],by[(high,gh)]]
        a,b,c,d=[f for _,f in corners]
        row=dict(base,C_low=low,C_high=high,G_low=gl,G_high=gh,
            delta_C_E=b['E']-a['E'],delta_C_R=b['R']-a['R'],
            delta_G_E=c['E']-a['E'],delta_G_R=c['R']-a['R'],
            interaction_E=d['E']-c['E']-b['E']+a['E'],
            interaction_R=d['R']-c['R']-b['R']+a['R'],
            all_shapes_known=all(candidate.known for candidate,_ in corners))
        for family,weight in [('O2',8),('O6',2),('O10',8)]:
            u=[rank[0] for rank in scores(family,corners,weight=weight)]
            row[family+'_interaction_utility']=u[3]-u[2]-u[1]+u[0]
        interactions.append(row)
    csv_file(ROOT/'objective_interactions.csv',interactions)
    csv_file(ROOT/'objective_endpoint_diagnostics.csv',endpoints)
    return len(interactions),len(endpoints)

def observed():
    conditionals=[];overhead=[];cohort_matches=[];slo=[];observer=[]
    lock=json.loads((ROOT/'slo_lock.json').read_text()) if (ROOT/'slo_lock.json').exists() else None
    for folder in sorted((ROOT/'raw').glob('dual_p[23]_*')):
        if not (folder/'completed.json').exists():continue
        values={}
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or 'scenario' not in data or 'label' not in data:continue
            values[(data['scenario']['id'],data['label'])]=data
            ds=data.get('optimizer_decisions',[]);fresh=[d for d in ds if d['state']['kind']=='NEW_CHUNK']
            for d in fresh:
                s=d['state'];c=d['selected'];fixed=data['cell'].get('fixed_C',2048)
                actual_fixed=min(fixed,s['prompt_remaining'],s['token_budget'])
                alt=next((v for v in d['candidates'] if v['C']==actual_fixed and v['G']==c['G']),None)
                chosen_features=next(v['features'] for v in d['candidates'] if v['C']==c['C'] and v['G']==c['G'])
                conditionals.append(dict(run_id=data['run_id'],scenario=data['scenario']['id'],
                    split=data['split'],lane=data['lane'],workload=data['workload'],policy=data['label'],
                    host_s=d['host_s'],request_id=s['id'],actual_C=c['C'],planned_G=c['G'],prompt_remaining=s['prompt_remaining'],
                    token_budget=s['token_budget'],issued_depth=s['depth'],
                    actual_C_candidates=json.dumps(sorted({v['C'] for v in d['candidates']})),
                    span_start=s['start'],phase=s['phase'],decode_rows=len(s.get('issued_decode',[])),
                    decode_age_ms=s.get('decode_age_max_ms',0),prompt_age_ms=s.get('prompt_age_ms',0),
                    multiple_actual_C=len({v['C'] for v in d['candidates']})>1,
                    differs_fixed_C=c['C']!=actual_fixed,
                    same_G_fixed_alternative_present=alt is not None,
                    same_G_fixed_alternative_known=None if alt is None else alt['known'],
                    delta_E_vs_fixed_same_G=None if alt is None else chosen_features['E']-alt['features']['E'],
                    delta_R_vs_fixed_same_G=None if alt is None else chosen_features['R']-alt['features']['R'],
                    selected_shape_known=c['known'],fallback='FALLBACK' in d['reason'],
                    evidence='LIVE_MEASURED_HOST_SELECTOR_CHOICE_NOT_GPU_FINISH_TIME'))
            ns=[d['selector_ns']/1e6 for d in ds];admission=data.get('admission_cpu') or {}
            wall_ms=1000*(data['end']-data['start'])
            overhead.append(dict(run_id=data['run_id'],scenario=data['scenario']['id'],policy=data['label'],
                split=data['split'],lane=data['lane'],workload=data['workload'],rate=data['scenario'].get('rate'),calls=len(ns),
                selector_P50_ms=None if not ns else float(np.percentile(ns,50)),
                selector_P95_ms=None if not ns else float(np.percentile(ns,95)),
                selector_P99_ms=None if not ns else float(np.percentile(ns,99)),
                selector_total_ms=sum(ns),selector_wall_ratio=sum(ns)/wall_ms,
                admission_total_ms=admission.get('total_ns',0)/1e6,
                admission_checks=admission.get('checks',0),drained_wall_ms=wall_ms,
                admission_wall_ratio=admission.get('total_ns',0)/1e6/wall_ms,
                evidence='LIVE_MEASURED_CPU_TIMERS_NOT_REMOVABLE_GPU_CRITICAL_PATH_SHARE'))
            if lock and data['split']=='FINAL':
                from .analyze_serving import completed_response
                for context,targets in lock['multiplier_tiers'].items():
                    for factor,target_ms in zip((5,10,25),targets):
                        reqs=data['requests'];passed=[r for r in reqs if completed_response(r) and r['TTFT_s']<=5 and
                            all(1000*g<=target_ms for g in r['ITLs_s'])]
                        slo.append(dict(run_id=data['run_id'],scenario=data['scenario']['id'],policy=data['label'],
                            lane=data['lane'],workload=data['workload'],rate=data['scenario'].get('rate'),
                            reference_P=context,multiplier=factor,reference_ms=lock['decode_reference'][context],
                            TBT_ms=target_ms,TTFT_ms=5000,offered=len(reqs),passed=len(passed),
                            attainment_pct=100*len(passed)/len(reqs),drain_goodput_rps=len(passed)/(data['end']-data['start']),
                            scope='POSTPROCESS_FROZEN_TIERS_ON_SAME_LIVE_OUTPUT_CLOCKS_NOT_CONTROLLER_RETUNING'))
        for (scenario,label),left in values.items():
            controls=[]
            if label.endswith('_J'):
                family=label[:-2];controls=[family+'_G',family+'_G1024',family+'_C','BEST_FIXED','MIXED_STOCK','MIXED_FAIR']
            if label.endswith('_OBS'):controls=[label.removesuffix('_OBS')+'_NOOBS']
            for control in controls:
                right=values.get((scenario,control))
                if right is None:continue
                a={r['id']:r for r in left['requests']};b={r['id']:r for r in right['requests']}
                exact=[k for k in a if a[k]['output_ids']==b[k]['output_ids']]
                from .analyze_serving import actual_plan
                lp,rp=actual_plan(left),actual_plan(right)
                row=dict(run_id=left['run_id'],scenario=scenario,split=left['split'],lane=left['lane'],
                    workload=left['workload'],rate=left['scenario'].get('rate'),candidate=label,reference=control,
                    requests=len(a),token_exact=len(exact),all_request_tokens_exact=len(exact)==len(a),
                    count_equal=sum(len(a[k]['output_ids'])==len(b[k]['output_ids']) for k in a),
                    plan_exact=sum(lp[k]==rp[k] for k in a),
                    complete_status_equal=sum(str(a[k]['finish_reason'])==str(b[k]['finish_reason']) for k in a),
                    evidence='LIVE_MEASURED_OUTPUT_PAIRING. Equal-output subset is selected after execution, not randomized evidence.')
                cohort_matches.append(row)
                if label.endswith('_OBS'):
                    divergences=[]
                    for k in a:
                        aa,bb=a[k]['output_ids'],b[k]['output_ids']
                        mismatch=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),None)
                        if mismatch is None and len(aa)!=len(bb):mismatch=min(len(aa),len(bb))
                        if mismatch is not None:divergences.append(dict(id=k,first_divergent_index=mismatch))
                    observer.append(dict(row,first_divergences=json.dumps(divergences)))
    csv_file(ROOT/'live_action_conditions.csv.gz',conditionals)
    selected=[r for r in conditionals if r['split']=='FINAL' and r['policy']=='O6_J']
    changed=[r for r in selected if r['differs_fixed_C']]
    csv_file(ROOT/'Joint_C_change_decomposition.csv',changed)
    write(ROOT/'Joint_C_change_summary.json',dict(fresh_choices=len(selected),real_C_changes=len(changed),
        selected_C_histogram=dict(Counter(r['actual_C'] for r in changed)),
        changed_remaining_le_512=sum(r['prompt_remaining']<=512 for r in changed),
        changed_remaining_513_to_2048=sum(512<r['prompt_remaining']<=2048 for r in changed),
        changed_remaining_gt_2048=sum(r['prompt_remaining']>2048 for r in changed),
        changed_with_no_decode=sum(r['decode_rows']==0 for r in changed),
        all_changed_selected_shapes_known=all(r['selected_shape_known'] for r in changed),
        all_changed_fixed_same_G_alternatives_unknown=all(r['same_G_fixed_alternative_known'] is False for r in changed),
        changed_E_above_fixed_same_G=sum(r['delta_E_vs_fixed_same_G'] is not None and r['delta_E_vs_fixed_same_G']>0 for r in changed),
        changed_R_below_fixed_same_G=sum(r['delta_R_vs_fixed_same_G'] is not None and r['delta_R_vs_fixed_same_G']<0 for r in changed),
        changed_fallbacks=sum(r['fallback'] for r in changed),
        all_fresh_G_histogram=dict(Counter(r['planned_G'] for r in selected)),
        scope='Actual causal selector choices. Known selected shapes do not prove the unchosen clipped alternative is physically slower.'))
    csv_file(ROOT/'optimizer_overhead.csv',overhead)
    csv_file(ROOT/'whole_cohort_output_parity.csv',cohort_matches)
    csv_file(ROOT/'observer_execution_diagnostics.csv',observer)
    csv_file(ROOT/'frozen_multiplier_SLO_results.csv',slo)
    groups=defaultdict(list)
    for r in conditionals:groups[(r['split'],r['lane'],r['workload'],r['policy'])].append(r)
    info=[]
    for key,rs in groups.items():
        states=[(min(3,r['decode_rows']//2),min(3,int(r['decode_age_ms']//100)),
            min(3,int(r['prompt_age_ms']//2000)),min(3,r['prompt_remaining']//2048)) for r in rs]
        info.append(dict(zip(('split','lane','workload','policy'),key),fresh_calls=len(rs),
            MI_state_C_bits=mi(list(zip(states,[r['actual_C'] for r in rs]))),
            MI_state_G_bits=mi(list(zip(states,[r['planned_G'] for r in rs]))),
            state_bin_rule='decode count //2, delivered gap age //100ms, prompt age //2s, remaining prompt //2048; each clipped to3',
            scope='Descriptive plug-in MI. Sparse state bins and clipping can inflate apparent dependence. No causal inference.'))
    csv_file(ROOT/'state_action_mutual_information.csv',info)
    return len(conditionals),len(overhead)

def main():
    interactions,endpoints=offline();actions,cohorts=observed()
    write(ROOT/'mechanism_analysis_status.json',dict(interactions=interactions,endpoint_rows=endpoints,
        observed_fresh_choices=actions,completed_cohorts=cohorts,changes_registered_policy=False))
    print('Mechanism diagnostics',interactions,endpoints,actions,cohorts)

if __name__=='__main__':main()
