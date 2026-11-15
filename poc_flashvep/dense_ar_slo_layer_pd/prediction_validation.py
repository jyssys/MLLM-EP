"""Validate past online predictions against later retained Host observations."""
import collections
from .common import *
from .analyze import records,percentile


def error_summary(values):
    if not values:return dict(samples=0,evidence='NOT_MEASURED')
    absolute=[abs(v) for v in values]
    return dict(samples=len(values),MAE_ms=sum(absolute)/len(values),
        p95_absolute_error_ms=percentile(absolute,.95),p99_absolute_error_ms=percentile(absolute,.99),
        signed_mean_observed_minus_predicted_ms=sum(values)/len(values),
        underprediction_frequency=sum(v>0 for v in values)/len(values),
        empirical_upper_coverage=sum(v<=0 for v in values)/len(values),
        maximum_underprediction_ms=max(0.,max(values)))


def analyze(name):
    folder=ROOT/'raw'/name
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    rows={f"q{r['sample_id']}":r for r in records(folder/'requests.jsonl')}
    source={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    groups=records(folder/'online_decisions.jsonl');cuts=collections.Counter();reasons=collections.Counter()
    errors=collections.defaultdict(list);per_cut=collections.defaultdict(lambda:collections.defaultdict(list))
    overhead=[];candidates=collections.defaultdict(lambda:collections.Counter());exposed=[]
    for decision in groups:
        state=decision['observation'];choice=decision['choice'];layer=choice['layers']
        ids=list(state.get('group_request_ids',()))
        if not ids:ids=[rid for rid,r in rows.items() if r.get('dynamic_reservation')==decision['generation']]
        cuts[layer]+=state['group_size'];reasons[choice['reason']]+=1
        if 'policy_decision_host_ms' in decision:overhead.append(decision['policy_decision_host_ms'])
        for c in choice['candidates']:
            tally=candidates[c['layers']];tally['evaluated']+=1;tally['accepted' if c['accepted'] else 'rejected']+=1
            for reason in set(c['rejection_reasons']):tally['gate:'+reason]+=1
        known=decision['active_D_requests']+decision['pending_D_requests']
        actual=[rows[rid]['finish'] for rid in known if rid in rows]
        if actual:
            observed=max(actual)
            errors['known_D_retirement_nominal'].append(1000*(observed-state.get('D_available_nominal',state['D_available'])))
            errors['known_D_retirement_base_bound'].append(1000*(observed-state['D_available']))
        selected=next((c for c in choice['candidates'] if c['layers']==layer),None)
        if not layer or selected is None:continue
        frontier=selected.get('frontier_plan');nominal=selected.get('nominal_frontier_plan',frontier)
        if not frontier:continue
        present=[rid for rid in ids if rid in rows and rid in source]
        if len(present)==len(ids) and present:
            errors['P_group_source_ready'].append(1000*(max(source[rid] for rid in present)-frontier['source_ready']))
            errors['group_first_output_nominal'].append(1000*(max(rows[rid]['first_token'] for rid in present)-nominal['group_first_output']))
            errors['group_first_output_upper'].append(1000*(max(rows[rid]['first_token'] for rid in present)-
                frontier['group_first_output']-selected['safety_margin_ms']/1000))
        for index,rid in enumerate(ids):
            if rid not in rows or rid not in source:continue
            observed=rows[rid]['first_token']
            upper=frontier['first_outputs'][str(index)] + selected['safety_margin_ms']/1000
            predicted=nominal['first_outputs'][str(index)]
            source_prediction=frontier['source_frontiers'][str(index)]
            eligible=max(source[r] for r in present) if cfg.get('D_wait_for_source_macro') and present else source[rid]
            values=dict(first_output_nominal=1000*(observed-predicted),
                first_output_upper=1000*(observed-upper),
                D_eligible_source_frontier=1000*(eligible-source_prediction))
            for key,value in values.items():errors[key].append(value);per_cut[layer][key].append(value)
            exposed.append(dict(request_id=rid,layers=layer,decision=decision['time'],
                predicted_first_output_nominal=predicted,predicted_first_output_upper=upper,
                actual_first_output=observed,actual_source_ready=source[rid],
                actual_D_eligible_source_frontier=eligible,
                predicted_source_ready=source_prediction,own_deadline=state['own_TTFT_deadlines'][index]))
    return dict(name=name,policy=cfg.get('policy'),group_decisions=len(groups),selected_requests=dict(cuts),
        group_reasons=dict(reasons),candidate_gate_counts={k:dict(v) for k,v in candidates.items()},
        errors={k:error_summary(v) for k,v in errors.items()},
        by_cut={k:{key:error_summary(v) for key,v in value.items()} for k,value in per_cut.items()},
        policy_overhead_ms=dict(samples=len(overhead),p50=percentile(overhead,.5),p95=percentile(overhead,.95),maximum=max(overhead,default=None)),
        request_predictions=exposed,evidence='TRACE_ANALYSIS',
        analysis_version=2,
        scope='Validate predictions written before the real prefix submission. Group Source-ready uses the last real Source cut, not an earlier individual cut under a whole-macro guard. Observed times are Host notifications and coordinator output delivery. They are not CUDA kernel intervals. Error samples within a run are dependent. Gate counts overlap and are not request counts. Small-sample upper coverage is not WCET.')


def main():
    values=[]
    for p in sorted((ROOT/'raw').glob('*/online_decisions.jsonl')):
        receipt=p.parent/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        values.append(analyze(p.parent.name))
    write(ROOT/'analysis/online_prediction_validation_v2.json',dict(runs=values,evidence='TRACE_ANALYSIS',
        prior_analysis_preserved='online_prediction_validation.json',
        clarification='Version2 compares the whole-macro modeled source frontier with the actual whole-macro readiness. Version1 compared an earlier individual P cut with the later intentional guard frontier. No observed serving latency or online decision changed.'))
    print('PREDICTION_RUNS',len(values),flush=True)


if __name__=='__main__':main()
