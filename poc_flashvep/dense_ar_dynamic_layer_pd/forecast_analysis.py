"""Compare past-only predictions with outcomes after each run ends."""
import collections,statistics
from .common import *
from .analyze import records,percentile


def errors(values):
    return dict(samples=len(values),MAE_ms=statistics.mean(abs(x) for x in values) if values else None,
        signed_mean_ms=statistics.mean(values) if values else None,
        absolute_error_p95_ms=percentile([abs(x) for x in values],.95),
        underprediction_fraction=sum(x>0 for x in values)/len(values) if values else None,
        definition='Error equals observed time minus predicted time. A positive error is an underprediction.')


def analyze(name):
    folder=ROOT/'raw'/name;requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
    decisions=records(folder/'online_decisions.jsonl');commits=records(folder/'actual_cut_commits.jsonl')
    if not requests or not decisions:return None
    commit_by_generation={r['generation']:r for r in commits}
    handoffs={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
    active_errors=[];known_errors=[];prefix_errors=[];finish_errors=[];observations=[]
    reasons=collections.Counter();candidate_reasons=collections.Counter();gate_failures=collections.Counter()
    for d in decisions:
        reasons[d['choice']['reason']]+=1
        for c in d['choice']['candidates']:
            candidate_reasons[c['reason']]+=1
            tags=c.get('rejection_reasons')
            if tags is None:
                tags=[];s=d['observation'];guard=c['safety_margin_ms']
                if c['predicted_suffix_ms']+guard>c['idle_budget_ms']:tags.append('INSUFFICIENT_IDLE_PREDICTION')
                if d['policy'] in ('DEADLINE','COST') and (s['active_decode'] or s.get('admitted_D_requests',0)):
                    if s['D_available']+guard/1000>c.get('predicted_unblocked_D_ready',c['predicted_D_ready']):
                        tags.append('DECODE_BUSY_AT_PREDICTED_HANDOFF')
                if d['policy']=='COST' and c['predicted_TTFT_saving_ms']<=0:tags.append('NO_COMPLETION_BENEFIT')
            gate_failures.update(tags)
        active=d.get('active_D_requests',[])
        if active and all(r in requests for r in active):
            actual=max(requests[r]['finish'] for r in active)
            active_errors.append(1000*(actual-d['predicted_active_D_finish']))
        known=active+d.get('pending_D_requests',[])
        if known and all(r in requests for r in known):
            actual=max(requests[r]['finish'] for r in known)
            known_errors.append(1000*(actual-d['observation']['D_available']))
        commit=commit_by_generation.get(d['generation'])
        if not commit or not commit['offloaded_layers']:continue
        ids=commit['requests'];layers=commit['offloaded_layers']
        candidate=next((c for c in d['choice']['candidates'] if c['layers']==layers),None)
        if candidate is None or not all(r in requests and r in handoffs for r in ids):continue
        observed_prefix=max(handoffs[r] for r in ids)
        observed_first=max(requests[r]['first_token'] for r in ids)
        pe=1000*(observed_prefix-candidate['predicted_P_cut_ready'])
        fe=1000*(observed_first-candidate['predicted_D_finish'])
        prefix_errors.append(pe);finish_errors.append(fe)
        observations.append(dict(time=d['time'],requests=ids,**candidate,
            observed_source_ready=observed_prefix,observed_first_token=observed_first,
            prefix_error_ms=pe,completion_error_ms=fe))
    work=collections.Counter(r.get('completed_D_prefill_layers',0) for r in requests.values())
    result=dict(name=name,decisions=len(decisions),decision_reasons=dict(reasons),candidate_reasons=dict(candidate_reasons),
        independent_candidate_gate_failures=dict(gate_failures),
        gate_count_scope='A candidate can fail multiple gates. Counts are not an exclusive partition. Older logs reconstruct the independent checks from recorded predictions.',
        actual_completed_layer_distribution=dict(work),migrated_requests=sum(n for k,n in work.items() if k),
        completed_migrated_layers=sum(k*n for k,n in work.items()),total_prefill_layers=36*len(requests),
        active_cohort_drain=errors(active_errors),known_active_and_pending_drain=errors(known_errors),
        source_ready=errors(prefix_errors),suffix_completion_to_first=errors(finish_errors),
        accepted_predictions=observations,evidence='TRACE_ANALYSIS',
        physical_idle_prediction_accuracy='NOT_IDENTIFIED',
        scope='Outcomes enter this offline analysis only. Drain errors use the known request set at the decision. They include realized queue and control costs. Prediction accuracy is not a physical slack measurement.')
    return result


def main():
    rows=[]
    for p in sorted((ROOT/'raw').glob('*')):
        if not p.is_dir() or not (p/'online_decisions.jsonl').exists():continue
        receipt=p/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        value=analyze(p.name)
        if value:rows.append(value)
    write(ROOT/'analysis/forecast_validation.json',dict(results=rows,
        scope='Prediction errors do not establish unknown-future safety. Physical kernel idle remains a separate metrology gate.'))
    print('FORECAST_ANALYSIS',len(rows))


if __name__=='__main__':main()
