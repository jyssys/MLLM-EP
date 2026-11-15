"""Quantify gate overlap, retained source state, and delivery timestamp identity."""
import collections
from .common import *
from .analyze import records,percentile


def distribution(values):
    return dict(samples=len(values),mean=sum(values)/len(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),maximum=max(values,default=None))


def main():
    initialize();audit=json.loads((ROOT/'analysis/long_output_gate_attribution.json').read_text())
    summary={};removed_sets=[('WAIT_ONLY',{'BOUNDED_ADMISSION_WAIT_EXCEEDED'}),
        ('INGRESS_WINDOW_ONLY',{'INSUFFICIENT_KNOWN_INGRESS_WINDOW'}),
        ('WAIT_AND_INGRESS_WINDOW',{'BOUNDED_ADMISSION_WAIT_EXCEEDED','INSUFFICIENT_KNOWN_INGRESS_WINDOW'})]
    for cap in (64,200):
        group=[r for r in audit['runs'] if f'_O{cap}_' in r['name']]
        gates={str(m):collections.Counter() for m in (2,4,8,12)}
        counterfactual=collections.Counter();requests=0;choice=collections.Counter()
        for r in group:
            replay=json.loads((ROOT/'analysis'/(r['name']+'_gate_replay.json')).read_text())
            for m,values in replay['candidate_overlapping_gate_request_counts'].items():gates[m].update(values)
            choice.update(r['choice_reasons'])
            for d in replay['details']:
                requests+=d['requests']
                for label,removed in removed_sets:
                    if d['global_rejections']:continue
                    if any(not (set(c['rejection_reasons'])-removed) for c in d['independent_candidates']):
                        counterfactual[label]+=d['requests']
        summary[str(cap)]=dict(runs=len(group),requests=requests,choice_reasons=choice,
            candidate_overlapping_request_counts=gates,removed_predicate_counterfactual_requests=counterfactual,
            evidence='TRACE_ANALYSIS',counterfactual_evidence='OFFLINE_ORACLE',
            scope='Removing predicates uses the original observed states. It neither updates later queues nor proves safe execution.')
    retention=[]
    for seed in (2026111381,2026111382,2026111383):
        name=f'heldout_Revised_O200_N128_q4.8_s{seed}_v1';folder=PREVIOUS/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text());window=json.loads((folder/'window.json').read_text())
        source={r['request_id']:r['time'] for r in records(folder/'native_handoffs.jsonl')}
        admissions={r['request_id']:r['time'] for r in records(folder/'D_host_admission.jsonl')}
        waits=[1000*(admissions[r]-at) for r,at in source.items() if r in admissions]
        p=json.loads(next(folder.glob('native_prefill_*.json')).read_text())
        d=json.loads(next(folder.glob('native_decode_*.json')).read_text())
        samples=[s for s in p['queue_samples'] if window['start']<=s['time']<=window['end']]
        blocked=[s for s in samples if s['waiting'] and s['KV_free_blocks']<500]
        retention.append(dict(name=name,source_ready_to_D_host_admission_ms=distribution(waits),
            P_KV_allocation_retries=len(p['KV_allocation_refusals']),P_KV_free_min=min(s['KV_free_blocks'] for s in samples),
            D_KV_free_min=min(s['KV_free_blocks'] for s in d['queue_samples']),
            P_queue_samples=len(samples),P_waiting_and_fewer_than_one_prompt_blocks_samples=len(blocked),
            P_waiting_and_low_KV_sample_fraction=len(blocked)/len(samples),
            structural_prompt_blocks=500,source_prompt_capacity_at19113_free_blocks=38,
            scope='Source must retain the valid500-block prompt until destination completion notification. Source-ready to D host admission is a lower bound on this retention interval. It excludes wire completion and ACK. Allocation retries are repeated attempts, not distinct rejected requests.'))
    name='heldout_Revised_O200_N128_q4.8_s2026111383_v1';rows=records(PREVIOUS/'raw'/name/'requests.jsonl')
    row=next(r for r in rows if r['sample_id']==107)
    a,b=next((a,b) for a,b in zip(row['token_events'],row['token_events'][1:]) if a['count']==17 and b['count']==18)
    parent=1000*(b['time']-a['time']);worker=1000*(b['worker_time']-a['worker_time'])
    lag=1000*((b['time']-b['worker_time'])-(a['time']-a['worker_time']))
    assert abs(parent-worker-lag)<1e-7
    write(ROOT/'analysis/long_output_zero_migration_root_cause.json',dict(gates=summary,
        retention=retention,host_spike=dict(name=name,request='q107',counts=[17,18],
            coordinator_ITL_ms=parent,worker_emission_interval_ms=worker,delivery_lag_change_ms=lag,
            identity_residual_ms=parent-worker-lag,exact_Host_cause='NOT_IDENTIFIED',
            scope='This identity explains the coordinator spike numerically. The worker interval is not CUDA kernel time. The lag increase does not identify its scheduling or GC cause.'),
        physical_GPU_idle='NOT_MEASURED',evidence='TRACE_ANALYSIS'))
    print(json.dumps(dict(gates=summary,retention=retention),indent=2))


if __name__=='__main__':main()
