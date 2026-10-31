"""Classify finite pressure. Do not equate utilization or a ratio with a bottleneck."""
import collections
import statistics
from .common import *
from .summary_tables import records,save_csv,percentile
from .analyze_serving import analyze

def main():
    result=[];slo_file=ROOT/'configs/frozen_SLO.json'
    slos=json.loads(slo_file.read_text()) if slo_file.exists() else {}
    for run in analyze():
        folder=ROOT/'raw'/run['name'];cfg=json.loads((ROOT/'configs'/(run['name']+'.json')).read_text())
        if cfg.get('phase') not in ('crossover','focused_crossover','critical_repeat') or run['status']!='COMPLETE':continue
        if cfg['model'] not in ('ouro','huginn'):continue
        window=json.loads((folder/'window.json').read_text());start=window['start'];last=window['last_arrival']
        end=window['end'];capacity=cfg['capacity_planning'];muP=capacity['mu_P'];muD=capacity['mu_D']
        matching_capacity=not bool(cfg.get('protocol_control'))
        metrics=json.loads((folder/'closed_stats.json').read_text())['metrics']
        metrics=[r for r in metrics if r['request_id'].startswith('r0_')]
        p_live=[];core_wait=[];last_prelude={};first_core={};samples=[]
        for path in folder.glob('host_execution_*.json'):
            host=json.loads(path.read_text())
            if host['role']!='prefill':continue
            samples.extend(s for s in host['queue_samples'] if start<=s['time']<=last)
            for event in host['frontiers']:
                if event['event']=='P_PRELUDE_FINAL_SUBMISSION':last_prelude[event['request_id']]=event['time']
                if event['event']=='P_FIRST_RECURRENT_SUBMISSION':first_core[event['request_id']]=event['time']
        for r in metrics:
            t=r['timings'];created=r['created_monotonic']
            if 'prefill_started' in t and 'prefill_compute_completed' in t:
                p_live.append((created+t['prefill_started'],created+t['prefill_compute_completed']))
            tid=r['transfer_id']
            if tid in last_prelude and tid in first_core:core_wait.append(max(0.,first_core[tid]-last_prelude[tid]))
        points=[start+(last-start)*i/100 for i in range(101)] if last>start else [start]
        backlog=[sum(a<=t<b for a,b in p_live) for t in points]
        growth=statistics.mean(backlog[-25:])-statistics.mean(backlog[:25]) if len(backlog)>25 else 0.
        queue_RNN=[s['queued'].get('RequestPhase.PREFILLING_RECURRENT',s['queued'].get('prefilling_recurrent',0)) for s in samples]
        admissions=[r for r in records(folder/'admission_reasons.jsonl') if start<=r['time']<=last]
        kv_fraction=sum(any('DECODE_NO_KV_CREDIT' in w['reasons'] for w in r['waiting']) for r in admissions)/max(1,len(admissions))
        slot_fraction=sum(any('DECODE_NO_SLOT' in w['reasons'] for w in r['waiting']) for r in admissions)/max(1,len(admissions))
        reservation_s=sum(r['timings'].get('admitted',0.) for r in metrics)
        request_s=sum(r['timings'].get('finished',0.) for r in metrics)
        reservation_share=reservation_s/max(request_s,1e-12)
        D_blocked_request_s=0.
        for a,b in zip(admissions,admissions[1:]):
            dt=b['time']-a['time']
            D_blocked_request_s+=dt*sum(any(reason in ('DECODE_NO_KV_CREDIT','DECODE_NO_SLOT') for reason in w['reasons']) for w in a['waiting'])
        D_blocked_share=D_blocked_request_s/max(request_s,1e-12)
        realized=(cfg['count']-1)/(last-start) if last>start else None
        decode_slack=cfg['model'] in slos and run['request_TPOT_ms_p95']<=.9*slos[cfg['model']]['TPOT_ms']
        growth_evidence=growth>=1. or (max(backlog,default=0)>=3 and percentile(core_wait,.95) is not None and percentile(core_wait,.95)>.1)
        P_candidate=bool(matching_capacity and realized and realized/muP>=.85 and realized/muD<=.85 and growth_evidence and
            kv_fraction<=.1 and slot_fraction<=.1 and D_blocked_share<=.1 and decode_slack)
        # This flag never authorizes a candidate GPU campaign by itself.
        if kv_fraction>.1 and reservation_share>.1:category='D_KV_ADMISSION_LIMITED'
        elif P_candidate:category='FINITE_P_PRESSURE_D_CAPACITY_CANDIDATE_PENDING_KERNEL_PROFILE'
        elif realized and realized/muD>=.85 and muD<muP:category='D_SERVICE_PRESSURE_GPU_HOST_CAUSE_PENDING_PROFILE'
        elif growth_evidence:category='INCONCLUSIVE_PRESSURE_WITHOUT_VERIFIED_D_SPARE'
        else:category='BALANCED_OR_LOW_LOAD_FINITE_COHORT'
        result.append(dict(name=run['name'],model=cfg['model'],cell=cfg['workload'],
            target_arrival_rate=cfg['arrival_rate'],realized_arrival_rate=realized,
            mu_P_planning=muP,mu_D_planning=muD,realized_to_muP=realized/muP if realized else None,
            realized_to_muD=realized/muD if realized else None,
            P_live_backlog_max=max(backlog,default=0),P_live_backlog_growth=growth,
            P_recurrent_queue_sample_max=max(queue_RNN,default=0),
            prelude_last_to_first_core_host_wait_p95_ms=1000*percentile(core_wait,.95) if core_wait else None,
            D_KV_refusal_observation_fraction=kv_fraction,D_slot_refusal_observation_fraction=slot_fraction,
            reservation_share_of_summed_request_time=reservation_share,decode_TPOT_SLO_margin_pass=decode_slack,
            sampled_D_credit_wait_share=D_blocked_share,
            capacity_matches_execution_configuration=matching_capacity,
            rate_planning_scope='Original C256 rate source retained for an identical-arrival chunk control.' if not matching_capacity else 'Independent service measurements for the declared configuration.',
            category=category,P_candidate=P_candidate,C_GPU_GATE_AUTHORIZED=False,
            scope='Finite diagnostic only. Require selected actual P/D kernel profile and source evidence before Phase C.'))
    save_csv(ROOT/'bottleneck_screening.csv',result)
    write(ROOT/'bottleneck_screening.json',result);print('SCREENED_RUNS',len(result))
    return result

if __name__=='__main__':main()
