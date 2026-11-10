"""Account for every old request. Keep observations separate from hypotheses."""
import csv
from collections import Counter
from .common import *


def events(folder,begin):
    result=[]
    for path in folder.glob('intervention_*.json'):
        result.extend(r for r in json.loads(path.read_text()).get('records',[]) if r.get('time',0)>=begin)
    return sorted(result,key=lambda r:r['time'])


def main():
    initialize();rows=[];summary=[];gaps=[]
    for model in ('ouro','huginn'):
        for seed in (2026110911,2026110912,2026110913):
            name=f'confirm_{model}_tile_seed{seed}';folder=OLD/'raw'/name
            begin=json.loads((folder/'measurement_begin.json').read_text())['time'];ev=events(folder,begin)
            arms={r['seed']:r for r in ev if r['event']=='SOURCE_ARM_DECISION'}
            reqs=sorted([json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()
                if l and not json.loads(l).get('warmup')],key=lambda r:r['sample_id'])
            ms=json.loads((folder/'stats.json').read_text())['metrics']
            mids={m['request_id']:m for m in ms}
            residence=[]
            by_tid={m['transfer_id']:m for m in ms}
            for c in (r for r in ev if r['event']=='DESTINATION_COMMIT'):
                m=by_tid.get(c['tid'])
                if m and 'first_token' in m['timings']:
                    residence.append((c['time'],m['created_monotonic']+m['timings']['first_token']))
            for req in reqs:
                a=arms.get(req['seed'],{});tid=mids[f"r0_q{req['sample_id']}"]['transfer_id']
                ad=[r for r in ev if r['event']=='ADMISSION' and r['tid']==tid]
                commit=[r for r in ev if r['event']=='SOURCE_COMMIT' and r['tid']==tid]
                offers=[r for r in ev if r['event']=='GPU_COMPLETED_BOUNDARY_OFFER' and r['tid']==tid]
                ad_time=a.get('time',0)-(a.get('advertisement_age_ms') or 0)/1000
                rows.append(dict(model=model,name=name,sample_id=req['sample_id'],sampling_seed=req['seed'],
                    tid=tid,arm_recorded=bool(a),armed=a.get('armed'),P_distinct_stage_work=a.get('P_distinct_stage_work'),
                    P_credit_clear_waiters=a.get('P_credit_clear_waiters'),D_available=a.get('D_available'),
                    D_future_quantum_budget_ms=a.get('D_future_quantum_budget_ms'),
                    advertisement_age_ms=a.get('advertisement_age_ms'),offers=len(offers),
                    capacity_unavailable_with_committed_resident=a.get('D_available') is False
                        and any(start<=ad_time<=end for start,end in residence),
                    D_admission_evaluations=len(ad),D_preparations=sum(r['accepted'] for r in ad),
                    committed=bool(commit),offloaded_loops=commit[0]['remaining'] if commit else 0,
                    admission_reasons='|'.join(str(r.get('rejection_reason')) for r in ad),
                    request_TTFT_ms=req['ttft_ms']))
                if commit:
                    qs=sorted([r for r in ev if r['event']=='D_QUANTUM_HOST_COMPLETION'
                        and r.get('choice',{}).get('tid')==tid],key=lambda r:r['submitted_host_time'])
                    largest=max((b['submitted_host_time']-a['time'] for a,b in zip(qs,qs[1:])),default=None)
                    m=mids[f"r0_q{req['sample_id']}"];received=m['created_monotonic']+m['timings']['d_received']
                    gaps.append(dict(model=model,name=name,tid=tid,quanta=len(qs),
                        largest_inter_quantum_submission_gap_s=largest,
                        import_to_first_output_ms=(req['first_token']-received)*1000,
                        scope='Measured host observations. Not GPU kernel idle.'))
            subset=[r for r in rows if r['name']==name]
            rejects=Counter(r.get('rejection_reason') for r in ev if r['event']=='ADMISSION' and not r['accepted'])
            summary.append(dict(model=model,name=name,requests=len(reqs),arm_evaluations=len(arms),
                armed=sum(bool(r['armed']) for r in subset),committed=sum(r['committed'] for r in subset),
                offloaded_loops=sum(r['offloaded_loops'] for r in subset),
                initial_D_unavailable=sum(r['D_available'] is False for r in subset),
                initial_no_advertisement=sum(r['D_available'] is None for r in subset),
                unavailable_with_committed_resident=sum(r['capacity_unavailable_with_committed_resident'] for r in subset),
                offers=sum(r['offers'] for r in subset),rejection_counts=dict(rejects),
                preparations=sum(r['event']=='ADMISSION' and r['accepted'] for r in ev),
                source_continue_reasons=dict(Counter(r.get('reason') for r in ev if r['event']=='SOURCE_CONTINUE')),
                lease_timeouts=sum(r['event']=='SOURCE_LEASE_TIMEOUT' for r in ev),
                observations=[r for r in ev if r['event']=='EXTRA_WORK_SUSPENDED'],
                interpretation='Initial gates can overlap. Missing an offer is not a measured GPU idle interval.'))
    for filename,values in (('old_request_gate_inventory.csv',rows),('old_continuation_gaps.csv',gaps)):
        with (ROOT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    write(ROOT/'old_gate_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
