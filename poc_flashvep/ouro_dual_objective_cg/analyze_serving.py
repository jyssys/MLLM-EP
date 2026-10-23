"""Request clocks, exact-plan comparisons, and paired process-block intervals."""
import argparse
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write,digest
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def quantile(values,q):return float(np.percentile(values,q)) if values else None

def completed_response(request):
    value=str(request.get('finish_reason','')).split('.')[-1].lower()
    return request['status']=='COMPLETE' and value in ('stop','eos','frontend_stop','frontend_registered_stop')

def comparison_stratum(data):
    return (data['split'],data['lane'],data['workload'],data['scenario'].get('rate'),data['configured_N'])

def summarize(data):
    requests=data['requests'];gaps=[1000*g for r in requests for g in r.get('ITLs_s',[])]
    wall=data['end']-data['start'];scenario=data['scenario'];cell=data['cell']
    row=dict(run_id=data['run_id'],scenario=scenario['id'],GPU=data['physical_GPU'],
        split=data['split'],workload=data['workload'],width=data['configured_N'],policy=data['label'],
        lane=data['lane'],offered_rate_rps=scenario.get('rate'),arrival_type=scenario.get('arrival_type','FINITE_INTERFERENCE'),
        realized_offered_rate_rps=len(requests)/max(.001,data['arrival_window_s']),
        objective=cell.get('objective','NATIVE_CONTROL'),surface=cell.get('surface','NATIVE'),
        B_ms=cell.get('budget_ms'),offered=len(requests),
        completed=sum(completed_response(r) for r in requests),caps=sum(not completed_response(r) for r in requests),
        output_tokens=sum(len(r['output_ids']) for r in requests),drained_wall_s=wall,
        output_tokens_s=sum(len(r['output_ids']) for r in requests)/wall,
        drained_completed_requests_s=sum(completed_response(r) for r in requests)/wall,
        peak_HBM_GiB=data['peak_allocated_bytes']/2**30,peak_reserved_GiB=data['peak_reserved_bytes']/2**30,
        full_R4=data['full_R4'],source_families=len(set(data['families'])),
        evidence=data['lane'],qualification='FINITE_COHORT_NOT_SUSTAINABLE_CAPACITY')
    for q in (50,95,99):
        row[f'TTFT_P{q}_ms']=quantile([1000*r['TTFT_s'] for r in requests if r.get('TTFT_s') is not None],q)
        row[f'ITL_P{q}_ms']=quantile(gaps,q)
        row[f'TPOT_P{q}_ms']=quantile([1000*r['TPOT_s'] for r in requests if r.get('TPOT_s') is not None],q)
        row[f'request_max_ITL_P{q}_ms']=quantile([1000*max(r.get('ITLs_s',[]),default=0) for r in requests],q)
        row[f'E2E_P{q}_ms']=quantile([1000*r['E2E_s'] for r in requests],q)
        row[f'complete_E2E_P{q}_ms']=quantile([1000*r['E2E_s'] for r in requests if completed_response(r)],q)
    row['E2E_scope']='All returned stops/caps for E2E_*. complete_E2E_* contains actual EOS/frontend stops only.'
    for threshold in (50,100,150,200,250,500):
        row[f'ITL_gt_{threshold}ms']=sum(g>threshold for g in gaps)
    decisions=data.get('optimizer_decisions',[])
    fresh=[d for d in decisions if d['state']['kind']=='NEW_CHUNK']
    multi=[d for d in fresh if len({c['C'] for c in d['candidates']})>1]
    fixed=cell.get('fixed_C',2048)
    row.update(selector_calls=len(decisions),fresh_calls=len(fresh),multiple_feasible_C_fresh=len(multi),
        C_differs_fixed_same_state=sum(d['selected']['C']!=min(fixed,d['state']['prompt_remaining'],d['state']['token_budget']) for d in fresh),
        C_histogram=json.dumps(dict(__import__('collections').Counter(d['selected']['C'] for d in fresh))),
        G_histogram=json.dumps(dict(__import__('collections').Counter(d['chosen_G'] for d in data.get('chunk_decisions',[]) if d['phase']=='prefilling_recurrent'))),
        selector_total_ms=sum(d['selector_ns'] for d in decisions)/1e6,
        admission_total_ms=0. if data['admission_cpu'] is None else data['admission_cpu']['total_ns']/1e6,
        progress_escape=len(data.get('overbudget_admission',[])))
    for ttft in (2.,5.,10.):
        row[f'TTFT_attainment_{ttft:g}s_pct']=100*sum(r.get('TTFT_s',float('inf'))<=ttft for r in requests)/len(requests)
        for tbt in (.100,.150,.200,.250,.500):
            passed=[r for r in requests if completed_response(r) and r.get('TTFT_s',float('inf'))<=ttft
                and all(g<=tbt for g in r.get('ITLs_s',[]))]
            key=f'TTFT{ttft:g}s_TBT{int(tbt*1000)}ms'
            row[f'joint_attainment_{key}_pct']=100*len(passed)/len(requests)
            row[f'drain_goodput_{key}_rps']=len(passed)/wall
            row[f'arrival_window_goodput_{key}_rps']=sum(r['completion']<=data['start']+data['arrival_window_s'] for r in passed)/max(.001,data['arrival_window_s'])
    for tbt in (.100,.150,.200,.250,.500):
        row[f'complete_every_gap_TBT_attainment_{int(tbt*1000)}ms_pct']=100*sum(completed_response(r) and
            all(g<=tbt for g in r.get('ITLs_s',[])) for r in requests)/len(requests)
        row[f'pooled_P99_TBT_pass_{int(tbt*1000)}ms']=None if not gaps else np.percentile(gaps,99)<=1000*tbt
    queue=[r for r in data['queue_series'] if .5*data['arrival_window_s']<=r['time']<=data['arrival_window_s']]
    row['queue_window_samples']=len(queue)
    row['queue_slope_requests_s']=None if len(queue)<3 else float(np.polyfit([r['time'] for r in queue],[r['outstanding'] for r in queue],1)[0])
    row['unfinished_at_arrival_window_end']=sum(r['scheduled_arrival']<=data['start']+data['arrival_window_s']<r['completion'] for r in requests)
    return row

def actual_plan(data):
    result=defaultdict(list)
    for row in data['dispatch_descriptors']:
        for span in row['spans']:result[span['id']].append((span['start'],span['tokens'],span['depth'],span['G'],span['phase']))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--require-validation-complete',action='store_true');a=p.parse_args()
    plan=json.loads((ROOT/'p2_objective_validation_plan.json').read_text())
    if a.require_validation_complete:
        assert all((ROOT/'raw'/j['id']/'completed.json').exists() for j in plan['jobs'])
    rows=[];raw={};requests=[];events=[];parity=[]
    for folder in sorted((ROOT/'raw').glob('dual_p[2345]_*')):
        if not (folder/'completed.json').exists():continue
        for file in folder.glob('*.json'):
            data=json.loads(file.read_text())
            if not isinstance(data,dict) or 'scenario' not in data or 'label' not in data:continue
            row=summarize(data);row['source']=str(file.relative_to(ROOT));row['sha256']=digest(file)
            rows.append(row);raw[(data['run_id'],data['scenario']['id'],data['label'])]=data
            for request in data['requests']:
                requests.append(dict(run_id=data['run_id'],scenario=data['scenario']['id'],policy=data['label'],
                    id=request['id'],family=data['families'][int(request['id'])],
                    input_tokens=request['input_tokens'],output_tokens=len(request['output_ids']),
                    TTFT_s=request.get('TTFT_s'),TPOT_s=request.get('TPOT_s'),E2E_s=request['E2E_s'],
                    scheduled_arrival=request['scheduled_arrival'],completion=request['completion'],
                    finish_reason=request['finish_reason'],complete=completed_response(request),
                    max_ITL_ms=max([1000*g for g in request.get('ITLs_s',[])],default=0.),
                    output_ids_sha256=__import__('hashlib').sha256(json.dumps(request['output_ids']).encode()).hexdigest(),
                    full_R4=data['full_R4'],source=str(file.relative_to(ROOT))))
                base=dict(run_id=data['run_id'],scenario=data['scenario']['id'],policy=data['label'],id=request['id'])
                events.append(dict(base,event='SCHEDULED_ARRIVAL',index=None,time_s=request['scheduled_arrival']))
                events.extend(dict(base,event='OUTPUT_TOKEN',index=i,time_s=t) for i,t in enumerate(request['token_times']))
                events.append(dict(base,event='REQUEST_END',index=None,time_s=request['completion']))
    comparisons=defaultdict(list)
    lookup={(r['run_id'],r['scenario'],r['policy']):r for r in rows}
    for key,left in raw.items():
        run,scenario,label=key
        if not label.endswith('_J'):continue
        family=label[:-2]
        for control in (family+'_G',family+'_G1024',family+'_C',family+'_CA',
            'BEST_FIXED','ATOMIC','STAGED4','MIXED_STOCK','MIXED_FAIR'):
            right=raw.get((run,scenario,control))
            if right is None:continue
            assert left['scenario']==right['scenario'] and left['families']==right['families']
            lp,rp=actual_plan(left),actual_plan(right)
            byid={r['id']:r for r in right['requests']}
            for request in left['requests']:
                rid=request['id'];other=byid[rid]
                parity.append(dict(run_id=run,scenario=scenario,candidate=label,reference=control,id=rid,
                    plan_equal=lp[rid]==rp[rid],token_exact=request['output_ids']==other['output_ids'],
                    output_count_equal=len(request['output_ids'])==len(other['output_ids']),
                    candidate_tokens=len(request['output_ids']),reference_tokens=len(other['output_ids']),
                    finish_equal=str(request['finish_reason'])==str(other['finish_reason'])))
            l,r=lookup[key],lookup[(run,scenario,control)]
            for metric in ('output_tokens_s','TTFT_P95_ms','ITL_P95_ms','ITL_P99_ms','E2E_P95_ms',
                'joint_attainment_TTFT5s_TBT200ms_pct','drain_goodput_TTFT5s_TBT200ms_rps'):
                if l[metric] is None or r[metric] is None:continue
                comparisons[(*comparison_stratum(left),label,control,metric)].append(
                    (run,scenario,l[metric],r[metric]))
    paired=[];rng=np.random.default_rng(2026102302)
    for key,values in comparisons.items():
        split,lane,workload,rate,width,label,control,metric=key;blocks=defaultdict(list)
        for run,scenario,l,r in values:blocks[run].append((l,r))
        block=np.array([np.mean(v,axis=0) for v in blocks.values()])
        boot=block[rng.integers(0,len(block),size=(3000,len(block)))].mean(axis=1)
        delta=boot[:,0]-boot[:,1]
        paired.append(dict(split=split,lane=lane,workload=workload,offered_rate_rps=rate,width=width,candidate=label,reference=control,metric=metric,
            independent_starts=len(block),paired_cohorts=len(values),candidate_mean=float(block[:,0].mean()),
            reference_mean=float(block[:,1].mean()),delta=float(np.mean(block[:,0]-block[:,1])),
            CI95_low=None if len(block)<2 else float(np.percentile(delta,2.5)),
            CI95_high=None if len(block)<2 else float(np.percentile(delta,97.5)),
            ratio=float(block[:,0].mean()/block[:,1].mean()) if block[:,1].mean()!=0 else None,
            scope='Paired process-block means. No CI for a single start. Repeated source cohorts limit generalization. Tokens are not resampled independently.'))
    csv_file(ROOT/'serving_results.csv',rows);csv_file(ROOT/'request_metrics.csv.gz',requests)
    csv_file(ROOT/'request_events.csv.gz',events)
    csv_file(ROOT/'paired_comparisons.csv',paired);csv_file(ROOT/'serving_output_plan_parity.csv',parity)
    write(ROOT/'serving_analysis_status.json',dict(cohorts=len(rows),requests=len(requests),
        completed_validation_jobs=sum((ROOT/'raw'/j['id']/'completed.json').exists() for j in plan['jobs']),
        registered_validation_jobs=len(plan['jobs']),qualified_capacity=False))
    print('Analyzed completed serving cohorts',len(rows),'requests',len(requests),'paired intervals',len(paired))

if __name__=='__main__':main()
