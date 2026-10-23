"""Export request clocks and quiet pooled metrics without mixing lanes."""
import gzip
import hashlib
import itertools
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file
from .metrics import summarize

def main():
    requests=[];events=[];queues=[];quiet=defaultdict(list);matrix=[]
    for plan_path in sorted(ROOT.glob('*plan.json')):
        plan=json.loads(plan_path.read_text())
        jobs=plan.get('jobs',[j for group in plan.get('groups',[]) for j in group['jobs']])
        for job in jobs:
            folder=ROOT/'raw'/job['id']
            identity=json.loads((folder/'identity.json').read_text()) if (folder/'identity.json').exists() else {}
            matrix.append(dict(plan=plan_path.name,run_id=job['id'],kind=job['kind'],registered_GPU=job['gpu'],
                actual_GPU=identity.get('physical_GPU'),cells_json=json.dumps(job.get('cells')),trials_json=json.dumps(job.get('trials')),
                cap=job.get('cap'),width=job.get('width'),compute_budget_ms=job.get('compute_budget_ms'),quiet=job.get('quiet',False),
                observed=job.get('observe',job['kind'] in ('service','fairness','numerical','observer_control','isolated_service')),
                status='COMPLETE' if (folder/'completed.json').exists() else 'FAILED' if (folder/'failure.json').exists() else
                    'RUNNING_OR_INTERRUPTED' if (folder/'identity.json').exists() else 'NOT_RUN',
                shape=json.dumps(job.get('shape',job.get('shapes'))),split=job.get('split'),
                scope=plan.get('scope',plan.get('qualification','See registered plan'))))
    for pattern in ('cb_p1_fairness_*','cb_p3_budget_*','cb_p5_confirm_*'):
        for folder in sorted((ROOT/'raw').glob(pattern)):
            if not (folder/'completed.json').exists():
                continue
            identity=json.loads((folder/'identity.json').read_text());job=identity['job']
            budget=next((c['budget_ms'] for c in job['cells'] if c.get('label')=='J1'),None)
            for file in folder.glob('*_T*.json'):
                data=json.loads(file.read_text())
                if 'requests' not in data:
                    continue
                label=data.get('label',data['policy'])
                meta=dict(run_id=folder.name,policy=label,P=data['P'],N=data['configured_N'],trial=data['trial'],
                    GPU=data['physical_GPU'],budget_ms=budget,policy_budget_ms=data.get('configured_budget_ms'),
                    evidence=data['lane'],source=str(file.relative_to(ROOT)))
                if folder.name.startswith('cb_p5_confirm'):
                    quiet[(data['P'],data['configured_N'],budget,label)].append(data)
                for request in data['requests']:
                    base=dict(meta,id=request['id'],family=request['family'])
                    requests.append(dict(base,input_tokens=request['input_tokens'],output_tokens=len(request['output_ids']),
                        output_sha256=hashlib.sha256(json.dumps(request['output_ids']).encode()).hexdigest(),
                        TTFT_s=request.get('TTFT_s'),TPOT_s=request.get('TPOT_s'),E2E_s=request['E2E_s'],
                        cap=request['cap'],finish_reason=request.get('finish_reason','ERROR_OR_TIMEOUT'),status=request['status'],
                        scheduled_arrival=request['scheduled_arrival'],actual_send=request.get('actual_send'),
                        engine_enqueue=request.get('engine_enqueue'),completion=request['completion']))
                    events.append(dict(base,event='SCHEDULED_ARRIVAL',token_index=None,time_s=request['scheduled_arrival']))
                    for index,clock in enumerate(request['token_times']):
                        events.append(dict(base,event='OUTPUT_TOKEN',token_index=index,time_s=clock))
                    events.append(dict(base,event='REQUEST_COMPLETION',token_index=None,time_s=request['completion']))
                for row in data['queue_series']:
                    queues.append(dict(meta,**row))
    csv_file(ROOT/'experiment_matrix.csv',matrix);csv_file(ROOT/'request_metrics.csv.gz',requests)
    csv_file(ROOT/'request_events.csv.gz',events);csv_file(ROOT/'queue_metrics.csv',queues)
    table=[]
    for (P,N,B,label),cohorts in sorted(quiet.items()):
        rows=[r for d in cohorts for r in d['requests']];gaps=[g for r in rows for g in r.get('ITLs_s',[])]
        values=[summarize(d) for d in cohorts];wall=sum(r['drained_wall_s'] for r in values)
        row=dict(P=P,N=N,budget_ms=B,policy=label,independent_starts=len({d['run_id'] for d in cohorts}),
            policy_budget_ms=cohorts[0].get('configured_budget_ms'),
            cohorts=len(cohorts),offered=len(rows),completed=sum(r['complete_responses'] for r in values),
            capped=sum(r['capped'] for r in values),output_tokens=sum(r['output_tokens'] for r in values),
            drained_output_tokens_s=sum(r['output_tokens'] for r in values)/wall,
            drained_completed_requests_s=sum(r['complete_responses'] for r in values)/wall,
            peak_allocated_GiB=max(d['peak_allocated_bytes'] for d in cohorts)/2**30,
            peak_reserved_GiB=max(d['peak_reserved_bytes'] for d in cohorts)/2**30,
            evidence='MEASURED_CLEAN_FINITE_COHORT_NOT_CAPACITY')
        for q in (50,95,99):
            row[f'TTFT_pooled_P{q}_ms']=1000*float(np.percentile([r['TTFT_s'] for r in rows],q))
            row[f'ITL_pooled_P{q}_ms']=1000*float(np.percentile(gaps,q))
            row[f'E2E_request_pooled_P{q}_ms']=1000*float(np.percentile([r['E2E_s'] for r in rows],q))
        table.append(row)
    bykey={(r['P'],r['N'],r['budget_ms'],r['policy']):r for r in table}
    for row in table:
        ref=bykey.get((row['P'],row['N'],row['budget_ms'],'C1A'))
        row['output_throughput_speedup_over_C1A']=None if ref is None else row['drained_output_tokens_s']/ref['drained_output_tokens_s']
    csv_file(ROOT/'main_table.csv',table)
    paired=ROOT/'pilot_paired_effects.csv'
    if paired.exists():
        import csv
        with paired.open() as stream:
            quiet_pairs=[r for r in csv.DictReader(stream) if r['lane']=='QUIET_FINAL']
        csv_file(ROOT/'matched_pairs.csv',quiet_pairs)
    print('Exported requests',len(requests),'events',len(events),'quiet table rows',len(table))

if __name__=='__main__':
    main()
