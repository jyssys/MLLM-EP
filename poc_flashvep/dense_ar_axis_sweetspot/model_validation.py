"""Fit historical discovery costs, then predict separate confirmation starts."""
import collections
import json
import statistics
import time
from .common import PREVIOUS,ROOT,write,digest,initialize
from .historical_audit import records
from .replay import Cost,simulate,summary


def extract(root,name):
    host=json.loads((root/'analysis'/(name+'_host_paths.json')).read_text())
    cfg=json.loads((root/'configs'/(name+'.json')).read_text())
    rows=records(root/'raw'/name/'requests.jsonl')
    byid={'q'+str(r['sample_id']):r for r in rows}
    p=sorted(host['request_paths'],key=lambda r:int(r['request_id'][1:]))
    preparations=[];decodes=[];D_gaps=[];P_gaps=[]
    previous=None
    for c in host['cohorts']:
        first=max(byid[r]['first_token'] for r in c['requests'])
        finish=max(byid[r]['finish'] for r in c['requests'])
        preparations.append((first-c['D_host_submit'])/c['size'])
        decodes.append(finish-first)
        allready=max(p[int(r[1:])]['boundaries_monotonic'][2] for r in c['requests'])
        if previous is not None and allready<=previous['last_client_notice']:
            D_gaps.append(c['D_host_submit']-previous['last_client_notice'])
        previous=c
    for prev,current in zip(p,p[1:]):
        a,ps,ready=current['boundaries_monotonic'][:3]
        prevready=prev['boundaries_monotonic'][2]
        if a<prevready and 0<=ps-prevready<.05:
            P_gaps.append(ps-prevready)
    costs=dict(P=statistics.mean(r['spans_ms']['P_submit_to_source_notification'] for r in p)/1000,
        preparation=statistics.mean(preparations),decode=statistics.mean(decodes),
        P_gap=statistics.median(P_gaps) if P_gaps else .001,
        D_gap=statistics.median(D_gaps) if D_gaps else .001)
    return cfg,rows,costs


def main():
    initialize()
    starts=json.loads((PREVIOUS/'analysis/independent_start_statistics.json').read_text())['starts']
    training=[r for r in starts if r['stage'] in ('batch1_discovery','cohort2_discovery') and r['method']!='Native']
    testing=[r for r in starts if r['stage'] in ('batch1_confirmation','cohort2_confirmation') and r['method']!='Native']
    grouped=collections.defaultdict(list)
    for r in training:
        cfg,rows,cost=extract(PREVIOUS,r['name'])
        grouped[(r['method'],r['layers'],r['cap'],cfg['D_admission_batch'])].append(cost)
    models={k:{f:statistics.median(c[f] for c in costs) for f in costs[0]} for k,costs in grouped.items()}
    # Freeze a numerical validation gate before inspecting replay errors.
    gate=dict(TTFT_p95_relative_error_max=.10,output_tokens_s_relative_error_max=.05,
        Ready_D_mean_absolute_or_relative_error_max=dict(absolute_ms=10.,relative=.20),
        interpretation='All three bounds must pass. Passing fixed traces does not validate counterfactual KV pressure, mixed-axis execution or measured ITL tails.')
    path=ROOT/'state/replay_validation_gate_before_results.json'
    if path.exists():
        assert json.loads(path.read_text())==gate
    else:
        write(path,gate)
    results=[]
    for r in testing:
        cfg,rows,_=extract(PREVIOUS,r['name'])
        key=(r['method'],r['layers'],r['cap'],cfg['D_admission_batch'])
        if key not in models:
            continue
        ordered=sorted(rows,key=lambda row:row['sample_id'])
        start=min(row['arrival'] for row in ordered)
        arrivals=[row['arrival']-start for row in ordered]
        sim=simulate(arrivals,[Cost(**models[key])]*len(rows),cfg['D_admission_batch'],cfg['source_outstanding_limit'])
        prediction=summary(sim,r['cap'])
        actual=json.loads((PREVIOUS/'analysis'/(r['name']+'_host_paths.json')).read_text())
        actual_wait=actual['summaries']['source_notification_to_D_host_submit']['mean']
        errors=dict(TTFT_p95_relative=abs(prediction['TTFT_p95_ms']/r['TTFT_p95_ms']-1),
            output_tokens_s_relative=abs(prediction['output_tokens_s']/r['output_tokens_s']-1),
            Ready_D_mean_absolute_ms=abs(prediction['Ready_D_wait_mean_ms']-actual_wait))
        passed=(errors['TTFT_p95_relative']<=.10 and errors['output_tokens_s_relative']<=.05
            and errors['Ready_D_mean_absolute_ms']<=max(10.,.20*actual_wait))
        results.append(dict(name=r['name'],method=r['method'],layers=r['layers'],output=r['cap'],
            batch=cfg['D_admission_batch'],seed=r['seed'],prediction=prediction,
            actual=dict(TTFT_p95_ms=r['TTFT_p95_ms'],output_tokens_s=r['output_tokens_s'],Ready_D_mean_ms=actual_wait),
            errors=errors,passed=passed,evidence='OFFLINE_REPLAY'))
    write(ROOT/'analysis/historical_replay_validation.json',dict(time=time.time(),
        training_names=[r['name'] for r in training],testing_names=[r['name'] for r in testing],
        models=[dict(method=k[0],layers=k[1],output=k[2],batch=k[3],cost_seconds=v) for k,v in models.items()],
        gate=gate,results=results,passed=sum(r['passed'] for r in results),total=len(results),
        scope='Fit only historical discovery starts. Predict separate historical confirmation seeds. Costs include host/transport and guards. No future test completion enters prediction. Mean-cost replay cannot establish Native-relative Decode5% noninferiority or physical GPU slack.'))
    print('HISTORICAL_REPLAY_GATE',sum(r['passed'] for r in results),len(results),flush=True)
    for b in (1,2):
        for m in (4,8,12):
            v=[r for r in results if r['batch']==b and r['layers']==m]
            if v:
                print(b,m,len(v),sum(r['passed'] for r in v),
                    {k:round(statistics.mean(r['errors'][k] for r in v),4) for k in v[0]['errors']},flush=True)


if __name__=='__main__':
    main()
