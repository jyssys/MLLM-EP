"""Keep start-level uncertainty and evidence classes explicit."""
import csv,json,math,random,statistics
from .common import *
from .analyze import metrics,compare,records,percentile
from .serving_analysis import work,KEYS
from .state_qualification import events


def bootstrap(values):
    assert values and all(v>0 for v in values)
    rng=random.Random(2026111230);n=len(values)
    samples=[math.exp(statistics.mean(math.log(values[rng.randrange(n)]) for _ in range(n))) for _ in range(20000)]
    return dict(start_ratios=values,paired_geometric_mean=math.exp(statistics.mean(map(math.log,values))),
        CI95=[percentile(samples,.025),percentile(samples,.975)],independent_starts=n,
        scope='Paired resampling of independent Start units, not requests or tokens. Small Start counts give coarse intervals and limited power.')


def inventory():
    rows=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(p.read_text());name=c.get('name')
        if not name:continue
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        r=json.loads(receipt.read_text()) if receipt.exists() else {}
        row=dict(name=name,phase=c.get('phase'),family=c.get('ar_execution_family'),
            state='MEASURED' if r.get('complete') else 'FAILED' if r else 'INCOMPLETE' if folder.exists() else 'NOT_RUN',
            elapsed_seconds=r.get('elapsed_seconds'),role=c.get('arm'),cut=c.get('layer_cut'),
            config_SHA256=digest(p),GPU_mapping=c.get('gpus'),arrival_seed=c.get('arrival_seed'),
            count=c.get('count'),length=c.get('length'),output=c.get('cap'))
        if r.get('complete') and (folder/'requests.jsonl').exists():row['metrics']=metrics(name)
        rows.append(row)
    write(ROOT/'analysis/job_inventory.json',dict(rows=rows,scope='Failed and unexecuted configurations remain separate from measured results.'))
    return rows


def pilot():
    path=ROOT/'state/eager_serving_pilot_progress.json'
    if not path.exists():return
    rows=[];native=None
    for r in json.loads(path.read_text())['results']:
        m=metrics(r['name']);native=native or m
        rows.append(dict(name=r['name'],metrics=m,output_gate=r['output_gate'],work=work(r['name']),
            ratios={k:m[k]/native[k] if native[k] else None for k in KEYS},
            native_relative_Decode_pass=m['ITL_p99_ms']<=1.05*native['ITL_p99_ms'] and m['TPOT_p95_ms']<=1.05*native['TPOT_p95_ms']))
    write(ROOT/'analysis/eager_N64_comparison.json',dict(rows=rows,evidence='MEASURED_MARKER_FREE_SELECTION',
        scope='Pilot selection only. No confidence interval and no sustainable capacity claim.'))


def confirmation(plan_file='eager_confirmation_plan_v23.json',output_file='eager_N128_confirmation.json',csv_file='N128_serving_results.csv'):
    path=ROOT/'state'/plan_file
    if not path.exists():return
    plan=json.loads(path.read_text());groups=[]
    for g in plan['groups']:
        ms={k:metrics(v) for k,v in g['arms'].items()}
        if not all(x.get('complete') for x in ms.values()):continue
        checks=[compare(g['arms']['native'],g['arms'][k]) for k in ('layer','colocated')]
        groups.append(dict(seed=g['seed'],arms=g['arms'],metrics=ms,output_checks=checks,
            work={k:work(n) for k,n in g['arms'].items()}))
    if not groups:return
    stats={}
    for label in ('layer','colocated'):
        stats[label]={}
        for key in KEYS:
            ratios=[g['metrics'][label][key]/g['metrics']['native'][key] for g in groups if g['metrics']['native'][key]>0 and g['metrics'][label][key]>0]
            stats[label][key]=bootstrap(ratios) if len(ratios)==len(groups) else dict(evidence='NOT_DEFINED_ZERO_DENOMINATOR_OR_NUMERATOR')
    between={key:bootstrap([g['metrics']['layer'][key]/g['metrics']['colocated'][key] for g in groups])
        for key in KEYS if all(g['metrics']['colocated'][key]>0 and g['metrics']['layer'][key]>0 for g in groups)}
    all_outputs=all(x['passed'] for g in groups for x in g['output_checks'])
    decode_point=all(g['metrics']['layer']['Decode_absolute_SLO_pass'] and
        g['metrics']['layer']['ITL_p99_ms']<=1.05*g['metrics']['native']['ITL_p99_ms'] and
        g['metrics']['layer']['TPOT_p95_ms']<=1.05*g['metrics']['native']['TPOT_p95_ms'] for g in groups)
    decode_ci=all(stats['layer'][key].get('CI95',[0,float('inf')])[1]<=1.05 for key in ('ITL_p99_ms','TPOT_p95_ms'))
    write(ROOT/'analysis'/output_file,dict(groups=groups,paired_statistics=stats,
        layer_over_colocated=between,exact_output_pass=all_outputs,
        all_Native_absolute_Decode_SLO_pass=all(g['metrics']['native']['Decode_absolute_SLO_pass'] for g in groups),
        Decode_noninferiority_point_pass=decode_point,Decode_noninferiority_CI_pass=decode_ci,
        evidence='MEASURED_MARKER_FREE_PAIRED_STARTS',
        scope='Only arrival seeds are held out. All 128 prompt IDs are distinct, but 64 occurred in selection. Finite-cohort drain rates and goodput are not sustainable capacity.'))
    columns=['seed','arm','name',*KEYS,'count','distinct_prompt_ids','offered_QPS','realized_arrival_QPS',
        'actual_submit_QPS','TTFT_p50_ms','TTFT_p99_ms','E2E_p50_ms','E2E_p99_ms','ITL_p50_ms',
        'ITL_p95_ms','TPOT_p50_ms','TPOT_p99_ms','ITL_threshold_exceedances','ITL_intervals',
        'input_tokens_s','output_tokens_s','TTFT_attainment','TPOT_attainment',
        'Host_arrival_to_worker_accept_p95_ms','Service_TTFT','exact_execution_queue_latency']
    with (ROOT/'analysis'/csv_file).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for g in groups:
            for label,m in g['metrics'].items():writer.writerow(dict(seed=g['seed'],arm=label,name=m['name'],**{k:m[k] for k in columns[3:]}))


def diagnostics():
    rows=[]
    for label in ('guard','c35','c28'):
        name=f'eager_qwen8_L8000_O16_N32_{label}_event_diagnostic_v23';folder=ROOT/'raw'/name
        path=folder/'CUDA_event_enclosures_decode.json'
        if not path.exists():continue
        groups={};all_rows=json.loads(path.read_text())['rows']
        for r in all_rows:
            if not all(x.startswith('q') and x[1:].isdigit() for x in r['requests']):continue
            key=(r['prefilling'],len(r['requests']))
            groups.setdefault(key,[]).append(r)
        summary=[]
        for (prefilling,batch),values in groups.items():
            durations=[r['elapsed_ms'] for r in values];intervals=[];gaps=[]
            ordered=sorted(values,key=lambda r:r['start_monotonic'])
            for a,b in zip(ordered,ordered[1:]):
                if a['requests']==b['requests']:
                    intervals.append(1000*(b['start_monotonic']-a['start_monotonic']))
                    gaps.append(1000*(b['start_monotonic']-a['end_monotonic']))
            summary.append(dict(prefilling=prefilling,batch=batch,count=len(values),
                event_enclosure_p50_ms=percentile(durations,.5),event_enclosure_p95_ms=percentile(durations,.95),
                consecutive_submit_interval_p50_ms=percentile(intervals,.5),inter_enclosure_gap_p50_ms=percentile(gaps,.5)))
        rows.append(dict(name=name,metrics=metrics(name),grouped_events=summary,work=work(name),
            scope='Stream-event enclosures include launch gaps and waits. Inter-enclosure gaps are not physical GPU idle. This diagnostic does not isolate HBM/SM contention.'))
    write(ROOT/'analysis/Decode_event_diagnostic.json',dict(rows=rows,evidence='DIAGNOSTIC_CUDA_EVENT_AND_HOST_TRACE',
        exclusive_kernel_resource_contention='NOT_IDENTIFIED',exclusive_host_scheduling_contribution='NOT_IDENTIFIED'))


def main():
    preserve();inventory();pilot();confirmation();diagnostics()
    confirmation('graph_confirmation_plan_v27.json','graph_N128_confirmation.json','graph_N128_serving_results.csv')
    print('ANALYSIS_UPDATED',time.time())


if __name__=='__main__':main()
