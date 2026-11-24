"""Reconstruct historical Main400 metrics without writing into old evidence."""
import argparse
import csv
import json
import math
import time
import numpy as np
from .common import ROOT,V2,initialize,preserve,write,digest


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def intervals(events,n):
    assert n>=1 and events and events[-1]['count']==n
    result=[];previous_count=0;previous_time=events[0]['time']
    for event in events:
        assert event['count']>previous_count and event['time']>=previous_time
        amount=event['count']-previous_count
        result.extend(([1000*(event['time']-previous_time)]+[0.]*(amount-1))
            if previous_count else [0.]*(amount-1))
        previous_count,previous_time=event['count'],event['time']
    assert len(result)==n-1
    return result


def pct(values,q):return float(np.percentile(values,q)) if values else None


def frozen_gates(ttft,tpot,slots):
    return ttft<=5000,all(value<=34 for value in slots),tpot is None or tpot<=33


def reconstruct(folder):
    cfg=json.loads((folder/'configuration.json').read_text())
    rows=records(folder/'requests.jsonl');window=json.loads((folder/'window.json').read_text())
    seconds=window['end']-window['start'];assert seconds>0 and len(rows)==cfg['count']
    assert len({row['sample_id'] for row in rows})==len(rows)
    ttft=[];tpots=[];itls=[];passed=[];requests=[]
    for row in rows:
        n=row['output_tokens'];assert n==len(row['output_ids']) and n>=1
        event=row['token_events'][0];a=1000*(event['time']-row['arrival'])
        b=1000*(row['finish']-event['time'])/(n-1) if n>1 else None
        assert math.isclose(a,row['TTFT_ms'],abs_tol=1e-6)
        assert (b is None and row['TPOT_ms'] is None) or math.isclose(b,row['TPOT_ms'],abs_tol=1e-6)
        slots=intervals(row['token_events'],n);itls.extend(slots);ttft.append(a)
        if b is not None:tpots.append(b)
        p=frozen_gates(a,b,slots)
        passed.append(p)
        requests.append(dict(sample_id=row['sample_id'],prompt_id=row['prompt_id'],task=row.get('task'),
            input_tokens=row['input_tokens'],output_tokens=n,arrival=row['arrival'],finish=row['finish'],
            finish_reason=row.get('finish_reason','NOT_INSTRUMENTED'),cap_hit=row.get('cap_hit','NOT_INSTRUMENTED'),
            TTFT_ms=a,TPOT_ms=b,maximum_ITL_ms=max(slots,default=0),TTFT_PASS=p[0],ITL_PASS=p[1],TPOT_PASS=p[2],
            JOINT_PASS=all(p),ITL_count=len(slots),ITL_intervals_ms=slots))
    output=sum(row['output_tokens'] for row in rows)
    joint=[all(p) for p in passed]
    value=dict(n=len(rows),TTFT_mean_ms=float(np.mean(ttft)),TTFT_p50_ms=pct(ttft,50),TTFT_p90_ms=pct(ttft,90),
        TTFT_p95_ms=pct(ttft,95),TTFT_p99_ms=pct(ttft,99),mean_TBT_ms=float(np.mean(itls)) if itls else None,
        ITL_p95_ms=pct(itls,95),ITL_p99_ms=pct(itls,99),max_ITL_ms=max(itls,default=0),
        TPOT_mean_ms=float(np.mean(tpots)) if tpots else None,TPOT_p95_ms=pct(tpots,95),
        output_tokens=output,measurement_seconds=seconds,generated_output_tokens_per_second=output/seconds,
        completed_requests_per_second=len(rows)/seconds,SLO_qualified_output_tokens_per_second=sum(
            row['output_tokens'] for row,p in zip(rows,joint) if p)/seconds,
        TTFT_SLO_attainment=float(np.mean([p[0] for p in passed])),
        every_ITL_SLO_attainment=float(np.mean([p[1] for p in passed])),
        TPOT_SLO_attainment=float(np.mean([p[2] for p in passed])),Joint_SLO_attainment=float(np.mean(joint)))
    return cfg,value,requests


EXPECTED={
    'Agg':(1477.006,16.387,277.839,141.579,118.693,37.482,.5200),
    'NativeStrong':(8161.605,7.872,18.723,8.968,109.402,70.139,.7350),
    'StockToken':(11483.980,7.525,9.478,8.438,114.035,69.358,.6075),
    'Layer':(11679.085,7.219,8.538,7.955,112.677,49.029,.4642),
    'Hybrid':(11767.609,7.272,8.572,8.050,105.978,49.802,.4650)}
FIELDS=('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second',
    'SLO_qualified_output_tokens_per_second','Joint_SLO_attainment')


def analyze(symptom=False,minimal=False):
    initialize();preserve()
    path=V2/'figures/c3_main_serving_v2_per_start.csv'
    published=list(csv.DictReader(path.open()))
    selected=[row for row in published if float(row['QPS'])==2 and row['label'] in EXPECTED]
    assert len(selected)==15
    starts=[];source_files={str(path):digest(path)}
    for item in selected:
        folder=V2/'raw'/item['name'];cfg,value,requests=reconstruct(folder)
        assert cfg['count']==400 and not cfg['ignore_eos'] and cfg['qps']==2 and cfg['batch_invariant'] is False
        for field in FIELDS:
            assert math.isclose(value[field],float(item[field]),rel_tol=1e-10,abs_tol=1e-6),(item['name'],field)
        for filename in ('configuration.json','requests.jsonl','window.json'):
            source_files[str(folder/filename)]=digest(folder/filename)
        starts.append(dict(label=item['label'],start=int(item['start']),seed=cfg['seed'],name=item['name'],
            GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],metrics=value))
    summary=[]
    for label,expected in EXPECTED.items():
        group=[row for row in starts if row['label']==label];assert len(group)==3
        means={field:float(np.mean([row['metrics'][field] for row in group])) for field in FIELDS}
        for field,target in zip(FIELDS,expected):
            tolerance=.000501 if field!='Joint_SLO_attainment' else .000051
            assert abs(means[field]-target)<=tolerance,(label,field,means[field],target)
        summary.append(dict(label=label,independent_starts=3,**means))
    result=dict(time=time.time(),passed=True,evidence='HISTORICAL_RAW_RECALCULATION',starts=starts,summary=summary,
        source_files_SHA256=source_files,scope='Reconstruct scheduled-arrival to coordinator-delivery metrics from immutable Main400 Natural N3. '
            'Compare each start before averaging. Do not pool requests into independent server starts. No new GPU measurement.')
    write(ROOT/'analysis/historical_main400_metric_reproduction.json',result)
    print('MAIN400_N3_RAW_REPRODUCTION_PASS',len(starts),'starts',flush=True)
    if symptom:
        lookup={row['label']:row for row in summary}
        targets=('StockToken','Layer','Hybrid')
        if minimal:
            lookup={row['label']:row['metrics'] for row in starts if row['start']==2 and row['label'] in ('NativeStrong','StockToken','Layer')}
            targets=('StockToken','Layer')
        observed=all(lookup[label]['TTFT_p95_ms']>lookup['NativeStrong']['TTFT_p95_ms'] for label in targets)
        observed=observed and all(lookup[label]['ITL_p99_ms']<lookup['StockToken']['ITL_p99_ms'] and
            lookup[label]['SLO_qualified_output_tokens_per_second']<lookup['StockToken']['SLO_qualified_output_tokens_per_second']
            for label in (('Layer',) if minimal else ('Layer','Hybrid')))
        write(ROOT/'analysis'/('minimal_symptom_reproduction.json' if minimal else 'symptom_reproduction.json'),
            dict(time=time.time(),symptom_reproduced=observed,expected_exit_code=1,methods=list(lookup),
                paired_starts=1 if minimal else 3,requests_per_start=400,
                scope='Frozen Main400 cannot be reduced by deleting selected requests without changing the tested workload and percentile. '
                    'The minimal observable contrast keeps three methods and one paired start. This expected red signal is not a correctness failure.'))
        assert not observed,'REPRODUCED_SYMPTOM: Native TTFT regression and better decode but worse split goodput'
    preserve();return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--symptom-test',action='store_true');parser.add_argument('--minimal',action='store_true');args=parser.parse_args()
    analyze(args.symptom_test,args.minimal)
