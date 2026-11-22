"""Recalculate frozen client metrics from immutable request notifications."""
import argparse
import collections
import json
import math
import time
import numpy as np
from .common import ROOT,write,digest,preserve

def near(actual,expected,field):
    if actual is None or expected is None:assert actual is expected,(field,actual,expected)
    else:assert math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-6),(field,actual,expected)

def one(folder):
    cfg=json.loads((folder/'configuration.json').read_text());path=ROOT/'analysis'/(folder.name+'_metrics.json')
    if not path.exists():return dict(name=folder.name,status='NO_CLIENT_METRIC',scope='A teacher or diagnostic can omit client metrics.')
    m=json.loads(path.read_text());rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==cfg['count']==len({r['sample_id'] for r in rows})
    assert {r['sample_id'] for r in rows}==set(range(cfg['count']))
    all_itl=[];ttft=[];e2e=[];tpot=[];output=0;components=[];qualified_tokens=0
    for row in rows:
        events=row['token_events'];count=row['output_tokens'];assert count==len(row['output_ids'])>=1
        assert events and events[-1]['count']==count and all(a['count']<b['count'] for a,b in zip(events,events[1:]))
        assert all(a['time']<=b['time'] for a,b in zip(events,events[1:]))
        near(row['first_token'],events[0]['time'],'first delivery');near(row['finish'],events[-1]['time'],'last delivery')
        first=row['first_token'];end=row['finish'];arrival=row['arrival']
        near(row['TTFT_ms'],1000*(first-arrival),'request TTFT')
        near(row['E2E_ms'],1000*(end-arrival),'request E2E')
        value=1000*(end-first)/(count-1) if count>1 else None;near(row['TPOT_ms'],value,'request TPOT')
        intervals=[0.]*(events[0]['count']-1)
        for a,b in zip(events,events[1:]):intervals += [1000*(b['time']-a['time'])]+[0.]*(b['count']-a['count']-1)
        assert len(intervals)==count-1
        if cfg.get('ignore_eos'):
            expected=cfg['replay_output_lengths'][row['sample_id']] if cfg.get('replay_output_lengths') else cfg['output_cap']
            assert count==expected,(folder.name,row['sample_id'],count,expected)
        a=row['TTFT_ms']<=5000;b=all(x<=34 for x in intervals);c=value is None or value<=33
        components.append((a,b,c));qualified_tokens+=count if a and b and c else 0
        all_itl+=intervals;ttft.append(row['TTFT_ms']);e2e.append(row['E2E_ms']);output+=count
        if value is not None:tpot.append(value)
    window=json.loads((folder/'window.json').read_text());duration=window['end']-window['start'];assert duration>0
    pct=lambda values,p:float(np.percentile(values,p)) if values else None
    expected=dict(n=len(rows),output_tokens=output,ITL_interval_count=len(all_itl),
        measurement_seconds=duration,TTFT_mean_ms=float(np.mean(ttft)),TTFT_p50_ms=pct(ttft,50),
        TTFT_p95_ms=pct(ttft,95),TTFT_p99_ms=pct(ttft,99),E2E_p95_ms=pct(e2e,95),
        mean_TBT_ms=float(np.mean(all_itl)) if all_itl else None,ITL_p95_ms=pct(all_itl,95),ITL_p99_ms=pct(all_itl,99),
        max_ITL_ms=max(all_itl,default=0),TPOT_p50_ms=pct(tpot,50),TPOT_p95_ms=pct(tpot,95),TPOT_p99_ms=pct(tpot,99),
        generated_output_tokens_per_second=output/duration,completed_requests_per_second=len(rows)/duration,
        SLO_qualified_output_tokens_per_second=qualified_tokens/duration,
        TTFT_SLO_attainment=sum(a for a,b,c in components)/len(rows),every_ITL_SLO_attainment=sum(b for a,b,c in components)/len(rows),
        TPOT_SLO_attainment=sum(c for a,b,c in components)/len(rows),decode_SLO_attainment=sum(b and c for a,b,c in components)/len(rows),
        Joint_SLO_attainment=sum(a and b and c for a,b,c in components)/len(rows))
    for field,value in expected.items():near(m[field],value,field)
    for rpath in folder.glob('actual_runtime_*.json'):
        actual=json.loads(rpath.read_text())
        assert actual['scheduler_B']==8192 and actual['scheduler_S']==1024 and actual['model_layers']==36
        assert actual['parameter_dtypes']==['torch.bfloat16'] and actual['chunked_prefill']
    return dict(name=folder.name,status='PASS',requests=len(rows),output_tokens=output,client_intervals=len(all_itl),
        fields_recalculated=len(expected),metrics_SHA256=digest(path),requests_SHA256=digest(folder/'requests.jsonl'))

def run():
    preserve();results=[];status=collections.Counter()
    for receipt in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        r=json.loads(receipt.read_text());folder=receipt.parent
        if not r['complete']:
            value=dict(name=folder.name,status=r.get('status','FAILED_EXECUTION'),receipt_SHA256=digest(receipt),
                scope='Preserve this failed or unlaunched run. Do not promote it to a valid performance measurement.')
        else:
            try:value=one(folder)
            except Exception as error:value=dict(name=folder.name,status='VALIDATION_FAILED',error=repr(error))
        status[value['status']]+=1;results.append(value)
    passed=not status['VALIDATION_FAILED'];write(ROOT/'analysis/raw_metric_recalculation_v2.json',dict(time=time.time(),passed=passed,
        status_counts=dict(status),runs=results,
        scope='Independent arithmetic from unchanged client notifications. No metric or timestamp is repaired. '
            'Failed experiments remain in the inventory. A metric arithmetic pass does not replace state,quality,observer or decode gates.'))
    preserve();print('RAW_METRIC_RECALCULATION',dict(status),flush=True);assert passed
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run');a=p.parse_args()
    print(one(ROOT/'raw'/a.run)) if a.run else run()
