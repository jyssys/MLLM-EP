"""Audit closed HTTP cohorts from token timestamps, not from cached summaries."""
import argparse
from collections import Counter
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import gates
from .token_discovery_analysis import table

PREFIXES=('ec_NATIVE_C11_S','base_C14_','token_CONFIRM_C37_B','whole_DEFLECTION_C43R4_B',
    'interference_PROBE_C40R4_B','ec_INLINE_CAUSAL_C44_B')


def request_contract(row):
    assert row['status']=='MEASURED_ONLINE'
    times=row['token_times'];n=row['output_tokens']
    assert n==len(times)==len(row['token_ids'])==row['usage']['completion_tokens']>0
    assert all(b>=a for a,b in zip(times,times[1:]))
    assert row['scheduled_arrival']<=times[0]<=times[-1]<=row['finish']
    ttft=(times[0]-row['scheduled_arrival'])*1000
    intervals=[(b-a)*1000 for a,b in zip(times,times[1:])]
    tpot=(times[-1]-times[0])*1000/(n-1) if n>1 else 0.
    assert abs(ttft-row['TTFT_ms'])<1e-7 and abs(tpot-row['TPOT_ms'])<1e-7
    assert intervals==row['ITL_ms']
    assert abs((row['finish']-row['scheduled_arrival'])*1000-row['E2E_ms'])<1e-7
    ttft_pass=ttft<=5000;itl_pass=all(value<=34 for value in intervals);tpot_pass=tpot<=33
    failed=[name for name,value in [('TTFT',ttft_pass),('ITL',itl_pass),('TPOT',tpot_pass)] if not value]
    category='+'.join(failed)+'_FAIL' if failed else 'JOINT_PASS'
    expected=dict(TTFT_PASS=ttft_pass,ITL_PASS=itl_pass,TPOT_PASS=tpot_pass,
        JOINT_PASS=ttft_pass and itl_pass and tpot_pass,category=category)
    assert expected==gates(row)
    return expected


def cohort_contract(rows,done,reported):
    assert done['passed'] and not done['serial']
    assert len(rows)==done['actual_count']==done['planned_count']==reported['completed_count']
    assert len({r['runtime_request_id'] for r in rows})==len({r['request_id'] for r in rows})==len(rows)
    assert reported['failed_count']==reported['not_run_count']==0 and reported['primary_execution_valid']
    duration=max(r['finish'] for r in rows)-done['monotonic_start']
    assert abs(duration-done['duration_seconds'])<1e-8 and duration==reported['duration_seconds']
    gate=[request_contract(r) for r in rows];tokens=sum(r['output_tokens'] for r in rows)
    intervals=[v for r in rows for v in r['ITL_ms']]
    manual=dict(TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
        ITL_mean_ms=float(np.mean(intervals)) if intervals else None,
        ITL_max_ms=max(intervals) if intervals else None,total_output_tokens=tokens,
        raw_output_tokens_per_second=tokens/duration,completed_requests_per_second=len(rows)/duration,
        total_input_output_tokens_per_second=(sum(r['usage']['prompt_tokens'] for r in rows)+tokens)/duration,
        SLO_goodput=sum(r['output_tokens'] for r,g in zip(rows,gate) if g['JOINT_PASS'])/duration)
    for metric,values,percentiles in [('TTFT',[r['TTFT_ms'] for r in rows],[50,95,99]),
            ('ITL',intervals,[95,99]),('TPOT',[r['TPOT_ms'] for r in rows],[95])]:
        for percentile in percentiles:
            manual[f'{metric}_p{percentile}_ms']=float(np.percentile(values,percentile)) if values else None
    for key,value in manual.items():
        if value is None:assert reported[key] is None,key
        else:assert abs(reported[key]-value)<1e-8,(key,reported[key],value)
    categories=Counter(g['category'] for g in gate);category_tokens=Counter()
    for r,g in zip(rows,gate):category_tokens[g['category']]+=r['output_tokens']
    assert dict(categories)==reported['categories'] and dict(category_tokens)==reported['category_output_tokens']
    for key in ['TTFT_PASS','ITL_PASS','TPOT_PASS','JOINT_PASS']:
        assert sum(g[key] for g in gate)/len(rows)==reported['gate_attainment'][key]
    conjunction=sum(g['ITL_PASS'] and g['TPOT_PASS'] for g in gate)/len(rows)
    return dict(requests=len(rows),output_tokens=tokens,ITL_observations=len(intervals),
        actual_all_ITL_AND_TPOT_percent=100*conjunction,
        separate_ITL_percent=100*reported['gate_attainment']['ITL_PASS'],
        separate_TPOT_percent=100*reported['gate_attainment']['TPOT_PASS'],
        absolute_joint_decode_95_gate=conjunction>=.95,
        metric_recalculation='PASS',duration_seconds=duration)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True)
    parser.add_argument('--closed-only',action='store_true');args=parser.parse_args();preserve()
    assert args.closed_only, 'Explicit partial scope is required until final completion is audited'
    report=ROOT/'analysis'/f'campaign_metric_audit_{args.version}.json';assert not report.exists()
    results=[];incomplete=[]
    for raw in sorted((ROOT/'raw').glob('*_requests.jsonl')):
        run=raw.name.removesuffix('_requests.jsonl')
        if not run.startswith(PREFIXES) or any(part in run.lower() for part in ['warmup','smoke','correct']):continue
        completion=ROOT/'state'/f'{run}_complete.json';cached=ROOT/'analysis'/f'{run}_metrics.json'
        if not completion.exists() or not cached.exists():
            incomplete.append(dict(run=run,status='NOT_AUDITED_LIVE_OR_MISSING_COMPLETION'));continue
        done=json.loads(completion.read_text());reported=json.loads(cached.read_text())
        assert done['raw_SHA256']==reported['raw_SHA256']==digest(raw)
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        value=cohort_contract(rows,done,reported)
        results.append(dict(run=run,raw_SHA256=digest(raw),completion_SHA256=digest(completion),
            cached_metric_SHA256=digest(cached),**value))
    assert results
    table(ROOT/'analysis'/f'campaign_metric_audit_{args.version}.csv',results)
    write(report,dict(time=time.time(),status='CLOSED_COHORTS_RECALCULATED_NOT_FINAL_CLOSEOUT',
        cohorts=len(results),requests=sum(r['requests'] for r in results),results=results,incomplete=incomplete,
        frozen_SLO=dict(TTFT_ms=5000,every_client_ITL_ms=34,TPOT_ms=33),
        evidence='Metric integrity only. This audit does not prove output quality, profiler validity or safe windows.',
        absolute_decode='Per-request All-ITL AND TPOT. Separate95% pass rates alone are not this gate.',
        scope='No raw, cached metric or frozen report is overwritten. Live and absent cohorts are explicitly unaudited.'))
    preserve();print('CLOSED_COHORT_METRIC_AUDIT',len(results),sum(r['requests'] for r in results),flush=True)


if __name__=='__main__':main()
