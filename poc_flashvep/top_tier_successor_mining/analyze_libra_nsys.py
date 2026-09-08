"""Native Libra diagnostic: same-process/device intervals, launch-associated NVTX.

This is not a clean speed benchmark. GPU intervals are not attributed to CPU
ranges by coincident execution time: stage attribution uses the launch API's
correlation id and thread. Unknown launches remain unclassified.
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def duration(intervals):
    return sum(e-s for s,e in union(intervals))/1e6


def intersect(left,right):
    left=union(left);right=union(right);out=[];i=j=0
    while i<len(left) and j<len(right):
        s=max(left[i][0],right[j][0]);e=min(left[i][1],right[j][1])
        if e>s:out.append((s,e))
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(f'file:{a.sqlite}?mode=ro',uri=True);c.row_factory=sqlite3.Row
    tables={x[0] for x in c.execute("select name from sqlite_master where type='table'")}
    schema={t:[r[1] for r in c.execute(f'pragma table_info("{t}")')] for t in tables}
    (a.output/'schema.json').write_text(json.dumps(schema,indent=2))
    names={r['id']:r['value'] for r in c.execute('select * from StringIds')}
    def rows(table):return [dict(x) for x in c.execute(f'select * from "{table}"')]
    activity=[]
    for table in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY']:
        assert table in tables,table
        for x in rows(table):
            assert all(k in x for k in ['start','end','globalPid','deviceId','streamId']),table
            x['kind']='KERNEL' if table.endswith('KERNEL') else 'MEMCPY'
            x['name']=names.get(x.get('demangledName',x.get('shortName')),'MEMCPY')
            activity.append(x)
    pids={x['globalPid'] for x in activity}
    # Nsight global thread id encodes the process id above the low24 bits.
    def pid(tid):
        candidate=(int(tid)>>24)<<24
        return candidate if candidate in pids else None
    nvtx=[]
    for x in rows('NVTX_EVENTS'):
        text=x.get('text') or names.get(x.get('textId'))
        if x.get('end') is None or not text:continue
        x.update(text=text,pid=pid(x['globalTid']))
        if x['pid'] is not None:nvtx.append(x)
    pattern=re.compile(r'successor_native\|rank=(\d+)\|case=(\d+)\|rep=(-?\d+)\|policy=(\w+)')
    outer=[]
    for x in nvtx:
        match=pattern.fullmatch(x['text'])
        if match and int(match[3])>=0:
            outer.append(dict(x,rank=int(match[1]),case=int(match[2]),rep=int(match[3]),policy=match[4]))
    assert len(outer)==24,('expected4 ranks*3 reps*2 policies',len(outer))
    launches=collections.defaultdict(list)
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if table not in tables:continue
        for x in rows(table):
            if 'correlationId' in x and 'globalTid' in x:
                launches[(pid(x['globalTid']),x['correlationId'])].append(x)
    bypid=collections.defaultdict(list)
    for x in activity:bypid[x['globalPid']].append(x)
    summaries=[];phases=[];kernels=[]
    for o in outer:
        aa=[x for x in bypid[o['pid']] if o['start']<=x['start'] and x['end']<=o['end']]
        assert aa,(o['text'],'no activity for process mapping')
        devices={x['deviceId'] for x in aa};assert len(devices)==1,devices
        stages=[x for x in nvtx if x['pid']==o['pid'] and o['start']<=x['start']
                and x['end']<=o['end'] and not pattern.fullmatch(x['text'])]
        bythread=collections.defaultdict(list)
        for stage in stages:bythread[stage['globalTid']].append(stage)
        for x in aa:
            candidates=[]
            for launch in launches.get((o['pid'],x.get('correlationId')),[]):
                if not o['start']<=launch['start']<=o['end']:continue
                for stage in bythread[launch['globalTid']]:
                    if stage['start']<=launch['start'] and launch['end']<=stage['end']:
                        candidates.append((stage['end']-stage['start'],stage['text']))
            x['stage']=min(candidates)[1] if candidates else 'UNATTRIBUTED'
        span=(o['end']-o['start'])/1e6
        intervals=[(x['start'],x['end']) for x in aa]
        expert=[(x['start'],x['end']) for x in aa if x['kind']=='KERNEL' and x['stage'].startswith('MoE')]
        copies=[(x['start'],x['end']) for x in aa if x['kind']=='MEMCPY']
        comm=[(x['start'],x['end']) for x in aa if x['kind']=='KERNEL' and 'nccl' in x['name'].lower()]
        merged=union(intervals);boundaries=[o['start']]+[v for pair in merged for v in pair]+[o['end']]
        gaps=[boundaries[i+1]-boundaries[i] for i in range(0,len(boundaries)-1,2)]
        identity={k:o[k] for k in ['rank','case','rep','policy']}
        summaries.append(dict(identity,device_id=next(iter(devices)),global_pid=o['pid'],
            profiled_host_span_ms=span,gpu_activity_union_ms=duration(intervals),
            no_observed_gpu_activity_ms=span-duration(intervals),largest_activity_gap_ms=max(gaps)/1e6,
            expert_kernel_union_ms=duration(expert),memcpy_union_ms=duration(copies),
            nccl_kernel_union_ms=duration(comm),copy_expert_overlap_ms=duration(intersect(copies,expert)),
            nccl_expert_overlap_ms=duration(intersect(comm,expert)),
            unattributed_activity_count=sum(x['stage']=='UNATTRIBUTED' for x in aa),activity_count=len(aa)))
        for stage in sorted({x['stage'] for x in aa}|{x['text'] for x in stages}):
            selected=[x for x in aa if x['stage']==stage]
            phases.append(dict(identity,stage=stage,
                cpu_nvtx_union_ms=duration([(x['start'],x['end']) for x in stages if x['text']==stage]),
                gpu_activity_union_ms=duration([(x['start'],x['end']) for x in selected]),
                gpu_kernel_sum_ms=sum((x['end']-x['start'])/1e6 for x in selected if x['kind']=='KERNEL'),
                memcpy_sum_ms=sum((x['end']-x['start'])/1e6 for x in selected if x['kind']=='MEMCPY'),
                activity_count=len(selected)))
        for x in aa:kernels.append(dict(identity,stage=x['stage'],kind=x['kind'],name=x['name'],
            duration_ms=(x['end']-x['start'])/1e6,stream=x['streamId']))
    df=pd.DataFrame(summaries);df.to_csv(a.output/'invocation_intervals.csv',index=False)
    pd.DataFrame(phases).to_csv(a.output/'phase_intervals.csv',index=False)
    pd.DataFrame(kernels).groupby(['policy','stage','kind','name'],dropna=False).duration_ms.agg(
        ['count','sum','median','max']).reset_index().to_csv(a.output/'activity_breakdown.csv',index=False)
    metrics=[x for x in df if x.endswith('_ms')]
    result=dict(scope='PROFILED_NATIVE_TEXT_PREFILL_DIAGNOSTIC_NOT_REQUEST_E2E',
        mapping='same process and device; CPU launch correlation id to innermost NVTX stage',
        observations=len(df),coupled_executions=6,
        median_by_policy=df.groupby('policy')[metrics].median().to_dict(orient='index'),
        caveats=['Profiled CPU/control overhead may be substantial; use unprofiled runs for speed.',
                 'No-activity intervals are not automatically causally attributable to CPU planning.',
                 'Copy/communication overlap is time overlap,not measured throughput or SM efficiency.',
                 'No cross-device CUDA-event timestamp subtraction or summed rank latency.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
