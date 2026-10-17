"""Correlate actual launch scopes with GPU intervals. No clean-time scaling."""
import collections
import gzip
import re
import sqlite3
from poc_recurrent_asymmetry.contract import *


def partition(events):
    if not events:return {}
    changes=[]
    for i,row in enumerate(events):
        changes.extend([(row['start'],1,i),(row['end'],-1,i)])
    changes.sort();active=set();last=changes[0][0];result=collections.defaultdict(float)
    for now,action,i in changes:
        if now>last:
            categories={events[j]['category'] for j in active}
            name='GPU_IDLE_OR_UNRESOLVED_WAIT' if not categories else next(iter(categories)) if len(categories)==1 else 'MULTICATEGORY_OVERLAP'
            result[name]+=now-last
        if action==1:active.add(i)
        else:active.discard(i)
        last=now
    assert abs(sum(result.values())-(changes[-1][0]-changes[0][0]))<1e-6
    return dict(result)


def main():
    source=ROOT/'outputs/nsys_v1/capture.sqlite';output=ROOT/'metrics/nsight';output.mkdir(exist_ok=False)
    db=sqlite3.connect(source);db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    spans=collections.defaultdict(list)
    for row in db.execute('select * from NVTX_EVENTS where end is not null'):
        row=dict(row);name=row['text'] or strings.get(row['textId'],'')
        if name.startswith('ASYM_'):spans[row['globalTid']].append(dict(start=row['start'],end=row['end'],name=name))
    apis=collections.defaultdict(list)
    for row in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        row=dict(row)
        if row['globalTid'] in spans:apis[row['globalTid']].append(row)
    launches={}
    for tid,values in apis.items():
        ranges=sorted(spans[tid],key=lambda r:r['start']);index=0;active=[]
        for api in sorted(values,key=lambda r:r['start']):
            active=[r for r in active if r['end']>=api['start']]
            while index<len(ranges) and ranges[index]['start']<=api['start']:
                if ranges[index]['end']>=api['start']:active.append(ranges[index])
                index+=1
            enclosing=[r['name'] for r in active if r['end']>=api['end']]
            request=next((r for r in enclosing if r.startswith('ASYM_REQUEST|')),None)
            if not request:continue
            component=next((r for r in enclosing if r.startswith('ASYM_COMP|')),None)
            phase=next((r for r in enclosing if r.startswith('ASYM_PHASE|')),'ASYM_PHASE|OTHER')
            launches[(tid//(2**24)*(2**24),api['correlationId'])]=dict(request=request,phase=phase,
                category=component.split('|')[1] if component else 'OTHER_KERNEL',
                round=int(re.search(r'\|r=(-?\d+)',component).group(1)) if component else None,
                layer=int(re.search(r'\|l=(-?\d+)',component).group(1)) if component else None,
                launch_start=api['start']/1e6,launch_end=api['end']/1e6)
    events=[]
    for row in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        row=dict(row);tag=launches.get((row['globalPid'],row['correlationId']))
        if tag is None:continue
        events.append(dict(**tag,start=row['start']/1e6,end=row['end']/1e6,device=row['deviceId'],
            stream=row['streamId'],name=strings[row['shortName']],kind='KERNEL'))
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for row in db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
            row=dict(row);tag=launches.get((row['globalPid'],row['correlationId']))
            if tag is None:continue
            events.append(dict(**tag,start=row['start']/1e6,end=row['end']/1e6,device=row['deviceId'],
                stream=row['streamId'],name='CUDA_MEMCPY',kind='MEMCPY',bytes=row['bytes']))
    with gzip.open(output/'events.json.gz','wt') as stream:json.dump(events,stream)
    groups=collections.defaultdict(list)
    for row in events:groups[(row['request'],row['phase'])].append(row)
    summaries=[]
    for (request,phase),selected in sorted(groups.items()):
        begin=min(r['start'] for r in selected);end=max(r['end'] for r in selected)
        exclusive=partition(selected)
        summaries.append(dict(request=request,phase=phase,activity_envelope_ms=end-begin,
            exclusive_ms=exclusive,inclusive_service_ms={category:sum(r['end']-r['start'] for r in selected if r['category']==category)
                for category in sorted({r['category'] for r in selected})},
            kernel_count=sum(r['kind']=='KERNEL' for r in selected),
            scope='Observed heavy-trace single-GPU activity envelope. Not a clean E2E percentage or a removable-cost claim.'))
    write(output/'summary.json',dict(rows=summaries,actual_events=len(events),source_sha256=sha(source),
        clock='One Nsight capture clock, physical GPU 4 only.',
        limitation='Profile overhead fails the 10% clean-timing gate. Mechanism and trace-local activity only.',
        no_last_replay_multiplication=True))
    write(output/'heavy_manifest.json',dict(sqlite=str(source),sqlite_sha256=sha(source),
        nsys_report=str(source.with_suffix('.nsys-rep')),nsys_sha256=sha(source.with_suffix('.nsys-rep'))))
    print('ACTUAL_TRACE_EVENTS',len(events),'phase groups',len(summaries))


if __name__=='__main__':main()
