"""Correlate actual GPU kernels with their synchronous publication ranges."""
import argparse
import collections
import gzip
import json
import re
import sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .contract import ROOT,write,sha
from .analyze import union


STAGE=re.compile(r'^(layer_pp4|loop_pp4)\|(request-\d+)\|step(\d+)\|R(\d+)\|L(\d+)-(\d+)$')


def parse(source):
    source=source.resolve()
    db=sqlite3.connect(source);db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    ranges=[];window=None
    for rr in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r=dict(rr);name=r.get('text') or strings.get(r.get('textId'),'')
        if name=='MEASURED_WORKLOAD':window=(r['start'],r['end'])
        match=STAGE.match(name)
        if match:
            arm,rid,step,loop,first,last=match.groups()
            rank=int(first)//6 if arm=='layer_pp4' else int(loop)-1
            ranges.append(dict(start=r['start'],end=r['end'],rank=rank,id=rid,step=int(step),loop=int(loop),tid=r['globalTid']))
    assert window and ranges, 'No measured workload or worker NVTX ranges.'
    stages=collections.defaultdict(list)
    for r in ranges:stages[r['rank']].append(r)
    for group in stages.values():group.sort(key=lambda r:r['start'])
    kernels=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')
             if r['end']>window[0] and r['start']<window[1]]
    devices={r['deviceId'] for r in kernels}
    # The process validates the UUID of every visible device before profiling.
    physical_ids=bool(devices and min(devices)>=4)
    events=[]
    for r in kernels:
        rank=r['deviceId']-4 if physical_ids else r['deviceId']
        assert rank in [0,1,2,3]
        candidates=[s for s in stages[rank] if s['start']<=r['start'] and s['end']>=r['end']]
        tag=candidates[0] if len(candidates)==1 else None
        events.append(dict(start_ns=r['start'],end_ns=r['end'],rank=rank,kind='kernel',
                           name=strings.get(r.get('shortName'),strings.get(r.get('demangledName'),'')),
                           request=tag['id'] if tag else 'UNRESOLVED',step=tag['step'] if tag else None))
    copies=[]
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for rr in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            r=dict(rr)
            if r['end']<=window[0] or r['start']>=window[1]:continue
            rank=r['deviceId']-4 if physical_ids else r['deviceId']
            if rank not in range(4):continue
            copies.append(dict(start_ns=r['start'],end_ns=r['end'],rank=rank,kind='copy',
                bytes=r['bytes'],copyKind=r.get('copyKind'),srcDeviceId=r.get('srcDeviceId'),dstDeviceId=r.get('dstDeviceId'),
                srcKind=r.get('srcKind'),dstKind=r.get('dstKind')))
    duration=(window[1]-window[0])/1e9
    copy_categories=[]
    for kind in sorted({e['copyKind'] for e in copies}):
        cs=[e for e in copies if e['copyKind']==kind]
        copy_categories.append(dict(copyKind=kind,count=len(cs),bytes=sum(e['bytes'] for e in cs),
            service_ms=sum(e['end_ns']-e['start_ns'] for e in cs)/1e6))
    peer_edges=[]
    for src,dst in sorted({(e['srcDeviceId'],e['dstDeviceId']) for e in copies if e['copyKind']==10}):
        cs=[e for e in copies if e['copyKind']==10 and e['srcDeviceId']==src and e['dstDeviceId']==dst]
        peer_edges.append(dict(source=src,destination=dst,count=len(cs),bytes=sum(e['bytes'] for e in cs),
            service_ms=sum(e['end_ns']-e['start_ns'] for e in cs)/1e6))
    runtime_calls=collections.defaultdict(list)
    for rr in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        r=dict(rr)
        if r['start']<window[0] or r['end']>window[1]:continue
        name=strings.get(r['nameId'],'')
        runtime_calls[name].append((r['end']-r['start'])/1e6)
    runtime_summary=[dict(name=k,count=len(v),host_service_ms=sum(v),host_P50_ms=sorted(v)[len(v)//2])
                     for k,v in runtime_calls.items()]
    rank_rows=[]
    for rank in range(4):
        ks=[e for e in events if e['rank']==rank];cs=[e for e in copies if e['rank']==rank]
        active=union([(max(e['start_ns'],window[0]),min(e['end_ns'],window[1])) for e in ks+cs])
        ku=union([(e['start_ns'],e['end_ns']) for e in ks])
        rank_rows.append(dict(rank=rank,physical_GPU=rank+4,kernel_count=len(ks),
            kernel_service_ms=sum(e['end_ns']-e['start_ns'] for e in ks)/1e6,
            kernel_union_ms=ku/1e6,recorded_activity_union_ms=active/1e6,
            no_recorded_activity_fraction=1-active/(window[1]-window[0]),
            copy_count=len(cs),copy_bytes=sum(c['bytes'] for c in cs)))
    # Sweep actual kernels, not stage brackets. Count different request IDs explicitly.
    changes=[]
    for i,e in enumerate(events):
        if e['request']=='UNRESOLVED':continue
        changes.extend([(e['start_ns'],1,i),(e['end_ns'],-1,i)])
    changes.sort();active=set();previous=changes[0][0] if changes else 0
    overlap_ns=0;max_ranks=0;example=None
    for now,kind,i in changes:
        ranks={events[j]['rank'] for j in active};ids={events[j]['request'] for j in active}
        if len(ranks)>=2 and len(ids)>=2 and now>previous:
            overlap_ns+=now-previous
            if example is None or now-previous>example['interval_ns']:
                example=dict(interval_ns=now-previous,ranks=sorted(ranks),requests=sorted(ids))
        max_ranks=max(max_ranks,len(ranks))
        if kind==1:active.add(i)
        else:active.discard(i)
        previous=now
    enums={}
    for name in tables:
        if 'MEMCPY' in name and name.startswith('ENUM_'):
            enums[name]=[dict(r) for r in db.execute(f'SELECT * FROM "{name}"')]
    output=source.with_suffix('');output.mkdir(exist_ok=True)
    with gzip.open(output/'actual_events.json.gz','wt') as f:json.dump(dict(kernels=events,copies=copies),f)
    summary=dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),
        evidence='NSIGHT_ACTIVITY',window_s=duration,raw_device_ids=sorted(devices),
        device_mapping='Raw physical IDs' if physical_ids else 'Visible logical IDs, validated UUID mapping to physical 4/5/6/7',
        ranks=rank_rows,different_request_kernel_overlap_ms=overlap_ns/1e6,
        max_simultaneous_kernel_ranks=max_ranks,example=example,
        tagged_kernels=sum(e['request']!='UNRESOLVED' for e in events),total_kernels=len(events),
        copy_categories=copy_categories,peer_edges=peer_edges,host_CUDA_API=runtime_summary,
        copy_enums=enums,limitations='Heavy-lane activity only. Not clean latency or a GPU utilization percentage.')
    write(output/'summary.json',summary)
    fig,ax=plt.subplots(figsize=(13,4))
    cmap=plt.get_cmap('tab10')
    for rank in range(4):
        for rid in sorted({e['request'] for e in events if e['rank']==rank}):
            selected=[e for e in events if e['rank']==rank and e['request']==rid]
            color='grey' if rid=='UNRESOLVED' else cmap(int(rid.split('-')[-1])%10)
            ax.broken_barh([((e['start_ns']-window[0])/1e6,(e['end_ns']-e['start_ns'])/1e6) for e in selected],
                          (rank-.35,.7),facecolors=color)
    ax.set(yticks=range(4),yticklabels=['GPU4','GPU5','GPU6','GPU7'],xlabel='Profiler time from workload range (ms)',
           title=source.stem+' — actual GPU kernels, request colors')
    fig.tight_layout();fig.savefig(output/'kernel_gantt.png',dpi=170);plt.close(fig)
    db.close()
    print('PHYSICAL_OVERLAP',source.stem,overlap_ns/1e6,'ms',max_ranks,'ranks',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('sources',nargs='+',type=Path);args=parser.parse_args()
    for path in args.sources:parse(path)


if __name__=='__main__':main()
