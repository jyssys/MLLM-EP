"""Nsight activity unions and cross-request overlap. No E2E percentage inference."""
import collections
import csv
import gzip
import re
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_flashvep.ouro_request_pp.analyze import union
from .contract import *


STAGE=re.compile(r'^EIGHT\|([A-Z0-9]+)\|([^|]+)\|step(\d+)\|R(\d+)\|L(\d+)-(\d+)$')


def parse(file):
    db=sqlite3.connect(file);db.row_factory=sqlite3.Row
    command=read(ROOT/'commands'/f'{file.stem}.json')
    physical=list(command['GPUs']);assert set(physical)<={4,5,6,7}
    device_map={rank:gpu for rank,gpu in enumerate(physical)}
    inventory=[dict(r) for r in db.execute('SELECT * FROM TARGET_INFO_GPU')]
    recorded_UUIDs={str(r['uuid']).removeprefix('GPU-') for r in inventory}
    assert all(UUIDS[gpu-4] in recorded_UUIDs for gpu in physical)
    owner_checks=[]
    setup=ROOT/'raw'/file.stem/'setup.json'
    if setup.exists():
        for row in read(setup).get('ownership',[]):
            assert row['GPU_uuid'].removeprefix('GPU-')==UUIDS[device_map[row['rank']]-4]
            owner_checks.append(dict(rank=row['rank'],UUID=row['GPU_uuid']))
    for owner in (ROOT/'raw'/file.stem).glob('cohort_*_rank*.json'):
        record=read(owner);metadata=record.get('ownership')
        if metadata:
            assert metadata['UUID'].removeprefix('GPU-')==UUIDS[device_map[record['rank']]-4]
            owner_checks.append(dict(rank=record['rank'],UUID=metadata['UUID']))
    strings=dict(db.execute('SELECT id,value FROM StringIds'));windows=[];stages=collections.defaultdict(list);controllers=[]
    for row in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r=dict(row);name=r.get('text') or strings.get(r.get('textId'),'')
        if name=='EIGHT_COHORT':windows.append((r['start'],r['end']))
        if name.startswith('EIGHT_CONTROLLER|'):controllers.append(dict(start=r['start'],end=r['end'],name=name))
        match=STAGE.match(name)
        if match:
            arm,rid,step,loop,first,last=match.groups()
            rank=int(first)//6 if arm in ['L4','LD'] else int(loop)-1 if arm in ['O4','OD'] else 0
            stages[rank].append(dict(request=rid,step=int(step),loop=int(loop),start=r['start'],end=r['end']))
    assert windows,'No serving window. Do not profile graph construction as inference.'
    left=min(a for a,b in windows);right=max(b for a,b in windows);kernels=[];copies=[]
    for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        r=dict(row)
        if r['start']<left or r['end']>right:continue
        rank=r['deviceId'];assert rank in device_map,(file,rank,device_map)
        name=strings.get(r.get('shortName'),'')
        tag=next((s for s in stages[rank] if s['start']<=r['start'] and r['end']<=s['end']),None)
        kernels.append(dict(rank=rank,start=r['start'],end=r['end'],name=name,NCCL='nccl' in name.lower(),
            request=tag['request'] if tag else None,step=tag['step'] if tag else None,loop=tag['loop'] if tag else None))
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            r=dict(row)
            if r['start']<left or r['end']>right:continue
            copies.append(dict(rank=r['deviceId'],start=r['start'],end=r['end'],bytes=r['bytes'],kind=r['copyKind'],
                source=r.get('srcDeviceId'),destination=r.get('dstDeviceId')))
    ranks=[]
    for rank in sorted({k['rank'] for k in kernels}):
        ks=[k for k in kernels if k['rank']==rank];cs=[c for c in copies if c['rank']==rank];nccl=[k for k in ks if k['NCCL']]
        busy=union([(k['start'],k['end']) for k in ks+cs])
        ranks.append(dict(rank=rank,physical_GPU=device_map[rank],kernel_count=len(ks),kernel_service_ms=sum(k['end']-k['start'] for k in ks)/1e6,
            GPU_activity_union_ms=busy/1e6,no_recorded_activity_ms=(right-left-busy)/1e6,
            NCCL_kernel_count=len(nccl),NCCL_kernel_service_ms=sum(k['end']-k['start'] for k in nccl)/1e6))
    edges=[]
    for i,k in enumerate(kernels):
        if k['request'] is not None:edges.extend([(k['start'],1,i),(k['end'],-1,i)])
    active=set();before=left;overlap=0;maximum=0
    for now,kind,index in sorted(edges):
        distinct_ranks={kernels[i]['rank'] for i in active};distinct_requests={kernels[i]['request'] for i in active}
        maximum=max(maximum,len(distinct_ranks))
        if len(distinct_ranks)>1 and len(distinct_requests)>1:overlap+=now-before
        if kind==1:active.add(index)
        else:active.remove(index)
        before=now
    result=dict(label=file.stem,source_sha256=sha(file),window_ms=(right-left)/1e6,ranks=ranks,
        CUDA_logical_to_physical=device_map,UUID_owner_checks=owner_checks,
        mapping_rule='CUPTI deviceId is CUDA-visible ordinal. The launch mask and owner UUIDs define physical placement. TARGET_INFO_GPU.id is the system inventory index, not this CUDA ordinal.',
        different_request_actual_kernel_overlap_ms=overlap/1e6,max_simultaneous_tagged_kernel_ranks=maximum,
        NCCL_names=dict(collections.Counter(k['name'] for k in kernels if k['NCCL'])),
        controller_NVTX_ms=sum(r['end']-r['start'] for r in controllers)/1e6 if controllers else None,
        lane='HEAVY_TRACE. CUDA service sums and no-recorded-activity are not a non-overlapped E2E breakdown.')
    APIs=collections.defaultdict(lambda:dict(count=0,host_service_ms=0.))
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
            r=dict(row)
            if r['start']<left or r['end']>right:continue
            name=strings.get(r.get('nameId'),'')
            if any(s in name for s in ['cudaGraphLaunch','Synchronize','cudaMemcpy','cudaStreamWaitEvent']):
                APIs[name]['count']+=1;APIs[name]['host_service_ms']+=(r['end']-r['start'])/1e6
    result['CUDA_API_service']=dict(APIs)
    directory=ROOT/'timelines'/file.stem;directory.mkdir(exist_ok=True)
    write(directory/'summary.json',result)
    with gzip.open(directory/'gpu_events.json.gz','wt') as f:json.dump(dict(kernels=kernels,copies=copies,controller_NVTX=controllers),f,separators=(',',':'))
    if ranks:
        with (directory/'gpu_activity.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(ranks[0]),lineterminator='\n');w.writeheader();w.writerows(ranks)
    fig,ax=plt.subplots(figsize=(12,4));colors=plt.get_cmap('tab10')
    requests=sorted({k['request'] for k in kernels if k['request'] is not None})
    for rank in sorted({k['rank'] for k in kernels}):
        for rid in [None]+requests:
            points=[((k['start']-left)/1e6,(k['end']-k['start'])/1e6) for k in kernels if k['rank']==rank and k['request']==rid]
            if points:ax.broken_barh(points,(rank-.35,.7),facecolors='gray' if rid is None else colors(requests.index(rid)%10))
    observed=sorted({k['rank'] for k in kernels})
    ax.set(yticks=observed,yticklabels=[f'GPU{device_map[r]}' for r in observed],xlabel='Profiler clock (ms)',title=file.stem+' — actual GPU kernels')
    fig.tight_layout();fig.savefig(directory/'actual_kernel_timeline.png',dpi=160);plt.close(fig)
    print('TIMELINE',file.stem,result['different_request_actual_kernel_overlap_ms'],flush=True)


if __name__=='__main__':
    import sys
    for arg in sys.argv[1:]:parse(Path(arg))
