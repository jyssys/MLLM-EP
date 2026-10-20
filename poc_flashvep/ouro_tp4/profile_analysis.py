"""Analyze diagnostic kernels on the profiler clock, not the serving clock."""
import collections
import gzip
import json
import re
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from poc_flashvep.ouro_request_pp.analyze import union
from .contract import ROOT,write,sha

STAGE=re.compile(r'^MP_(layer_pp4|loop_pp4)\|(request-\d+)\|step(\d+)\|R(\d+)\|L(\d+)-(\d+)$')


def parse(file):
    db=sqlite3.connect(file);db.row_factory=sqlite3.Row
    strings=dict(db.execute('SELECT id,value FROM StringIds'));stages=collections.defaultdict(list);windows=[]
    for rr in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r=dict(rr);name=r.get('text') or strings.get(r.get('textId'),'')
        if name in ['MEASURED_WORKLOAD','TP_OPERATOR_DIAGNOSTIC']:windows.append((r['start'],r['end']))
        match=STAGE.match(name)
        if match:
            arm,rid,step,loop,first,last=match.groups();rank=int(first)//6 if arm=='layer_pp4' else int(loop)-1
            stages[rank].append(dict(id=rid,step=int(step),loop=int(loop),start=r['start'],end=r['end']))
    assert windows
    left=min(t[0] for t in windows);right=max(t[1] for t in windows)
    kernels=[]
    for rr in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        r=dict(rr)
        if r['start']<left or r['end']>right:continue
        rank=r['deviceId'];assert rank in range(4)
        name=strings.get(r.get('shortName'),'')
        tag=next((s for s in stages[rank] if s['start']<=r['start'] and s['end']>=r['end']),None)
        kernels.append(dict(rank=rank,start=r['start'],end=r['end'],name=name,
            request=tag['id'] if tag else None,step=tag['step'] if tag else None,loop=tag['loop'] if tag else None,
            NCCL='nccl' in name.lower()))
    copies=[]
    for rr in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
        r=dict(rr)
        if r['start']<left or r['end']>right:continue
        copies.append(dict(rank=r['deviceId'],start=r['start'],end=r['end'],bytes=r['bytes'],
            kind=r['copyKind'],source=r.get('srcDeviceId'),destination=r.get('dstDeviceId')))
    rank_rows=[]
    for rank in range(4):
        ks=[k for k in kernels if k['rank']==rank];cs=[c for c in copies if c['rank']==rank]
        active=union([(k['start'],k['end']) for k in ks+cs])
        comm=[k for k in ks if k['NCCL']]
        rank_rows.append(dict(rank=rank,physical_GPU=rank+4,kernel_count=len(ks),
            kernel_service_ms=sum(k['end']-k['start'] for k in ks)/1e6,
            NCCL_kernel_count=len(comm),NCCL_kernel_service_ms=sum(k['end']-k['start'] for k in comm)/1e6,
            activity_union_ms=active/1e6,no_recorded_activity_fraction=1-active/(right-left)))
    changes=[]
    for i,k in enumerate(kernels):
        if k['request'] is not None:changes.extend([(k['start'],1,i),(k['end'],-1,i)])
    changes.sort();active=set();previous=left;overlap=0;maxranks=0
    for now,kind,i in changes:
        ranks={kernels[j]['rank'] for j in active};requests={kernels[j]['request'] for j in active}
        if len(ranks)>1 and len(requests)>1:overlap+=now-previous
        maxranks=max(maxranks,len(ranks))
        if kind==1:active.add(i)
        else:active.remove(i)
        previous=now
    APIs=collections.defaultdict(lambda:dict(count=0,host_service_ms=0.))
    for rr in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        r=dict(rr)
        if r['start']<left or r['end']>right:continue
        name=strings.get(r.get('nameId'),'')
        if any(t in name for t in ['cudaGraphLaunch','cudaEventSynchronize','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpy']):
            APIs[name]['count']+=1;APIs[name]['host_service_ms']+=(r['end']-r['start'])/1e6
    summary=dict(source=str(file.relative_to(ROOT)),source_sha256=sha(file),
        evidence='DIAGNOSTIC_NSIGHT_ACTIVITY, not clean serving latency',window_ms=(right-left)/1e6,
        physical_logical_mapping='Profiler device 0/1/2/3 = validated physical GPU 4/5/6/7',ranks=rank_rows,
        NCCL_kernel_names=dict(collections.Counter(k['name'] for k in kernels if k['NCCL'])),
        different_request_kernel_overlap_ms=overlap/1e6,max_simultaneous_kernel_ranks=maxranks,
        tagged_kernels=sum(k['request'] is not None for k in kernels),kernels=len(kernels),
        copies=[dict(kind=kind,count=sum(c['kind']==kind for c in copies),
            bytes=sum(c['bytes'] for c in copies if c['kind']==kind),
            service_ms=sum(c['end']-c['start'] for c in copies if c['kind']==kind)/1e6)
            for kind in sorted({c['kind'] for c in copies})],
        CUDA_API_diagnostics=dict(APIs),
        limitation='Correctness/operator observer lane. CUDA service and inactivity do not establish an E2E critical-path decomposition.')
    directory=file.with_suffix('');directory.mkdir(exist_ok=True)
    write(directory/'summary.json',summary)
    with gzip.open(directory/'gpu_events.json.gz','wt') as f:json.dump(dict(kernels=kernels,copies=copies),f,separators=(',',':'))
    fig,ax=plt.subplots(figsize=(12,4));cmap=plt.get_cmap('tab10')
    for rank in range(4):
        selected=[k for k in kernels if k['rank']==rank]
        for rid in {k['request'] for k in selected}:
            kk=[k for k in selected if k['request']==rid]
            color='gray' if rid is None else cmap(int(rid.split('-')[1])%10)
            ax.broken_barh([((k['start']-left)/1e6,(k['end']-k['start'])/1e6) for k in kk],(rank-.35,.7),facecolors=color)
    ax.set(yticks=range(4),yticklabels=['GPU4','GPU5','GPU6','GPU7'],xlabel='Diagnostic profiler clock (ms)',title=file.stem+' — actual kernels')
    fig.tight_layout();fig.savefig(directory/'kernel_timeline.png',dpi=160);plt.close(fig)
    print('PROFILE_ANALYZED',file.stem,summary['different_request_kernel_overlap_ms'],flush=True)


def main():
    for file in sorted((ROOT/'profiles').glob('*.sqlite')):parse(file)


if __name__=='__main__':main()
