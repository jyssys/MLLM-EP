"""Interval unions and actual peer copies. Never add overlapping GPU durations."""
import argparse
import json
import sqlite3
from .common import ROOT,write


def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else:merged.append((start,end))
    return merged


def length(intervals):return sum(b-a for a,b in union(intervals))


def intersection(left,right):
    left,right=union(left),union(right)
    i=j=0;result=[]
    while i<len(left) and j<len(right):
        a,b=max(left[i][0],right[j][0]),min(left[i][1],right[j][1])
        if a<b:result.append((a,b))
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    folder=ROOT/'raw'/args.run
    contract=json.loads((folder/'serving_contract.json').read_text())
    db=sqlite3.connect('file:'+str(folder/'timeline.sqlite')+'?mode=ro',uri=True)
    start,end=db.execute("select start,end from NVTX_EVENTS where text='diagnostic_measured'").fetchone()
    kernels={}
    for device,a,b in db.execute('select deviceId,start,end from CUPTI_ACTIVITY_KIND_KERNEL'):
        if a<end and b>start:kernels.setdefault(device,[]).append((max(a,start),min(b,end)))
    memcpy=list(db.execute('select start,end,bytes,copyKind,srcDeviceId,dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY'))
    p2p=[r for r in memcpy if r[3]==10 and r[0]<end and r[1]>start]
    transfers=union([(max(r[0],start),min(r[1],end)) for r in p2p])
    summary=dict(run=args.run,range_ms=(end-start)/1e6,
        kernel_union_ms_by_visible_GPU={str(d):length(v)/1e6 for d,v in kernels.items()},
        P2P_GPU_copy_events=len(p2p),P2P_bytes=sum(r[2] for r in p2p),
        P2P_duration_sum_ms=sum(r[1]-r[0] for r in p2p)/1e6,P2P_union_ms=length(transfers)/1e6,
        P2P_kernel_overlap_ms={str(d):length(intersection(transfers,v))/1e6 for d,v in kernels.items()},
        evidence='MEASURED_NSYS_GPU_ACTIVITY_INSTRUMENTED_RUN',
        scope='NVTX measured region. Kernel unions and P2P unions can overlap. Do not add them as E2E components. Not a clean serving speedup.',
        contention=('All other GPUs were reserved empty. Check fabric_process_samples.jsonl for sampled validation.'
                    if contract.get('quiet') else 'Parallel numerical workers may use other GPU pairs. This pilot is not a quiet-lane serving comparison.'))
    summary['GPU_activity_partitions']={}
    for device,values in kernels.items():
        active=length(values);copy=length(transfers);both=length(intersection(values,transfers))
        pieces=dict(kernel_only_ms=(active-both)/1e6,P2P_only_ms=(copy-both)/1e6,
                    kernel_P2P_overlap_ms=both/1e6,
                    neither_recorded_kernel_nor_P2P_ms=(end-start-active-copy+both)/1e6)
        summary['GPU_activity_partitions'][str(device)]={**pieces,
            **{key.replace('_ms','_percent_of_NVTX_range'):100*value/summary['range_ms'] for key,value in pieces.items()}}
    before=__import__('json').loads((folder/'before_measured.json').read_text())
    after=__import__('json').loads((folder/'after_measured.json').read_text())
    payload=sum(v.get('transport_bytes_sent',0)-before.get(k,{}).get('transport_bytes_sent',0) for k,v in after.items())
    summary.update(measured_transport_bytes=payload,P2P_bytes_match_transport_payload=payload==summary['P2P_bytes'])
    strings=dict(db.execute('select id,value from StringIds'))
    runtime=list(db.execute('select nameId,end-start from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end)))
    api={}
    for name,duration in runtime:
        label=strings.get(name,str(name))
        if any(word in label for word in ('Event','Stream','Graph','Memcpy')):
            item=api.setdefault(label,dict(count=0,total_host_ms=0.))
            item['count']+=1;item['total_host_ms']+=duration/1e6
    summary['runtime_API_host_time_not_exclusive']=api
    write(folder/'timeline_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(12,3.2))
    labels=[]
    for index,(device,values) in enumerate(sorted(kernels.items())):
        axis.broken_barh([((a-start)/1e6,(b-a)/1e6) for a,b in union(values)],(index-.3,.6),facecolors='tab:blue')
        labels.append(f'Visible GPU{device} kernel union')
    index=len(kernels)
    axis.broken_barh([((a-start)/1e6,(b-a)/1e6) for a,b in transfers],(index-.3,.6),facecolors='tab:orange')
    labels.append('Actual peer-to-peer copy union')
    axis.set_yticks(range(len(labels)),labels);axis.set_xlabel('Milliseconds from measured NVTX region start')
    axis.set_title(args.run+' — instrumented GPU activity, not clean serving latency')
    axis.set_xlim(0,(end-start)/1e6);figure.tight_layout()
    (ROOT/'figures').mkdir(exist_ok=True)
    figure.savefig(ROOT/'figures'/f'{args.run}_timeline.svg')
    print(summary)


if __name__=='__main__':main()
