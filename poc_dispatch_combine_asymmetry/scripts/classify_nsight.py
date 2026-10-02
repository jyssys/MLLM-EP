"""CUDA kernel classification by launch correlation inside native NVTX ranges.

Profiler durations are diagnostic, not calibrated clean-stage components.
The fused copy/poll kernels cannot be turned into pure wire-bandwidth timers.
"""
import argparse
import collections
import json
import sqlite3
from pathlib import Path

import numpy as np

from poc_dispatch_combine_asymmetry.contract import write


def classify(name):
    if 'get_dispatch_layout' in name:return 'dispatch_layout'
    if 'notify_dispatch' in name:return 'dispatch_notify_counts_barriers'
    if 'intranode::dispatch<' in name:return 'dispatch_pack_move_receive_poll'
    if 'cached_notify_combine' in name:return 'combine_cached_notify_barriers'
    if 'intranode::combine<' in name:return 'combine_reverse_move_reduce_poll'
    return None


def extract(file,model):
    conn=sqlite3.connect(file)
    ranges=conn.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'ASYMMETRY_NATIVE_%'").fetchall()
    records=[];summaries=[]
    for begin,end,tid,text in ranges:
        # Nsight globalPid is globalTid with its low 24 thread-ID bits cleared.
        # Validate this against actual correlated launch records, not CUDA ranks.
        pid=tid & ~((1<<24)-1)
        rows=conn.execute('''SELECT DISTINCT k.start,k.end,k.deviceId,k.streamId,k.globalPid,s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME a ON a.correlationId=k.correlationId
              AND (a.globalTid & -16777216)=k.globalPid
            WHERE a.globalTid=? AND a.start>=? AND a.end<=? AND k.globalPid=?
            ORDER BY k.start''',(tid,begin,end,pid)).fetchall()
        classes=collections.defaultdict(list)
        for start,stop,device,stream,gpid,name in rows:
            cl=classify(name)
            if cl:
                r={'model':model,'arm':text.removeprefix('ASYMMETRY_NATIVE_'),'globalPid':gpid,
                   'os_pid':(gpid>>24)&0xffffff,
                   'nsight_device_id':device,'stream':stream,'class':cl,'duration_ms':(stop-start)/1e6,
                   'start_ns':start,'end_ns':stop,'label':'KERNEL_CLASSIFIED','name':name}
                records.append(r);classes[cl].append(r['duration_ms'])
        if not classes:raise ValueError('no correlated DeepEP launches within native range')
        summaries.append({'model':model,'arm':text.removeprefix('ASYMMETRY_NATIVE_'),'globalPid':pid,'os_pid':(pid>>24)&0xffffff,
            'native_range_host_ms':(end-begin)/1e6,
            'classes':{cl:{'n':len(v),'total_ms':sum(v),'mean_ms':float(np.mean(v)),
                           'first_layer_ms':v[0],'median_ms':float(np.median(v)),'max_ms':max(v)} for cl,v in classes.items()},
            'complete_48_layers':all(len(v)==48 for v in classes.values()) and len(classes)==5})
    return records,summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--mllm',type=Path,required=True);p.add_argument('--llm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    records=[];summaries=[]
    for model,file in [('mllm',args.mllm),('llm',args.llm)]:
        r,s=extract(file,model);records.extend(r);summaries.extend(s)
    write(args.output,{'records':records,'summaries':summaries,
        'scope':'single repeat, profiler diagnostic, own-rank kernel durations; no cross-device timestamp subtraction',
        'data_movement_only_interval_available':False,
        'unresolved':['packing versus movement versus polling inside dispatch kernel','movement versus reduction inside combine kernel',
                      'notify barrier network wait versus prior producer delay without additional intervention'],
        'kernels_from_installed_DeepEP_commit':'73b6ea4a439ba03a695563f9fd242c8e4b02b37c'})
    print('CLASSIFIED_NATIVE_RANGES',len(summaries),'COMPLETE',sum(s['complete_48_layers'] for s in summaries))


if __name__=='__main__':main()
