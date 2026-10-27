"""Derived CUPTI activity unions and same-host collective-arrival dispersion."""
from collections import Counter,defaultdict
import gzip
import json
import numpy as np
from .common import ROOT
from .aggregate_quality import csv_write
from .kernel_profile import union

def main():
    activity=[];arrival=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            directory=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
            calls=[]
            for rank in range(ep):
                path=directory/f'kineto_b1_p128.rank{rank}.json.gz'
                if not path.exists():continue
                with gzip.open(path,'rt') as source:data=json.load(source)
                events=data['traceEvents']
                kernels=[v for v in events if v.get('cat')=='kernel' and v.get('ph')=='X']
                copies=[v for v in events if v.get('cat') in ['gpu_memcpy','gpu_memset'] and v.get('ph')=='X']
                compute=[v for v in kernels if 'nccl' not in v['name'].lower()]
                nccl=[v for v in kernels if 'nccl' in v['name'].lower()]
                left=min(v['ts'] for v in kernels);right=max(v['ts']+v['dur'] for v in kernels)
                def covered(items):
                    return union([(max(left,v['ts']),min(right,v['ts']+v['dur'])) for v in items
                        if v['ts']<right and v['ts']+v['dur']>left])
                row=dict(model=model,ep=ep,batch=1,prefix_tokens=128,rank=rank,observer_window_ms=(right-left)/1000,
                    kernel_union_ms=covered(kernels)/1000,compute_kernel_union_ms=covered(compute)/1000,
                    nccl_kernel_union_ms=covered(nccl)/1000,copy_or_memset_union_ms=covered(copies)/1000,
                    no_recorded_gpu_activity_ms=((right-left)-covered(kernels+copies))/1000,
                    gpu_copy_events=len(copies),gpu_kernel_events=len(kernels),
                    evidence='LIVE_CUPTI_HEAVY_OBSERVER_NOT_CLEAN_E2E',
                    qualification='No recorded GPU activity excludes visible kernels, copies and memset. It is not an SM-occupancy counter.')
                for kind in ['DtoH','HtoD','DtoD']:
                    selected=[v for v in copies if kind in v['name']]
                    row[f'{kind}_explicit_bytes']=sum(v.get('args',{}).get('bytes',0) for v in selected)
                    row[f'{kind}_events']=len(selected)
                activity.append(row)
                c=sorted([v for v in events if v.get('cat')=='cpu_op' and v.get('name') in
                    ['c10d::_allgather_base_','c10d::alltoall_base_','c10d::allreduce_']],key=lambda v:v['ts'])
                calls.append((data['baseTimeNanoseconds'],c))
            if len(calls)!=ep:continue
            assert len(set(base for base,_ in calls))==1,'Do not align unrelated profiler clock origins'
            assert len(set(len(values) for _,values in calls))==1
            for index in range(len(calls[0][1])):
                values=[v[index] for _,v in calls]
                assert len(set(v['name'] for v in values))==1
                timestamps=[v['ts'] for v in values]
                arrival.append(dict(model=model,ep=ep,batch=1,prefix_tokens=128,collective_index=index,
                    operation=values[0]['name'],host_submit_skew_us=max(timestamps)-min(timestamps),
                    late_rank=int(np.argmax(timestamps)),early_rank=int(np.argmin(timestamps)),
                    evidence='LIVE_SAME_HOST_PROFILER_CPU_TIMESTAMPS',
                    qualification='Arrival dispersion, not isolated network latency or exact GPU waiting. Profiler perturbation remains.'))
    csv_write(ROOT/'kernel_activity.csv',activity)
    csv_write(ROOT/'collective_arrival_skew.csv',arrival)

if __name__=='__main__':main()
