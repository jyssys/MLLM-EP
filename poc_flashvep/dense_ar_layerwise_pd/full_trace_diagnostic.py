"""Capture an actual migrated request timeline, with explicit observer limits."""
import csv,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,metrics
from .physical_analysis import activity,merge
from .resume_campaign import restore,record


def intersection(a,b):
    a=merge(a);b=merge(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        x=max(a[i][0],b[j][0]);y=min(a[i][1],b[j][1])
        if y>x:out.append((x,y))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def actual_overlap(name):
    folder=ROOT/'raw'/name;data={};summaries=[]
    for role in ('prefill','decode'):
        anchor=json.loads((folder/f'activity_{role}_begin.json').read_text())['anchor']
        offset=anchor['host_midpoint']-anchor['cupti_ns']/1e9;kernels=[]
        with (folder/f'activity_{role}.csv').open() as f:
            for r in csv.DictReader(f):
                if r['kind']=='kernel':kernels.append((int(r['start_ns'])/1e9+offset,int(r['end_ns'])/1e9+offset))
        frames=json.loads((folder/f'CUDA_event_enclosures_{role}.json').read_text())['rows']
        frames=[r for r in frames if all(x.startswith('q') and x[1:].isdigit() for x in r['requests'])]
        selected=[]
        for frame in frames:
            union=intersection(kernels,[(frame['start_monotonic'],frame['end_monotonic'])])
            selected.append(dict(frame=frame,kernel_intervals=union))
            summaries.append(dict(role=role,requests=frame['requests'],prefilling=frame['prefilling'],
                event_enclosure_ms=frame['elapsed_ms'],enclosed_kernel_union_ms=1000*sum(b-a for a,b in union)))
        data[role]=selected
    overlaps=[];pairs=[]
    for source in data['prefill']:
        for dest in data['decode']:
            if not dest['frame']['prefilling']:continue
            if set(source['frame']['requests']) & set(dest['frame']['requests']):continue
            intervals=intersection(source['kernel_intervals'],dest['kernel_intervals'])
            if intervals:
                overlaps.extend(intervals);pairs.append(dict(P_requests=source['frame']['requests'],
                    D_suffix_requests=dest['frame']['requests'],simultaneous_kernel_seconds=sum(b-a for a,b in intervals)))
    union=merge(overlaps)
    return dict(pairs=pairs,simultaneous_P_forward_D_suffix_kernel_seconds=sum(b-a for a,b in union),
        frame_summaries=summaries,evidence='DIAGNOSTIC_ACTUAL_CUDA_ACTIVITY',
        scope='Different requests overlap on the two GPUs in this instrumented execution. Event bounds select actual kernel intervals. This is not a clean performance estimate or a counterfactual speedup.')


def main():
    template='graph_qwen8_L8000_O16_N32_c28_event_diagnostic_v29'
    receipt=ROOT/'raw'/template/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
        template='eager_qwen8_L8000_O16_N32_c28_event_diagnostic_v23'
    c=json.loads((ROOT/'configs'/(template+'.json')).read_text())
    c.update(name=('graph' if template.startswith('graph') else 'eager')+'_qwen8_L8000_O16_N32_c28_full_kernel_diagnostic_v29',
        physical_activity=True,activity_roles=['prefill','decode'],activity_host_pool_bytes=1024**3,
        activity_quiet_destination=False,activity_memcpy=True,hidden_source_backpressure=True,
        phase='ACTUAL_MIGRATION_KERNEL_TIMELINE_DIAGNOSTIC',clean_timing=False,
        scope='Full source and destination CUDA activity, events, and Host observations. This diagnostic never supplies clean online speedup. The matched Event-only control and a strict interference screen remain separate.')
    name=save(c)
    try:
        if not execute(name,restore=False):return
        window=json.loads((ROOT/'raw'/name/'window.json').read_text())
        acts=[activity(name,role,window['start'],window['end']) for role in ('prefill','decode')]
        a=metrics(template);b=metrics(name)
        keys=('completed_QPS','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
        ratios={k:b[k]/a[k] for k in keys};within=all(abs(v-1)<=.05 for v in ratios.values())
        result=dict(name=name,control=template,metrics=b,output_gate=compare(template,name),
            actual_kernel_activity=acts,observer_ratios=ratios,single_control_interference_screen_pass=within,
            actual_cross_GPU_overlap=actual_overlap(name),
            clean_capacity_qualified=False,evidence='DIAGNOSTIC_ACTUAL_CUDA_ACTIVITY',
            scope='One instrumented/control comparison is not an A/A/A qualification. Full kernel intervals can document the perturbed execution only. Kernel inactivity is not a safe execution budget.')
        write(ROOT/'analysis/migrated_kernel_timeline_diagnostic.json',result)
        record('full_kernel_diagnostic',dict(name=name,trace_integrity_pass=all(r['trace_integrity_pass'] for r in acts),
            observer_ratios=ratios,clean_capacity_qualified=False))
    finally:restore()


if __name__=='__main__':main()
