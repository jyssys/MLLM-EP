"""Measure kernel inactivity without promoting it to a safe offload window."""
import csv
import json
import numpy as np
from .common import ROOT,write
from .metrics import summarize,delivery_intervals


def merge(intervals):
    out=[]
    for left,right in sorted(intervals):
        if right<=left:continue
        if out and left<=out[-1][1]:out[-1][1]=max(out[-1][1],right)
        else:out.append([left,right])
    return out


def overlap(a,b):
    a=merge(a);b=merge(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        left=max(a[i][0],b[j][0]);right=min(a[i][1],b[j][1])
        if right>left:out.append((left,right))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return merge(out)


def duration(values):return sum(b-a for a,b in merge(values))


def distribution(values):
    return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),
        p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),
        max=float(np.max(values))) if values else None


def queues(rows,left,right,fields):
    selected=sorted(rows,key=lambda r:r['time']);out={};covered=0
    for i,row in enumerate(selected):
        a=max(left,row['time']);b=min(right,selected[i+1]['time'] if i+1<len(selected) else right)
        if b<=a:continue
        covered+=b-a
        for field in fields:
            value=row.get(field)
            if isinstance(value,(float,int)):out[field]=out.get(field,0)+(b-a)*value
    return dict(time_weighted_mean={k:v/covered for k,v in out.items()} if covered else {},
        covered_seconds=covered,requested_seconds=right-left,
        scope='Piecewise-constant observed scheduler states. State-change/50ms snapshots are host observations,not GPU occupancy.')


def role_analysis(name,role):
    folder=ROOT/'raw'/name;activity=folder/f'activity_{role}'
    roi=json.loads((activity/'actual_ROI.json').read_text());left,right=roi['left']['cupti_ns'],roi['right']['cupti_ns']
    assert right>left
    counter=json.loads((activity/'activity_counters.json').read_text());assert counter['passed']
    anchors=[(r['host_lower_ns']+r['host_upper_ns'])/2 for r in (roi['left'],roi['right'])]
    offset=left-anchors[0];host_left,host_right=anchors[0]/1e9,anchors[1]/1e9
    kernels=[];copies=[]
    for row in csv.DictReader((activity/'kernels.csv').open()):
        span=max(left,int(row['start_ns'])),min(right,int(row['end_ns']))
        if span[1]<=span[0]:continue
        if row['kind']=='kernel':kernels.append(span)
        elif row['kind'] in ('memcpy','memcpy2'):copies.append(span)
    busy=merge(kernels);gaps=[];cursor=left
    for a,b in busy:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<right:gaps.append([cursor,right])
    uncensored=[(a,b) for a,b in gaps if a>left and b<right]
    setup=json.loads((folder/f'NIXL_host_setup_{role}.json').read_text())['rows']
    host_api=[(max(left,r['start']*1e9+offset),min(right,r['end']*1e9+offset)) for r in setup
        if r.get('kind') in ('read_blocks_host','make_transfer_host','submit_transfer_host')]
    host_api=merge(host_api)
    source=json.loads((folder/f'scheduler_queues_{role}.json').read_text())['rows']
    witness=folder/'D_quiescent_witness.json';q=[]
    if role=='decode' and witness.exists():
        q=[(max(left,r['start']*1e9+offset),min(right,r['end']*1e9+offset)) for r in json.loads(witness.read_text())['rows']]
    cfg=json.loads((folder/'configuration.json').read_text())
    result=dict(role=role,ROI_seconds=(right-left)/1e9,kernel_busy_fraction=duration(busy)/(right-left),
        kernel_inactivity_fraction=duration(gaps)/(right-left),
        interior_kernel_gap_ms=distribution([(b-a)/1e6 for a,b in uncensored]),
        boundary_censored_gap_count=len(gaps)-len(uncensored),
        gaps_at_least_ms={str(ms):sum((b-a)>=ms*1e6 for a,b in uncensored) for ms in (1,5,10,20,34,50,100)},
        NIXL_host_enclosure_union_ms=duration(host_api)/1e6,
        kernel_gap_overlap_NIXL_host_ms=duration(overlap(gaps,host_api))/1e6,
        host_quiescence_after_stream_event_ms=duration(q)/1e6 if role=='decode' else None,
        host_quiescence_overlap_kernel_inactivity_ms=duration(overlap(q,gaps))/1e6 if role=='decode' else None,
        observed_copy_union_ms=duration(copies)/1e6 if cfg.get('activity_copies') else None,
        all_GPU_engine_idle='NOT_IDENTIFIED' if not cfg.get('activity_copies') else 'NOT_ESTABLISHED_BY_KERNEL_AND_COPY_CAPTURE_ALONE',
        decode_safe_executable_window='NOT_IDENTIFIED',
        queue=queues(source,host_left,host_right,('waiting_total','running','active_decode','decode_ready_waiting',
            'decode_ready_including_first_query','remote_KV_waiting','free_KV_blocks')),
        anchor_uncertainty_ns=max(r['host_upper_ns']-r['host_lower_ns'] for r in (roi['left'],roi['right'])),
        anchor_offset_drift_ns=(right-anchors[1])-offset,
        kernel_intervals_ns=busy,kernel_inactivity_intervals_ns=gaps,
        evidence='DIAGNOSTIC_PENDING_OBSERVER_GATE',
        scope='Intervals are clipped to the actual collector ROI. ROI_seconds records its measured length. Inactivity is not physical GPU idle. '
            'Copies are NOT_MEASURED when disabled. Nested host API intervals are unioned,not summed. '
            'API overlap is not proof of causality. Quiescence ends at an observed receipt,not a known future arrival. '
            'No partition state was ready in this Native trace. No safe-fit or dynamic speedup follows from a gap length.')
    return result


def request_metrics(rows):
    itls=[v for r in rows for v in delivery_intervals(r['token_events'],r['output_tokens'])]
    tpot=[r['TPOT_ms'] for r in rows if r['TPOT_ms'] is not None]
    return dict(TTFT_p95_ms=float(np.percentile([r['TTFT_ms'] for r in rows],95)),
        TTFT_mean_ms=float(np.mean([r['TTFT_ms'] for r in rows])),
        ITL_p99_ms=float(np.percentile(itls,99)),TPOT_p95_ms=float(np.percentile(tpot,95)),mean_TBT_ms=float(np.mean(itls)))


def observer(qps,roles,names=None):
    if names is None:names=[f'Native400_q{qps}_{label}_ROI_v1' for label in ('Clean1','BothGPU_ROI','Clean2')]
    if not all((ROOT/'raw'/n/'dispatcher_result.json').exists() for n in names):return dict(passed=False,status='PENDING_CLEAN_PROFILE_CLEAN')
    assert all(json.loads((ROOT/'raw'/n/'dispatcher_result.json').read_text())['complete'] for n in names)
    metrics=[summarize(n) for n in names];selected=set()
    for role in ('prefill','decode'):
        folder=ROOT/'raw'/names[1];roi=json.loads((folder/f'activity_{role}/actual_ROI.json').read_text())
        lo=roi['left']['host_lower_ns']/1e9;hi=roi['right']['host_upper_ns']/1e9
        selected.update(r['request_id'] for r in json.loads((folder/f'actual_queries_{role}.json').read_text())['rows'] if lo<=r['time']<=hi)
    local=[]
    for name in names:
        rows=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
        rows=[r for r in rows if 'q'+str(r['sample_id']) in selected];assert rows
        local.append(request_metrics(rows))
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')
    global_ratios={f:metrics[1][f]/((metrics[0][f]+metrics[2][f])/2) for f in fields}
    local_ratios={f:local[1][f]/((local[0][f]+local[2][f])/2) for f in fields}
    passed=all(abs(v-1)<=.05 for v in list(global_ratios.values())+list(local_ratios.values()))
    return dict(passed=passed,status='PASS' if passed else 'DIAGNOSTIC',independent_triplets=1,
        full_trace_profile_over_mean_clean=global_ratios,matched_exposed_requests_profile_over_mean_clean=local_ratios,
        exposed_request_ids=sorted(selected,key=lambda s:int(s[1:])),
        scope='One triplet only. The same request IDs are compared in the two cleans. '
            'Requests are exposed when their observed P/D query starts inside the actual profile ROI. '
            'This local check prevents a whole-trace average from hiding exposure effects. It is not a causal confidence interval.')


def main():
    results=[]
    for qps in (1.8,2.8):
        name=f'Native400_q{qps}_BothGPU_ROI_ROI_v1'
        if not all((ROOT/'raw'/name/f'activity_{role}/actual_ROI.json').exists() for role in ('prefill','decode')):continue
        roles=[role_analysis(name,role) for role in ('prefill','decode')];gate=observer(qps,roles)
        for role in roles:role['evidence']='MEASURED_KERNEL_ACTIVITY' if gate['passed'] else 'DIAGNOSTIC'
        value=dict(name=name,QPS=qps,roles=roles,observer_gate=gate,decode_safe_window='NOT_IDENTIFIED')
        write(ROOT/'analysis'/(name+'_windows.json'),value);results.append(value)
    print([(r['QPS'],r['observer_gate']['status'],[(v['role'],round(v['kernel_busy_fraction'],4)) for v in r['roles']]) for r in results],flush=True)
    return results


if __name__=='__main__':main()
