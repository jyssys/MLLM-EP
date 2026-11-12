"""Clip concurrent kernels to one Serving window and merge all streams."""
import csv,json,statistics
from .common import *
from .analyze import records,metrics,percentile


def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if not result or a>result[-1][1]:result.append([a,b])
        else:result[-1][1]=max(result[-1][1],b)
    return result


def gaps(intervals,a,b):
    result=[];cursor=a
    for x,y in merge([(max(a,x),min(b,y)) for x,y in intervals if y>a and x<b]):
        if x>cursor:result.append([cursor,x])
        cursor=max(cursor,y)
    if cursor<b:result.append([cursor,b])
    return result


def activity(name,role,left,right):
    folder=ROOT/'raw'/name;begin=json.loads((folder/f'activity_{role}_begin.json').read_text())
    end=json.loads((folder/f'activity_{role}_end.json').read_text());a=begin['anchor'];b=end['anchor']
    offset=a['host_midpoint']-a['cupti_ns']/1e9
    drift_ms=1000*((b['host_midpoint']-b['cupti_ns']/1e9)-offset)
    intervals=[];kinds={};stream_counts={}
    with (folder/f'activity_{role}.csv').open() as f:
        for r in csv.DictReader(f):
            x=int(r['start_ns'])/1e9+offset;y=int(r['end_ns'])/1e9+offset
            if y<=left or x>=right:continue
            kinds[r['kind']]=kinds.get(r['kind'],0)+1
            if r['kind']!='kernel':continue
            intervals.append((max(left,x),min(right,y)))
            key=r['stream'];stream_counts[key]=stream_counts.get(key,0)+1
    busy=merge(intervals);idle=gaps(busy,left,right);duration=right-left
    lengths=[1000*(y-x) for x,y in idle];union=sum(y-x for x,y in busy)
    result=dict(role=role,window_start=left,window_end=right,window_seconds=duration,
        kernel_records=len(intervals),kernel_union_seconds=union,kernel_busy_percent=100*union/duration,
        physical_kernel_inactive_seconds=duration-union,physical_kernel_inactive_percent=100*(duration-union)/duration,
        idle_window_count=len(idle),idle_mean_ms=statistics.mean(lengths) if lengths else 0,
        idle_max_ms=max(lengths,default=0),stream_kernel_counts=stream_counts,activity_kind_counts=kinds,
        clock_offset_drift_ms=drift_ms,counters=end['counters'],
        trace_integrity_pass=end['result']==0 and end['counters']['dropped']==0 and end['counters']['invalid']==0 and abs(drift_ms)<=.1,
        evidence='TRACE_ANALYSIS',
        scope='Measured concurrent compute-kernel interval union. It is not SM occupancy, Decode-safe slack, or an offload capacity. Native NIXL GPU kernels, if present, count as compute activity.')
    for q in (.5,.9,.95,.99):result[f'idle_p{int(q*100)}_ms']=percentile(lengths,q)
    result['thresholds']={str(ms):dict(count=sum(t>=ms for t in lengths),total_seconds=sum(t/1000 for t in lengths if t>=ms)) for ms in (1,2,5,10,20,30,50,100)}
    write(folder/f'kernel_union_{role}.json',result)
    with (folder/f'kernel_union_{role}.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['start_monotonic','end_monotonic','kind'])
        writer.writerows((x,y,'kernel_union') for x,y in busy);writer.writerows((x,y,'kernel_inactive') for x,y in idle)
    return result


def queue_screen(name,left,right):
    folder=ROOT/'raw'/name
    native=[json.loads(p.read_text()) for p in folder.glob('native_prefill_*.json')]
    assert len(native)==1;d=native[0]
    samples=[x for x in d['queue_samples'] if left<=x['time']<=right]
    middle=(left+right)/2
    halves=[[x['waiting'] for x in samples if (x['time']<middle)==first] for first in (True,False)]
    completed=[r for r in records(folder/'native_handoffs.jsonl') if left<=r['time']<=right]
    reference=json.loads((ROOT/'state/discovery_plan.json').read_text())['P_capacity_requests_s']
    service=len(completed)/(right-left)
    return dict(queue_samples=len(samples),P_waiting_mean_front=statistics.mean(halves[0]) if halves[0] else None,
        P_waiting_mean_back=statistics.mean(halves[1]) if halves[1] else None,
        P_waiting_max=max((x['waiting'] for x in samples),default=0),
        queue_growth=bool(all(halves)) and statistics.mean(halves[1])>statistics.mean(halves[0]),
        P_KV_allocation_refusals=sum(left<=r['time']<=right for r in d['KV_allocation_refusals']),
        P_completions=len(completed),P_service_requests_s=service,P_capacity_reference_requests_s=reference,
        P_service_proximity=service/reference,
        scope='Native P scheduler waiting counts need the detailed status/credit audit. Coordinator outstanding requests are not used as compute-ready counts.')


def main():
    plan_path=ROOT/'state/native_confirmation_lease_v2_plan.json'
    if not plan_path.exists():plan_path=ROOT/'state/native_confirmation_plan.json'
    plan=json.loads(plan_path.read_text());names=plan['jobs']
    ms=[metrics(n) for n in names]
    if not all(m['complete'] for m in ms):print('INCOMPLETE');return
    cleanA,host,kernel,cleanB=ms
    keys=('completed_QPS','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    comparisons={label:{key:value[key]/statistics.mean([cleanA[key],cleanB[key]]) for key in keys}
        for label,value in (('host',host),('kernel',kernel))}
    aa={key:cleanB[key]/cleanA[key] for key in keys}
    gate=all(abs(comparisons['kernel'][key]-1)<=.05 for key in keys)
    window=json.loads((ROOT/'raw'/names[2]/'window.json').read_text())
    length=window['last_arrival']-window['start'];left=window['start']+.2*length;right=window['start']+.8*length
    acts=[activity(names[2],role,left,right) for role in ('prefill','decode')]
    queue=queue_screen(names[2],left,right)
    qualified=gate and all(a['trace_integrity_pass'] for a in acts)
    result=dict(names=names,metrics=ms,A_A_ratios=aa,observation_ratios=comparisons,
        interference_pass=gate,trace_qualified=qualified,kernel_activity=acts,queue_screen=queue,
        spatial_compute_slack='NOT_MEASURED',request_feasible_decode_safe_capacity='NOT_MEASURED',
        phase_B_pass=False,
        scope='Physical timeline qualification is not extra-work admission qualification. Phase B also requires an actual Decode-safe intervention and an audited compute-ready queue.')
    write(ROOT/'analysis/native_same_window_physical_screen.json',result)
    print(json.dumps(dict(interference_pass=gate,trace_qualified=qualified,ratios=comparisons,
        activities=[{k:v for k,v in x.items() if k not in ('thresholds','stream_kernel_counts')} for x in acts],queue=queue),indent=2))


if __name__=='__main__':main()
