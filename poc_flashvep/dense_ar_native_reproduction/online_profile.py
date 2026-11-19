"""Measure Native kernel activity after the frozen Main sweep finishes."""
import csv
import json
import time
import numpy as np
from .common import ROOT,write
from .campaign import serving
from .qualification import records
from .serving_analysis import analyze,stats,time_weighted
from .profile_results import union


def window_metrics(folder,left,right):
    """Compare arrival-selected TTFT and interval-selected TBT across starts."""
    origin=json.loads((folder/'window.json').read_text())['start']
    ttft=[];intervals=[]
    for row in records(folder/'requests.jsonl'):
        if left<=row['arrival']-origin<right:ttft.append(row['TTFT_ms'])
        for previous,current in zip(row['token_events'],row['token_events'][1:]):
            if previous['time']-origin>=left and current['time']-origin<=right:
                intervals.extend([1000*(current['time']-previous['time'])]+
                    [0.]*(current['count']-previous['count']-1))
    return dict(TTFT_mean_ms=float(np.mean(ttft)) if ttft else None,
        mean_TBT_ms=float(np.mean(intervals)) if intervals else None,
        arrivals=len(ttft),intervals=len(intervals),relative_window_seconds=[left,right])


def merge(intervals):
    result=[]
    for left,right in sorted(intervals):
        if not result or left>result[-1][1]:result.append([left,right])
        else:result[-1][1]=max(right,result[-1][1])
    return result


def kernel_window(name):
    folder=ROOT/'raw'/name;bounds={};counter_gates={}
    for role in ('prefill','decode'):
        stamps=json.loads((folder/f'activity_window_{role}.json').read_text())
        points=[]
        for side in ('left','right'):
            s=stamps[side]
            points.append(((s['host_lower_ns']+s['host_upper_ns'])/2e9,s['cupti_ns']))
        bounds[role]=dict(points=points,stamps=stamps)
        counter_gates[role]=json.loads((folder/f'activity_{role}/activity_counters.json').read_text())
    left=max(v['points'][0][0] for v in bounds.values())+2.
    right=min(v['points'][1][0] for v in bounds.values())-2.
    assert right>left
    result={}
    for role,value in bounds.items():
        (h0,c0),(h1,c1)=value['points'];scale=(h1-h0)/(c1-c0)
        intervals=[];kernel_count=0
        with (folder/f'activity_{role}/kernels.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['kind']!='kernel':continue
                begin=h0+(int(row['start_ns'])-c0)*scale
                end=h0+(int(row['end_ns'])-c0)*scale
                if begin<right and end>left:
                    intervals.append((max(begin,left),min(end,right)));kernel_count+=1
        merged=merge(intervals);gaps=[];cursor=left
        for begin,end in merged:
            if begin>cursor:gaps.append([cursor,begin])
            cursor=end
        if cursor<right:gaps.append([cursor,right])
        active=union(intervals);duration=right-left
        uncertainty=max((value['stamps'][side]['host_upper_ns']-value['stamps'][side]['host_lower_ns'])/1e6 for side in ('left','right'))
        starts=np.arange(0,duration,1.);widths=np.minimum(1.,duration-starts);binned=np.zeros(len(starts))
        for begin,end in merged:
            a,b=begin-left,end-left
            for i in range(max(0,int(a)),min(len(starts),int(b)+1)):
                binned[i]+=max(0,min(b,starts[i]+widths[i])-max(a,starts[i]))
        result[role]=dict(kernel_union_seconds=active,window_seconds=duration,
            kernel_busy_fraction=active/duration,kernel_inactivity_fraction=1-active/duration,
            kernel_count=kernel_count,kernel_inactivity_gaps_ms=stats(1000*(b-a) for a,b in gaps),
            kernel_inactivity_gaps_at_least_ms={str(limit):dict(count=sum(1000*(b-a)>=limit for a,b in gaps),
                total_seconds=sum(b-a for a,b in gaps if 1000*(b-a)>=limit)) for limit in (1,5,10,20,50,100)},
            timestamp_bracket_max_ms=uncertainty,counters=counter_gates[role],
            kernel_union_bins_1s=[dict(start_seconds=float(a),end_seconds=float(a+w),kernel_union_seconds=float(t)) for a,w,t in zip(starts,widths,binned)],
            exact_interval_source=str((folder/f'activity_{role}/kernels.csv').relative_to(ROOT)),
            scope='Kernel activity across observed streams. Kernel inactivity can include DMA or control waits. It is not a decode-safe executable window.')
        setup=folder/f'NIXL_host_setup_{role}.json'
        calls=json.loads(setup.read_text())['rows'] if setup.exists() else []
        calls=[c for c in calls if c['kind']=='submit_transfer_host']
        large=[]
        for begin,end in gaps:
            if end-begin<.020:continue
            overlap=union([(max(begin,c['start']),min(end,c['end'])) for c in calls if c['start']<end and c['end']>begin])
            large.append(dict(start_relative_seconds=begin-left,end_relative_seconds=end-left,
                duration_ms=1000*(end-begin),overlapping_transfer_host_ms=1000*overlap))
        total_gap=sum(r['duration_ms'] for r in large)
        total_overlap=sum(r['overlapping_transfer_host_ms'] for r in large)
        result[role]['large_kernel_inactivity_intervals']=large
        result[role]['large_gap_transfer_overlap']=dict(threshold_ms=20,count=len(large),
            gaps_with_overlap=sum(r['overlapping_transfer_host_ms']>0 for r in large),
            gap_seconds=total_gap/1000,overlapping_host_seconds=total_overlap/1000,
            fraction=total_overlap/total_gap if total_gap else None,
            scope='Same-run temporal overlap with the host transfer-call enclosure. Not pure DMA time or a causal intervention.')
        queues=json.loads((folder/f'scheduler_queues_{role}.json').read_text())['rows']
        result[role]['sampled_queue_state_same_window']={key:time_weighted(queues,key,dict(start=left,end=right)) for key in
            ('waiting_total','remote_KV_waiting','active_decode','decode_ready_waiting','decode_ready_including_first_query')}
    origin=json.loads((folder/'window.json').read_text())['start']
    return dict(roles=result,relative_window_seconds=[left-origin,right-origin],
        common_host_window=[left,right],edge_trim_seconds=2.,
        clock_mapping='Affine mapping from two CUPTI timestamps and their host-call bracket midpoints. Different GPU streams are not combined across devices.')


def summarize_triplet(rate,names):
    profile=kernel_window(names['profile']);left,right=profile['relative_window_seconds']
    measured={label:json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()) for label,name in names.items()}
    subsets={label:window_metrics(ROOT/'raw'/name,left,right) for label,name in names.items()}
    changes={}
    for field in ('TTFT_mean_ms','mean_TBT_ms','TPOT_p95_ms'):
        clean=np.mean([measured[label][field] for label in ('clean_before','clean_after')])
        changes['whole_'+field]=measured['profile'][field]/clean-1
    for field in ('TTFT_mean_ms','mean_TBT_ms'):
        values=[subsets[label][field] for label in ('clean_before','clean_after')]
        changes['ROI_'+field]=subsets['profile'][field]/np.mean(values)-1 if subsets['profile'][field] is not None and all(v is not None and v>0 for v in values) else None
    counters_pass=all(v['counters']['passed'] and v['kernel_count']>0 for v in profile['roles'].values())
    clock_pass=all(v['timestamp_bracket_max_ms']<=.5 for v in profile['roles'].values())
    observer=all(value is not None and abs(value)<=.05 for value in changes.values())
    output=dict(offered_QPS=rate,names=names,kernel_activity=profile,relative_change=changes,
        ROI_client_metrics=subsets,observer_gate_passed=observer and counters_pass and clock_pass,
        evidence='MEASURED_KERNEL_ACTIVITY' if observer and counters_pass and clock_pass else 'DIAGNOSTIC_OBSERVER_GATE_FAILED',
        independent_starts=dict(clean=2,profiled=1),
        scope='One profiled start with two clean controls. The 5% observer check uses point estimates, not a confidence bound. No physical-idle, safe-window or sustainable-capacity claim.')
    write(ROOT/'analysis'/f'Native_online_kernel_activity_q{rate:g}_v1.json',output)
    return output


def main():
    declaration=ROOT/'configs/frozen_Native_online_kernel_activity_design_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),offered_QPS=[1.,2.35],count=200,
            dataset='longbench_v1_mixed200.json',arrival_seed=2026112041,
            order=['clean_before','profile','clean_after'],kernel_activity_seconds=90,
            observer_limit=.05,clock_bracket_limit_ms=.5,edge_trim_seconds=2,
            scope='Same standard Native engine, Natural EOS and primary B8192/S1024. Profile a prespecified initial interval. Do not replace Main latency points with profiled results.'))
    design=json.loads(declaration.read_text())
    while not (ROOT/'state/main_common_sweep_complete.json').exists():
        for name in ('STOP_REQUESTED.json','Main_driver_hold.json','campaign_hold.json'):
            assert not (ROOT/'state'/name).exists(),name
        time.sleep(5)
    result=[]
    for rate in design['offered_QPS']:
        names={}
        for label in design['order']:
            name=f'profile_Native_LongBench_N200_q{rate:g}_{label}_BI1_v1'
            names[label]=name
            serving(name,'Native',design['dataset'],design['count'],rate,design['arrival_seed'],
                phase='NATIVE_KERNEL_ACTIVITY_CONTROL',kernel_activity=label=='profile',
                kernel_activity_seconds=design['kernel_activity_seconds'])
            analyze(name)
        result.append(summarize_triplet(rate,names))
    write(ROOT/'state/online_kernel_activity_complete.json',dict(time=time.time(),complete=True,
        observer_passed=[r['offered_QPS'] for r in result if r['observer_gate_passed']]))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/online_kernel_activity_hold.json',dict(time=time.time(),error=repr(error)))
        raise
