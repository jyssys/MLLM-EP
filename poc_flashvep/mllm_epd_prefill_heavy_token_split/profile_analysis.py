"""Align actual GPU activity and report physical idle, not safe headroom."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT, digest, preserve, write


def union(intervals, left, right):
    intervals=np.asarray(intervals,dtype=float).reshape(-1,2)
    intervals=np.column_stack((np.maximum(intervals[:,0],left),np.minimum(intervals[:,1],right)))
    intervals=intervals[intervals[:,1]>intervals[:,0]]
    if not len(intervals):return np.empty((0,2))
    intervals=intervals[np.argsort(intervals[:,0],kind='stable')]
    ends=np.maximum.accumulate(intervals[:,1])
    starts=np.r_[True,intervals[1:,0]>ends[:-1]]
    begin=np.flatnonzero(starts)
    return np.column_stack((intervals[begin,0],ends[np.r_[begin[1:]-1,len(ends)-1]]))


def complement(intervals,left,right):
    intervals=union(intervals,left,right)
    bounds=np.r_[left,intervals.ravel(),right]
    pairs=bounds.reshape(-1,2)
    return pairs[pairs[:,1]>pairs[:,0]]


def intersection(a,b):
    result=[];i=j=0
    while i<len(a) and j<len(b):
        left=max(a[i,0],b[j,0]);right=min(a[i,1],b[j,1])
        if right>left:result.append((left,right))
        if a[i,1]<=b[j,1]:i+=1
        else:j+=1
    return np.asarray(result,dtype=float).reshape(-1,2)


def duration(intervals):return float(np.sum(intervals[:,1]-intervals[:,0]))


def windows(intervals):
    x=(intervals[:,1]-intervals[:,0])*1000
    if not len(x):return dict(count=0,total_seconds=0)
    return dict(count=len(x),total_seconds=float(x.sum()/1000),mean_ms=float(x.mean()),
                p50_ms=float(np.percentile(x,50)),p90_ms=float(np.percentile(x,90)),
                p95_ms=float(np.percentile(x,95)),p99_ms=float(np.percentile(x,99)),max_ms=float(x.max()),
                threshold_coverage=[dict(threshold_ms=t,count_fraction=float(np.mean(x>=t)),
                    duration_fraction=float(x[x>=t].sum()/x.sum())) for t in [.1,1,5,10,20,50,100]])


def load_activity(folder):
    record=json.loads((folder/'actual_ROI.json').read_text())
    a,b=record['left'],record['right']
    host_a=(a['host_lower_ns']+a['host_upper_ns'])/2
    host_b=(b['host_lower_ns']+b['host_upper_ns'])/2
    slope=(host_b-host_a)/(b['cupti_ns']-a['cupti_ns'])
    data=np.loadtxt(folder/'kernels.csv',delimiter=',',skiprows=1,usecols=(0,1,2),
                    dtype=[('kind','U7'),('start','i8'),('end','i8')],ndmin=1)
    convert=lambda x:((x-a['cupti_ns']).astype(float)*slope+host_a)/1e9
    intervals=np.column_stack((convert(data['start']),convert(data['end'])))
    anchor=dict(slope=slope,drift_ppm=(slope-1)*1e6,
                endpoint_host_bracket_max_us=max(a['host_upper_ns']-a['host_lower_ns'],b['host_upper_ns']-b['host_lower_ns'])/1000)
    return intervals[data['kind']=='kernel'],intervals[data['kind']!='kernel'],anchor,record


def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--proxy-name',required=True);args=p.parse_args()
    preserve();name=args.name;run=name+'_profile'
    done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
    request_rows=[json.loads(s) for s in (ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()]
    left=done['monotonic_start'];right=left+done['duration_seconds']
    arrivals=sorted(r['scheduled_arrival'] for r in request_rows)
    rois=dict(full_cohort=(left,right),central_arrival_window=tuple(np.percentile(arrivals,[10,90])))
    profile=ROOT/'raw/profiles'/name
    activity={role:load_activity(profile/role) for role in ['encoder','prefill','decode']}
    pending={role:[] for role in activity}
    proxy_rows=[json.loads(s) for s in (ROOT/'raw'/f'proxy_{args.proxy_name}.jsonl').read_text().splitlines()]
    for row in proxy_rows:
        if not row['runtime_request_id'].startswith(run+'_'):continue
        for role,a,b in [('encoder','E_submit','E_complete'),('prefill','P_submit','P_ready'),('decode','D_submit','finished')]:
            if a in row and b in row:pending[role].append((row[a],row[b]))
    qualification=ROOT/'analysis'/f'{name}_qualification.json'
    observer=json.loads(qualification.read_text()) if qualification.exists() else None
    result=dict(time=time.time(),name=name,QPS=json.loads((ROOT/'configs'/f'{name}.json').read_text())['QPS'],
        observer_gate_passed=observer['observer_gate_passed'] if observer else None,
        model_quality_gate='PENDING',evidence='TRACE_ANALYSIS of DIAGNOSTIC online activity',ROIs={},
        limitations=['Physical kernel inactivity is not an admission-safe window.',
            'Central arrival window is not proof of sustainable steady state.',
            'HTTP pending intervals include preprocessing, transfer, engine queue and service.',
            'Kernel/copy observation does not measure SM active, Tensor Core active or HBM bandwidth.',
            'Idle without pending HTTP encode/decode is distinguished from pending host/transfer waits.',
            'No additional prefill was inserted. Encode/decode-safe headroom is NOT_MEASURED.'])
    window_rows=[]
    for roi,(lo,hi) in rois.items():
        span=hi-lo;roles={};busy={};idle={}
        for role,(kernel,copy,anchor,counters) in activity.items():
            busy[role]=union(kernel,lo,hi);idle[role]=complement(busy[role],lo,hi)
            communications=union(copy,lo,hi)
            fully_quiet=complement(np.concatenate((busy[role],communications)),lo,hi)
            outstanding=union(pending[role],lo,hi)
            no_http_pending=intersection(idle[role],complement(outstanding,lo,hi))
            waiting=intersection(idle[role],outstanding)
            sched=json.loads((profile/role/'scheduler.json').read_text())['rows']
            selected=[r for r in sched if lo<=r['begin_ns']/1e9<=hi]
            roles[role]=dict(kernel_busy_fraction=duration(busy[role])/span,
                kernel_idle_fraction=duration(idle[role])/span,
                copy_or_memset_union_fraction=duration(communications)/span,
                kernel_and_copy_quiet_fraction=duration(fully_quiet)/span,
                no_pending_HTTP_kernel_idle_fraction=duration(no_http_pending)/span,
                pending_HTTP_kernel_idle_fraction=duration(waiting)/span,
                idle_windows=windows(idle[role]),no_pending_HTTP_idle_windows=windows(no_http_pending),
                scheduler_observations=len(selected),
                scheduler_waiting_mean_per_observation=float(np.mean([r['before']['waiting'] for r in selected])) if selected else None,
                scheduler_waiting_max=max((r['before']['waiting'] for r in selected),default=None),
                actual_scheduled_requests_max=max((r['actual_scheduled_requests'] for r in selected),default=None),
                actual_scheduled_tokens_max=max((r['scheduled_tokens'] for r in selected),default=None),
                anchor=anchor,collector_passed=counters['passed'])
            for kind,values in [('kernel_idle',idle[role]),('no_pending_HTTP_idle',no_http_pending)]:
                window_rows.extend(dict(ROI=roi,role=role,kind=kind,start_seconds=a,end_seconds=b,duration_ms=(b-a)*1000) for a,b in values)
        both=intersection(idle['encoder'],idle['decode'])
        during_p=intersection(both,busy['prefill'])
        result['ROIs'][roi]=dict(left_monotonic=lo,right_monotonic=hi,duration_seconds=span,roles=roles,
            E_D_simultaneous_kernel_idle_fraction=duration(both)/span,
            E_D_simultaneous_idle_during_P_busy_fraction_total=duration(during_p)/span,
            E_D_simultaneous_idle_fraction_of_P_busy=duration(during_p)/duration(busy['prefill']) if duration(busy['prefill']) else None,
            P_busy_E_idle_fraction_total=duration(intersection(busy['prefill'],idle['encoder']))/span,
            P_busy_D_idle_fraction_total=duration(intersection(busy['prefill'],idle['decode']))/span,
            all_kernel_active_fraction=duration(intersection(intersection(busy['encoder'],busy['prefill']),busy['decode']))/span)
    write(ROOT/'analysis'/f'{name}_idle_analysis.json',result)
    with (ROOT/'analysis'/f'{name}_idle_windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ROI','role','kind','start_seconds','end_seconds','duration_ms'])
        writer.writeheader();writer.writerows(window_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for role in ['encoder','decode']:
        x=np.sort([r['duration_ms'] for r in window_rows if r['ROI']=='central_arrival_window' and r['role']==role and r['kind']=='kernel_idle'])
        if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=role)
    ax.set_xscale('log');ax.set_xlabel('Physical kernel gap (ms)');ax.set_ylabel('Window-count CDF');ax.legend();ax.grid(alpha=.2)
    ax.set_title('DIAGNOSTIC physical idle | Not safe admission budgets');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}_E_D_idle_CDF.{ext}',dpi=160)
    plt.close(fig)
    preserve();print(json.dumps({roi:{'busy':{r:v['kernel_busy_fraction'] for r,v in x['roles'].items()},
        'simultaneous_idle_during_P_busy':x['E_D_simultaneous_idle_fraction_of_P_busy']} for roi,x in result['ROIs'].items()}))


if __name__=='__main__':main()
