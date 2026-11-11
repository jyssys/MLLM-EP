"""Keep observed idle subsets separate from unobserved Decode intervals."""
import argparse,csv,json
from .common import *
from .analyze import gaps,merged,summary,csv_write,native,local_itl
from .gates import interference


def trace(name):
    folder=ROOT/'raw'/name
    meta={r:json.loads((folder/f'physical_activity_{r}.json').read_text()) for r in ('prefill','decode')}
    assert meta['decode']['quiescent_only'] and not meta['prefill']['quiescent_only']
    left=max(v['start']['host_midpoint'] for v in meta.values());right=min(v['stop']['host_midpoint'] for v in meta.values())
    assert right>left;intervals={};copies={};witness=0
    for role,m in meta.items():
        a,b=m['start'],m['stop'];scale=(b['host_midpoint']-a['host_midpoint'])/(b['cupti_ns']-a['cupti_ns'])
        def host(ns):return a['host_midpoint']+(int(ns)-a['cupti_ns'])*scale
        intervals[role]=[];copies[role]=[]
        w=m.get('post_serving_collector_witness')
        for r in csv.DictReader((folder/f'physical_activity_{role}.csv').open()):
            if w and r['kind']=='kernel' and int(r['end_ns'])>w['start']['cupti_ns'] and int(r['start_ns'])<w['stop']['cupti_ns']:witness+=1
            start,end=host(r['start_ns']),host(r['end_ns'])
            if end<=left or start>=right:continue
            (intervals if r['kind']=='kernel' else copies)[role].append((start,end))
    P_union,P_idle=gaps(intervals['prefill'],left,right)
    D_epochs=merged((max(left,e['start']['host_midpoint']),min(right,e['stop']['host_midpoint']))
        for e in meta['decode']['quiet_epochs'] if e['start']['host_midpoint']<right and e['stop']['host_midpoint']>left)
    D_idle=[];D_busy=[];D_copy_free=[]
    for a,b in D_epochs:
        union,idle=gaps(intervals['decode'],a,b);D_busy+=union;D_idle+=idle
        _,free=gaps([*intervals['decode'],*copies['decode']],a,b);D_copy_free+=free
    idle_seconds=sum(b-a for a,b in D_idle)
    value=dict(name=name,left=left,right=right,window_seconds=right-left,
        P_kernel_union_seconds=sum(b-a for a,b in P_union),
        P_kernel_busy_percent=100*sum(b-a for a,b in P_union)/(right-left),
        D_observed_quiescent_epochs=len(D_epochs),D_observed_epoch_seconds=sum(b-a for a,b in D_epochs),
        D_measured_idle_subset_seconds=idle_seconds,
        D_idle_lower_bound_percent=100*idle_seconds/(right-left),
        D_whole_window_kernel_busy_percent='NOT_MEASURED',D_total_kernel_idle_percent='NOT_MEASURED',
        D_measured_idle_subset_ms=summary(1000*(b-a) for a,b in D_idle),
        D_post_serving_witness_kernels=witness,
        collector_valid=witness>0 and all(v['status']==0 and v['counters']['dropped']==v['counters']['invalid']==0 for v in meta.values()),
        D_recorded_copy_free_subset_seconds=sum(b-a for a,b in D_copy_free),
        request_feasible_decode_safe_capacity='NOT_MEASURED',
        scope='Actual kernel union only inside observed epochs. Untraced Decode periods never become idle. Kernel inactivity excludes recorded memcpy separately, but untraced RDMA bandwidth remains unknown. The complete idle distribution and request-feasible capacity remain unmeasured.')
    csv_write(folder/'D_quiescent_observed_epochs.csv',(dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in D_epochs))
    csv_write(folder/'D_measured_idle_subset.csv',(dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in D_idle))
    write(folder/'quiescent_physical_analysis.json',value);return value


def main(plan_path=None):
    initialize();plan_path=Path(plan_path) if plan_path else ROOT/'state/quiescent_capture_plan.json'
    plan=json.loads(plan_path.read_text());a=plan['clean_reference'];events,b,repeat=plan['jobs']
    names=[a,events,b,repeat];metrics={n:native(n,a) for n in names};assert all(m['complete'] for m in metrics.values())
    v=trace(b);origin=json.loads((ROOT/'raw'/b/'measurement_begin.json').read_text())['time']
    local={n:local_itl(n,v['left']-origin,v['right']-origin) for n in names}
    fields=('response_wall_seconds','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    aa=interference(metrics[repeat],metrics[a],fields)
    whole={n:interference(metrics[b],metrics[n],fields) for n in (a,repeat)}
    tails={n:interference(local[b],local[n],('p99','maximum')) if min(local[b]['count'],local[n]['count'])>=20 else dict(passed=False) for n in (a,repeat)}
    qualified=v['collector_valid'] and aa['passed'] and all(m['correctness_passed'] for m in metrics.values()) and all(s['passed'] for s in [*whole.values(),*tails.values()])
    result=dict(metrics=metrics,trace=v,Native_AA=aa,whole_run_screens=whole,local_screens=tails,local_ITL=local,
        observer_qualified=qualified,Phase_1_qualified=False,
        scope='Partial-coverage observer point screen. Independent repeats, source readiness, handoff, KV feasibility, and complete capacity accounting remain separate. Do not claim a full idle percentage.')
    write(ROOT/'analysis'/(plan_path.stem+'_results.json'),result)
    print(json.dumps({k:r for k,r in result.items() if k not in ('metrics','local_ITL')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan');main(p.parse_args().plan)
