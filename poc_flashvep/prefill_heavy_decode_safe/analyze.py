"""Separate physical intervals, host ownership, and profiler qualification."""
import argparse,csv,json,statistics
from .common import *
from .gates import interference
from ..loop_width_gemm_validation.cost_analysis import quantile


def summary(values):
    values=list(values)
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        **{f'p{q}':quantile(values,q) for q in (50,90,95,99)},maximum=max(values,default=None))


def merged(values):
    out=[]
    for a,b in sorted(values):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out


def gaps(intervals,left,right):
    union=merged((max(left,a),min(right,b)) for a,b in intervals if a<right and b>left)
    out=[];cursor=left
    for a,b in union:
        if a>cursor:out.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<right:out.append((cursor,right))
    return union,out


def csv_write(path,rows,fields=None):
    rows=list(rows)
    if not rows and fields is None:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def native(name,reference=None):
    configure()
    from ..huginn_net_benefit.common import analysis
    from ..looped_pd_atomic_prefill.evidence import full_work,compare
    from ..dynamic_loop_split_serving.lifetime_gate import check
    value=analysis().metrics(name)
    if not value['complete']:return value
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    if cfg.get('campaign_frozen_SLO'):
        value['inherited_length_specific_SLO_metadata']=value['frozen_SLO']
        value['frozen_SLO']=cfg['campaign_frozen_SLO']
        value['SLO_scope']=cfg['SLO_scope']
    work,lifetime=full_work(name),check(name)
    folder=ROOT/'raw'/name
    exits=[json.loads(p.read_text()) for p in folder.glob('native_worker_exit_*.json')]
    normal=any(x['passed'] and {r['role'] for r in x['records'] if r['exitcode']==0 and not r['forced_cleanup']}=={'prefill','decode'} for x in exits)
    parity=compare(reference,name) if reference else dict(passed=True,scope='Self reference only. A candidate needs a separate paired output comparison.')
    value.update(work=work,lifetime=lifetime,normal_worker_exit=normal,parity=parity,
        correctness_passed=work['passed'] and lifetime['passed'] and normal and parity['passed'],
        response_wall_seconds=value['count']/value['completed_QPS'])
    write(folder/'discovery_metrics.json',value);return value


def physical(name):
    folder=ROOT/'raw'/name
    meta={role:json.loads((folder/f'physical_activity_{role}.json').read_text()) for role in ('prefill','decode')}
    left=max(r['start']['host_midpoint'] for r in meta.values());right=min(r['stop']['host_midpoint'] for r in meta.values())
    assert right>left
    results={}
    for role,v in meta.items():
        start,stop=v['start'],v['stop'];scale=(stop['host_midpoint']-start['host_midpoint'])/(stop['cupti_ns']-start['cupti_ns'])
        def host(ns):return start['host_midpoint']+(int(ns)-start['cupti_ns'])*scale
        kernel=[];copy=[];streams=set()
        with (folder/f'physical_activity_{role}.csv').open() as f:
            for r in csv.DictReader(f):
                a,b=host(r['start_ns']),host(r['end_ns'])
                if b<=left or a>=right:continue
                (kernel if r['kind']=='kernel' else copy).append((a,b));streams.add(r['stream'])
        union,idle=gaps(kernel,left,right);durations=[1000*(b-a) for a,b in idle];busy=sum(b-a for a,b in union)
        results[role]=dict(run=name,role=role,left=left,right=right,window_seconds=right-left,
            actual_kernel_records=len(kernel),streams=sorted(streams),kernel_union_seconds=busy,
            kernel_busy_percent=100*busy/(right-left),physical_kernel_idle_seconds=(right-left)-busy,
            physical_kernel_idle_percent=100*(1-busy/(right-left)),idle_window_count=len(idle),idle_ms=summary(durations),
            thresholds={str(t):dict(count=sum(d>=t for d in durations),seconds=sum(d for d in durations if d>=t)/1000) for t in (1,2,5,10,20,30,50,100)},
            collector_status=v['status'],collector_counters=v['counters'],
            collector_valid=v['status']==0 and v['counters']['invalid']==v['counters']['dropped']==0 and len(kernel)>0,
            clock_bracket_max_ms=1000*max(r['host_upper']-r['host_lower'] for r in (start,stop)),
            copy_activity='MEASURED' if copy else 'NOT_MEASURED_OR_NO_RECORDED_COPIES',
            scope='Actual all-stream kernel interval union inside the same captured run/window. Profiling sensitivity is a separate gate. Physical kernel inactivity is not automatically Decode-safe slack.')
        csv_write(folder/f'physical_{role}_union.csv',(dict(start=a,end=b) for a,b in union),('start','end'))
        csv_write(folder/f'physical_{role}_idle_windows.csv',(dict(start=a,end=b,duration_ms=1000*(b-a)) for a,b in idle),('start','end','duration_ms'))
    write(folder/'physical_interval_analysis.json',results)
    return results


def local_itl(name,offset_left,offset_right):
    folder=ROOT/'raw'/name;origin=json.loads((folder/'measurement_begin.json').read_text())['time']
    rows=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
    values=[1000*(b['time']-a['time'])/b['new_tokens'] for r in rows for a,b in zip(r['token_events'],r['token_events'][1:])
        if offset_left<=a['time']-origin and b['time']-origin<=offset_right]
    return summary(values)


def anchor_controls(plan=None):
    initialize()
    if plan is None:
        plan=ROOT/'state/anchor_controls_v2.json'
        if not plan.exists():plan=ROOT/'state/initial_anchor_controls.json'
    else:plan=Path(plan)
    jobs=json.loads(plan.read_text())['jobs'];out=[]
    for index in (0,3):
        names=jobs[index:index+3];metrics={name:native(name,names[0]) for name in names}
        if not all(m['complete'] for m in metrics.values()):out.append(dict(names=names,status='NOT_COMPLETE'));continue
        fields=('response_wall_seconds','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
        screens={name:interference(metrics[names[1]],metrics[name],fields) for name in (names[0],names[2])}
        AA=interference(metrics[names[2]],metrics[names[0]],fields)
        try:trace=physical(names[1])
        except (FileNotFoundError,AssertionError) as e:
            out.append(dict(names=names,status='FAILED_OR_MISSING_CAPTURE',reason=repr(e),metrics=metrics));continue
        left,right=trace['prefill']['left'],trace['prefill']['right'];folder=ROOT/'raw'/names[1]
        origin=json.loads((folder/'measurement_begin.json').read_text())['time']
        local={name:local_itl(name,left-origin,right-origin) for name in names}
        local_screens={}
        for name in (names[0],names[2]):
            if all(local[x]['count']>=20 for x in (name,names[1])):
                local_screens[name]=interference(local[names[1]],local[name],('p99','maximum'))
            else:local_screens[name]=dict(passed=False,reason='Too few local-window intervals for this point screen')
        qualified=all(x['passed'] for x in screens.values()) and all(x['passed'] for x in local_screens.values()) and all(x['collector_valid'] for x in trace.values()) and all(x['correctness_passed'] for x in metrics.values())
        out.append(dict(names=names,status='QUALIFIED_PROFILER_PILOT' if qualified else 'DIAGNOSTIC_ONLY',
            metrics=metrics,Native_AA=AA,whole_run_screens=screens,local_screens=local_screens,local_ITL=local,
            trace=trace,qualified=qualified,
            Phase_1='NOT_YET_QUALIFIED: independent capacity, queue/readiness, source feasibility, and repeated starts remain required'))
    write(ROOT/'analysis'/(plan.stem+'_results.json'),out)
    print(json.dumps([dict(names=r['names'],status=r['status'],screens=r.get('whole_run_screens'),local=r.get('local_screens'),
        physical={k:{j:v[j] for j in ('kernel_busy_percent','physical_kernel_idle_percent','idle_window_count')} for k,v in r.get('trace',{}).items()}) for r in out]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--anchors',action='store_true');p.add_argument('--plan');p.add_argument('--name');args=p.parse_args()
    if args.anchors:anchor_controls(args.plan)
    elif args.name:print(json.dumps(physical(args.name)))
