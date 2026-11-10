"""Compute observed kernel unions and explicitly unconstrained packing bounds."""
import argparse
import csv
from collections import Counter
import numpy as np
from .common import *
from .analyze import metrics

OUT=ROOT/'idle_slack_oracle'


def union(intervals,lower,upper):
    intervals=np.asarray(intervals,dtype=np.int64).reshape((-1,2))
    keep=(intervals[:,1]>lower)&(intervals[:,0]<upper)
    x=intervals[keep].copy()
    if not len(x):return np.empty((0,2),dtype=np.int64)
    x[:,0]=np.maximum(x[:,0],lower);x[:,1]=np.minimum(x[:,1],upper)
    x=x[np.argsort(x[:,0],kind='stable')]
    ends=np.maximum.accumulate(x[:,1])
    boundaries=np.r_[True,x[1:,0]>ends[:-1]]
    starts=np.flatnonzero(boundaries)
    stops=np.r_[starts[1:]-1,len(x)-1]
    return np.column_stack((x[starts,0],ends[stops]))


def complement(merged,lower,upper):
    starts=np.r_[lower,merged[:,1]];ends=np.r_[merged[:,0],upper]
    return np.column_stack((starts[ends>starts],ends[ends>starts]))


def gaps_summary(gaps):
    ms=(gaps[:,1]-gaps[:,0])/1e6
    if not len(ms):return dict(count=0,total_seconds=0.)
    return dict(count=len(ms),total_seconds=float(ms.sum()/1000),mean_ms=float(ms.mean()),
        **{f'p{q}_ms':float(np.percentile(ms,q)) for q in (50,90,95,99)},maximum_ms=float(ms.max()),
        thresholds={str(threshold):dict(count=int((ms>=threshold).sum()),
            total_seconds=float(ms[ms>=threshold].sum()/1000)) for threshold in (1,2,5,10,20,30,50)})


def anchor_offset(value):
    a=value['anchor']
    return (a['monotonic_lower_ns']+a['monotonic_upper_ns'])//2-a['cupti_ns']


def records(folder):
    return [r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]


def primitive_costs(model):
    path=OLD/'new_primitive_costs.json'
    if not path.exists():return []
    values=json.loads(path.read_text());out=[]
    for value in values:
        if value['model']!=model:continue
        groups=[g for g in value['groups'] if g['role']=='decode' and g['operation']=='recurrent']
        for g in groups:out.append(dict(source_run=value['name'],**g))
    return out


def packing(gaps,model,requests=128):
    # Each quantum occupies an entire measured stream enclosure. No kernel is
    # split across gaps. Request readiness and transfer remain excluded here.
    cfg=json.loads((ROOT/'configs'/f'p1_{model}_loop_clean24.json').read_text())
    costs=primitive_costs(model);rows=[]
    durations=(gaps[:,1]-gaps[:,0])/1e6
    for width in (cfg['length'],128,256,512,1024):
        if cfg['length']%width:continue
        selected=[g for g in costs if g['tokens']==width and g['loops']==1 and not g['terminal']]
        terminal=[g for g in costs if g['tokens']==width and g['loops']==1 and g['terminal']]
        if not selected:
            rows.append(dict(tokens=width,status='NOT_MEASURED'));continue
        # This relaxed bound uses the fastest measured median. It ignores the
        # terminal coda and all readiness constraints. It is not a safe cost.
        core=min(g['p50_ms'] for g in selected)
        final=max((g['p99_ms'] for g in terminal),default=None)
        cost=core
        quanta=int(np.floor(durations/cost).sum());tiles=cfg['length']//width
        loops=quanta//tiles
        rows.append(dict(tokens=width,status='RELAXED_OFFLINE_COST_MODEL_BOUND_NOT_VERIFIED_CAPACITY',
            core_quantum_fastest_measured_p50_ms=cost,terminal_quantum_slowest_measured_p99_ms=final,
            tiles_per_full_loop=tiles,windows_with_one_quantum=int((durations>=cost).sum()),
            ready_free_quantum_count=quanta,completed_full_loop_upper_bound=min(loops,requests*(3 if model=='ouro' else 8)),
            idle_total_divided_by_loop_cost=int(durations.sum()//(tiles*cost)),
            qualified_request_cut_cap=requests*(3 if model=='ouro' else 8),
            calibration_runs=sorted({g['source_run'] for g in selected+terminal}),
            excludes=['terminal coda','request ready time','KV/state handoff','ACK','CPU scheduling','per-request completion deadline'],
            terminal_cost_measured=bool(terminal)))
    return rows


def analyze(name):
    configure();OUT.mkdir(exist_ok=True)
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    receipt=json.loads((folder/'dispatcher_result.json').read_text())
    if not receipt.get('complete'):
        result=dict(name=name,status='FAILED_CAPTURE',receipt=receipt,
            native_D_idle='NOT_MEASURED',Decode_safe_oracle='NOT_MEASURED',Completion_feasible_oracle='NOT_MEASURED')
        write(OUT/(name+'.json'),result);return result
    req=records(folder);lower=int(min(r['arrival'] for r in req)*1e9);upper=int(max(r['finish'] for r in req)*1e9)
    result=dict(name=name,model=cfg['model'],arrival_seed=cfg['arrival_seed'],count=len(req),
        cohort_window_seconds=(upper-lower)/1e9,metrics=metrics(name),roles={})
    mapping=json.loads((folder/'worker_map.json').read_text())
    for role in ('prefill','decode'):
        path=folder/f'full_activity_{role}.csv';begin=folder/f'full_activity_{role}_begin.json';end=folder/f'full_activity_{role}_end.json'
        if not path.exists() or not end.exists():
            result['roles'][role]=dict(status='NOT_MEASURED');continue
        a,b=json.loads(begin.read_text()),json.loads(end.read_text());offset=anchor_offset(a)
        offset_end=anchor_offset(b);capture_start=a['anchor']['cupti_ns']+offset;capture_end=b['anchor']['cupti_ns']+offset
        lo,hi=max(lower,capture_start),min(upper,capture_end)
        first=path.open().readline();columns=(1,2,5) if first.startswith('kind,') else (0,1,3)
        raw=np.loadtxt(path,delimiter=',',skiprows=1,usecols=columns,dtype=np.int64,ndmin=2)
        device=next(v['visible_device'] for v in mapping.values() if v['role']==role)
        device_counts={str(int(d)):int((raw[:,2]==d).sum()) for d in np.unique(raw[:,2])}
        intervals=raw[raw[:,2]==device,:2].copy()
        intervals+=offset;merged=union(intervals,lo,hi);gaps=complement(merged,lo,hi)
        seconds=float((merged[:,1]-merged[:,0]).sum()/1e9);duration=(hi-lo)/1e9
        item=dict(status='MEASURED_IN_INSTRUMENTED_EXECUTION',window_seconds=duration,
            start_monotonic=lo/1e9,end_monotonic=hi/1e9,
            leading_uncovered_ms=max(0,(capture_start-lower)/1e6),trailing_uncovered_ms=max(0,(upper-capture_end)/1e6),
            kernel_records=len(intervals),visible_device=device,record_device_counts=device_counts,
            busy_seconds=seconds,busy_percent=100*seconds/duration,
            physical_idle_seconds=duration-seconds,physical_idle_percent=100*(duration-seconds)/duration,
            idle_windows=gaps_summary(gaps),collector=b,clock_offset_drift_ns=offset_end-offset,
            scope='All-stream union for the selected model-stage process and device. Not SM occupancy. Copies are not measured in the kernel-only collector.',
            cross_process_device_activity_scope='Inspect both stage record-device inventories before calling this a complete physical-device union.')
        np.savez_compressed(OUT/f'{name}_{role}_intervals.npz',busy=merged,idle=gaps,lower=lo,upper=hi)
        if role=='decode':
            item['physical_packing_bounds']=packing(gaps,cfg['model'],len(req))
            item['Decode_safe_oracle']='NOT_MEASURED'
            item['Completion_feasible_oracle']='NOT_MEASURED'
            item['unresolved_constraints']=['Native graph internal cut ready timestamps','new transfer/admission cost in the same execution',
                'continuation state ownership and completion under changed batching','profiler interference gate']
        result['roles'][role]=item
    native=[metrics(n) for n in (f"baseline_{cfg['model']}_clean128_calibration",f"aa2_{cfg['model']}_native_clean128",f"aa3_{cfg['model']}_native_clean128")]
    result['instrumentation_comparison']={k:dict(clean_values=[v[k] for v in native],
        instrumented=result['metrics'][k],ratio_to_clean_median=result['metrics'][k]/float(np.median([v[k] for v in native])))
        for k in ('completed_QPS','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')}
    eligible=all(abs(result['instrumentation_comparison'][k]['ratio_to_clean_median']-1)<=.05
        for k in ('completed_QPS','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms'))
    result['profiler_5_percent_gate']=eligible
    result['maximum_ITL_within_observed_AA_range']=result['metrics']['maximum_ITL_ms']<=max(v['maximum_ITL_ms'] for v in native)
    result['eligible_as_clean_native_capacity_evidence']=eligible and result['maximum_ITL_within_observed_AA_range']
    result['qualification_scope']='Calibration-seed A/A comparison only. Three starts do not establish a hard tail guarantee.'
    write(OUT/(name+'.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    for name in a.names:
        r=analyze(name);print(name,r.get('status'),r.get('profiler_5_percent_gate'),
            {k:v.get('physical_idle_percent') for k,v in r.get('roles',{}).items()})
