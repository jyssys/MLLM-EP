"""Independent post-run reanalysis of actual raw serving/component evidence.

No fitted latency model, simulated samples, or automatic positive verdict.
Run only on completed directories. Discovery selection remains frozen.
"""
import argparse
import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json
from poc_native_lp_ep4.scripts.analyze import clean_rows, summarize, trace_rows
from poc_native_lp_ep4.scripts.aggregate import component_waves, component_summary
from poc_native_lp_ep4.scripts.evidence import (estimate, instrumentation_overhead,
    modality_deltas, per_layer_components, route_invariants, victim_dispatch_excess,
    restart_sensitivity)
from poc_native_lp_ep4.scripts.lifecycle import (lifecycle_rows, summarize_lifecycle, dispatch_shape)


def require_complete(run):
    status=read_json(run/'status.json')
    if not status.get('ok',False):raise ValueError(f'incomplete execution: {run}')
    return run


def policy_deltas(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r['arm']!='DECODE_ONLY':groups[(r['modality'],r['repeat'])][r['arm']]=r
    result=[]
    arms=sorted({r['arm'] for r in rows if r['arm']!='DECODE_ONLY'})
    for modality in sorted({r['modality'] for r in rows if r['arm']!='DECODE_ONLY'}):
        for arm in arms:
            for control in ('FULL','CP512','LP_AUTO'):
                if arm==control:continue
                pairs=[v for (m,_),v in groups.items() if m==modality and arm in v and control in v]
                fields={}
                for key in ('ttft_ms','lm_start_to_first_token_ms','peak_itl_ms','peak_vsel_ms','peak_itl_ratio','p99_itl_ms'):
                    values=[v[arm][key]-v[control][key] for v in pairs
                            if v[arm].get(key) is not None and v[control].get(key) is not None]
                    fields[key+'_delta']=estimate(values)
                result.append({'arm':arm,'control':control,'modality':modality,'pairs':len(pairs),**fields})
    return result


def write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def route_identity_comparison(run):
    waves=read_json(run/'manifest.json')['waves']
    events=[json.loads(line) for path in run.glob('native*.jsonl') for line in path.open()]
    indexed=defaultdict(dict)
    for e in events:
        if e.get('diagnostic')=='route_identity':indexed[e['wave']][(e['layer'],e['ep_rank'])]=e
    result=[]
    for number,w in enumerate(waves):
        if w['repeat']<0 or not w['arm'].startswith('LP_'):continue
        ref=next(i for i,c in enumerate(waves) if c['arm']=='FULL'
                 and c['modality']==w['modality'] and c['repeat']==w['repeat'])
        a,b=indexed[ref],indexed[number]
        same=len(a)==len(b)==96 and a.keys()==b.keys() and all(
            all(a[key][field]==b[key][field] for field in ('rows','ids_sha256','weights_sha256')) for key in a)
        result.append({'arm':w['arm'],'modality':w['modality'],'repeat':w['repeat'],
            'source_rank_layer_records':len(b),'same_exact_ids_weights':same,
            'scope':'EXACT_PREFILL_ROUTE_SHA256; source SP rows, not TP-duplicated logical paths'})
    return result


def route_geometry(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['prefill_active']:groups[(r['run'],r['wave'])].append(r)
    result=[]
    for (run,wave),items in groups.items():
        first=items[0];prompt=2029 if first['modality']=='VISUAL_PREFILL' else 2033
        if any(r['injected_histogram'] is None or r['logical_injected_dispatch_matrix'] is None for r in items):
            continue
        hist=np.asarray([r['injected_histogram'] for r in items],dtype=np.int64)
        traffic=np.asarray([r['logical_injected_dispatch_matrix'] for r in items],dtype=np.int64).sum(0)
        assignments=hist.sum(0).reshape(4,32).sum(1)
        per_layer=defaultdict(lambda:np.zeros(128,dtype=np.int64))
        for r,h in zip(items,hist):per_layer[r['layer']]+=h
        layer_skew=[h.reshape(4,32).sum(1).max()/h.reshape(4,32).sum(1).mean()
                    for h in per_layer.values() if h.sum()]
        send=traffic.sum(1);recv=traffic.sum(0)
        result.append({'run':run,'wave':wave,'arm':first['arm'],'modality':first['modality'],
            'repeat':first['repeat'],'logical_hidden_dispatch_bytes':int(traffic.sum()),
            'logical_hidden_combine_bytes':int(traffic.sum()),
            'mean_distinct_destination_ranks_per_prompt_token':float(traffic.sum()/(prompt*48*2048*2)),
            'per_rank_assignments':assignments.tolist(),
            'aggregate_rank_assignment_max_mean':float(assignments.max()/assignments.mean()),
            'mean_layer_rank_assignment_max_mean':float(np.mean(layer_skew)),
            'per_rank_logical_send_bytes':send.tolist(),'per_rank_logical_recv_bytes':recv.tolist(),
            'per_rank_logical_remote_send_bytes':(send-np.diag(traffic)).tolist(),
            'per_rank_logical_remote_recv_bytes':(recv-np.diag(traffic)).tolist(),
            'traffic_matrix':traffic.tolist(),
            'counter_scope':'ACTUAL_SOURCE_SP_ROUTES; DESTINATION_DEDUP_HIDDEN_PAYLOAD_ONLY; NOT_METADATA_WIRE_OR_HBM'})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v4')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,
                   help='verified restored compact runs; default uses local actual runs')
    p.add_argument('--extra-component-runs',type=Path,nargs='*',default=[])
    a=p.parse_args();root=Path('poc_native_lp_ep4');runs=a.raw_root or root/'artifacts/runs'
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    v=a.version
    def actual_run(saved):
        return require_complete((a.raw_root/Path(saved).name) if a.raw_root else Path(saved))
    if v in ('v5','v6'):
        plan=read_json(root/'artifacts'/f'campaign_blocks_{v}.json')
        assert plan['status']=='MEASUREMENTS_COMPLETE'
        discovery=[actual_run(p) for p in plan['discovery_runs']]
        final=[actual_run(p) for p in plan['final_runs']]
        components=[actual_run(p) for p in plan['component_runs']]
        routes=[actual_run(p) for p in plan['route_runs']]
        extra=[actual_run(p) for p in plan['extra_component_runs']]
    else:
        discovery=[require_complete(runs/('discovery_native_'+v))]
        final=[require_complete(runs/('final_paired_native_'+v))]
        components=[require_complete(runs/('component_overhead_native_'+v))]
        routes=[require_complete(runs/('route_accounting_native_'+v))]
        extra=[]
    ds=sum((clean_rows(p) for p in discovery),[])
    fs=sum((clean_rows(p) for p in final),[])
    cs=sum((clean_rows(p) for p in components),[])
    layers=sum((trace_rows(p) for p in components),[])
    for run in extra+a.extra_component_runs:layers+=trace_rows(require_complete(run))
    cw=component_waves(layers)
    rl=sum((trace_rows(p) for p in routes),[]);rw=component_waves(rl)
    route_checks=route_invariants(rl,{'VISUAL_PREFILL':2029,'TOKEN_MATCHED_TEXT':2033})
    lifecycle=sum((lifecycle_rows(p) for p in components),[])
    result={'schema':'native-lp-actual-metrics-v1','version':v,
        'discovery_summary':summarize(ds),'final_summary':summarize(fs),
        'final_paired_policy_deltas':policy_deltas(fs),'final_modality_deltas':modality_deltas(fs),
        'final_restart_cluster_sensitivity':restart_sensitivity(fs),
        'component_summary':component_summary(cw),'component_waves':cw,
        'per_layer_components':per_layer_components(layers),
        'component_modality_deltas':modality_component_deltas(cw),
        'instrumentation_overhead':instrumentation_overhead(cs),
        'route_summary':component_summary(rw),'route_checks':route_checks,
        'route_geometry_waves':route_geometry(rl),
        'all_logical_path_invariants_pass':bool(route_checks) and all(r['pass'] for r in route_checks),
        'current_rich_route_identity':sum((route_identity_comparison(p) for p in routes),[]),
        'all_four_rank_component_frames':bool(layers) and all(r['ranks']==4 for r in layers),
        'lifecycle_summary':summarize_lifecycle(lifecycle),
        'dispatch_shape':dispatch_shape(layers),
        'victim_dispatch_excess':sum((victim_dispatch_excess(trace_rows(p),read_json(p/'manifest.json')['waves']) for p in components),[]),
        'selection':read_json(root/'artifacts/manifests'/('frozen_final_selection_'+v+'.json')),
        'limits':['Clean engine-completion clock; CPU rendering before model-ready arrival.',
            'Local DeepEP prepare CUDA duration includes peer input-readiness waits.',
            'Route-rich times excluded; logical payload/weight bytes are not wire/HBM measurements.',
            'Primary paired wave CI plus complete engine-restart cluster sensitivity; same bounded request pool, not independent datasets.',
            'No simulator, replay substitution, or new communication method.']}
    atomic_json(a.output/'metrics.json',result)
    write_csv(a.output/'clean_discovery.csv',ds);write_csv(a.output/'clean_final.csv',fs)
    write_csv(a.output/'component_waves.csv',cw);write_csv(a.output/'lifecycle.csv',lifecycle)
    print('REANALYSIS_COMPLETE',a.output,flush=True)


def modality_component_deltas(rows):
    groups=defaultdict(dict)
    for row in rows:groups[(row['arm'],row['repeat'])][row['modality']]=row
    result=defaultdict(lambda:defaultdict(list))
    for (arm,_),pair in groups.items():
        if set(pair)!={'VISUAL_PREFILL','TOKEN_MATCHED_TEXT'}:continue
        v,t=pair['VISUAL_PREFILL'],pair['TOKEN_MATCHED_TEXT']
        for key in ('dispatch_total_ms','expert_total_ms','combine_total_ms','moe_total_ms','max_dispatch_quantum_ms'):
            result[arm][key+'_delta'].append(v[key]-t[key])
            if t[key]>0:result[arm][key+'_percent'].append(100*(v[key]/t[key]-1))
    return {arm:{key:estimate(vs) for key,vs in fields.items()} for arm,fields in result.items()}


if __name__=='__main__':main()
