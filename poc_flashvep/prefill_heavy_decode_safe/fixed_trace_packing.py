"""Replay measured enclosures into fixed Native gaps. Do not predict speedup."""
import csv,json,math
from .common import *
from .analyze import summary


def pack(windows,cost_ms,ready_times,quanta_per_loop=1):
    """Complete all tiles before one loop counts as complete."""
    assert cost_ms>0 and quanta_per_loop>=1
    waiting=sorted(ready_times.items(),key=lambda row:row[1]);cursor=0;remaining={};done=[];placements=[]
    for left,right in sorted(windows):
        now=left
        while now+cost_ms/1000<=right:
            available=[seed for seed,t in waiting if t<=now and seed not in done]
            if not available:
                future=[t for seed,t in waiting if seed not in done and t>now]
                if not future:break
                now=max(now,min(future));continue
            seed=available[0];before=remaining.setdefault(seed,quanta_per_loop)
            end=now+cost_ms/1000;placements.append(dict(seed=seed,start=now,end=end))
            remaining[seed]=before-1;now=end
            if remaining[seed]==0:done.append(seed)
    return dict(completed_full_loops=len(done),completed_requests=len(done),
        completed_seeds=done,executed_tiles=len(placements),
        unfinished_partial_tiles=sum(quanta_per_loop-v for s,v in remaining.items() if s not in done),
        placements=placements)


def cost(name):
    from ..loop_width_gemm_validation import cost_analysis
    cost_analysis.ROOT=ROOT;cost_analysis.configure=configure
    value,operations=cost_analysis.analyze(name);assert value['passed']
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    units=[r['ms'] for r in operations if r['role']=='decode' and r['loops']==1 and r['tokens']>1]
    assert units,'No actual one-loop quantum'
    graph=[]
    for p in (ROOT/'raw'/name).glob('core_graph_execution_*.json'):
        data=json.loads(p.read_text())
        if data['role']!='decode':continue
        graph.extend(r['ms'] for r in data['rows'] if r['tokens']>1
            and r['actual_depth']==cfg['depth']-1 and r['planned_loops']==1)
    return cfg,value,summary(units),summary(graph)


def analyze(trace_name,cost_names,kind='exact_kernel'):
    folder=ROOT/'raw'/trace_name
    if kind=='exact_kernel':
        trace=json.loads((folder/'physical_interval_analysis.json').read_text())['decode']
        path=folder/'physical_decode_idle_windows.csv'
        physical=dict(window_seconds=trace['window_seconds'],idle_seconds=trace['physical_kernel_idle_seconds'],
            idle_percent=trace['physical_kernel_idle_percent'],idle_ms=trace['idle_ms'])
    else:
        trace=json.loads((folder/'graph_envelope_bound_analysis.json').read_text())['roles']['decode']
        path=folder/'graph_bound_decode_inactive.csv'
        physical=dict(window_seconds=trace['window_seconds'],conservative_inactivity_seconds=trace['complement_seconds'],
            conservative_inactivity_percent=trace['complement_percent'],inactivity_ms=trace['complement_windows'],
            exact_kernel_idle='NOT_MEASURED')
    with path.open() as f:windows=[(float(r['start']),float(r['end'])) for r in csv.DictReader(f)]
    source=json.loads((folder/'discovery_events_prefill.json').read_text())['rows']
    cfg=json.loads((ROOT/'configs'/(trace_name+'.json')).read_text());cut=cfg['depth']-1
    ready={r['seed']:row['gpu_end_host_estimate'] for row in source if row.get('depth')==cut for r in row['requests']}
    ready={s:t for s,t in ready.items() if trace['left']<=t<=trace['right']}
    cells=[]
    for name in cost_names:
        config,measured,unit,core=cost(name);assert config['length']==cfg['length']
        assert config['native_partial_cut']==cut,'This model only admits one remaining loop'
        width=config.get('D_prefill_token_cap') or cfg['length'];tiles=math.ceil(cfg['length']/width)
        for model,distribution in (('MODEL_RUNNER_QUANTUM_ENCLOSURE',unit),('CORE_GRAPH_ONLY_OPTIMISM',core)):
            if not distribution['count']:continue
            for q in ('p50','p99','maximum'):
                duration=distribution[q];eligible=[(a,b) for a,b in windows if 1000*(b-a)>=duration]
                result=pack(eligible,duration,ready,tiles)
                cells.append(dict(cost_run=name,token_tile=width,cost_model=model,cost_quantile=q,cost_ms=duration,
                    tiles_per_completed_loop=tiles,eligible_windows=len(eligible),
                    unlimited_ready_quantum_slots=sum(math.floor(1000*(b-a)/duration) for a,b in eligible),
                    source_cut_count=len(ready),result=result,
                    handoff_model='ZERO_TRANSFER_FREE_RESERVED_STATE_OPTIMISM',
                    status='OFFLINE_ORACLE',
                    scope='Measured serial CUDA Event enclosures in a fixed Native timeline. Every full loop needs all prompt tiles. Core-only optimism removes surrounding runtime, Coda, and handoff costs. These enclosure costs are not exclusive kernel time or a universal physical upper bound. Source events do not retain export snapshots. No verified Decode-safe capacity or Serving speedup follows from this replay.'))
    value=dict(trace=trace_name,trace_kind=kind,physical=physical,cells=cells,
        source_clock_scope='Startup CUDA Event mapping. Nonlinear drift is not bounded. No retained source snapshot or counterfactual handoff is measured.',
        request_feasible_decode_safe_capacity='NOT_MEASURED',
        predicted_TTFT_or_goodput='NOT_MEASURED',
        limitation='The fixed Native timeline does not model changed Decode arrivals, handoff traffic, KV ownership, or online batching. A zero count is not a proof that every changed scheduler must fail.')
    write(folder/'fixed_trace_enclosure_packing.json',value);return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('costs',nargs='+');p.add_argument('--kind',default='exact_kernel')
    args=p.parse_args();initialize();print(json.dumps(analyze(args.trace,args.costs,args.kind)))
