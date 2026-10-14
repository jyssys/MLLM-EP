"""Copy assumptions vs a deliberately generous zero-expert-time envelope.

Not a deployable migration policy. Neither expert-byte balance nor this
fixed-other-stages envelope is a measured S2/E2E speedup.
"""
import csv
import gzip
import json
import statistics
import numpy as np
from poc_looped_moe_ep.contract import *
from poc_dllm_ep_comm.metrics import bootstrap_mean


def main():
    with gzip.open(ROOT/'results/per_loop_rank_metrics.csv.gz','rt') as inp:
        rows=[r for r in csv.DictReader(inp) if r['placement']=='S1']
    copies=json.loads((ROOT/'results/raw/copy_probe/p2p.json').read_text())['rows']
    cost={(r['src'],r['dst']):statistics.mean(r['samples_ms']) for r in copies if r['mode']=='actual_3_tensors'}
    bulk_bw={(a,b):max(r['bytes']/statistics.mean(r['samples_ms']) for r in copies
               if r['src']==a and r['dst']==b and r['mode']=='contiguous') for a,b in cost}
    static=json.loads((ROOT/'results/frozen_best_static.json').read_text())['placement']
    results=[];episodes=[]
    for request in sorted({r['request_id'] for r in rows}):
        previous=static;serial=parallel=bulk_parallel=expert=0
        selected=sorted([r for r in rows if r['request_id']==request],key=lambda r:(int(r['forward']),int(r['loop'])))
        for row in selected:
            placement=json.loads(row['S2_candidate_placement']);edges=np.zeros((4,4),dtype=int)
            for old,new in zip(previous,placement):
                if old!=new:edges[old,new]+=1
            serial_ms=sum(edges[src,dst]*value for (src,dst),value in cost.items())
            parallel_ms=max(edges[src,dst]*value for (src,dst),value in cost.items())
            bulk_parallel_ms=max(edges[src,dst]*9*1024**2/bw for (src,dst),bw in bulk_bw.items())
            # Over-generous: permit eliminating ALL measured routed expert time.
            # D/C/restoration held fixed; no claim that this is a universal bound.
            expert_ms=float(row['expert_ms']);serial+=serial_ms;parallel+=parallel_ms
            bulk_parallel+=bulk_parallel_ms;expert+=expert_ms
            results.append(dict(request_id=request,forward=int(row['forward']),loop=int(row['loop']),
                moved_experts=int(edges.sum()),copy_bytes=int(edges.sum())*9*1024**2,
                serial_copy_ms=serial_ms,perfect_parallel_edge_copy_ms=parallel_ms,
                zero_launch_bulk_parallel_copy_ms=bulk_parallel_ms,
                fixed_other_stages_zero_expert_envelope_ms=expert_ms,
                optimistic_envelope_over_parallel_copy=expert_ms/parallel_ms if parallel_ms else None,
                source_destination_copy_counts=edges.tolist()))
            previous=placement
        episodes.append(dict(request_id=request,serial_copy_ms=serial,
            perfect_parallel_edge_copy_ms=parallel,zero_expert_envelope_ms=expert,
            zero_launch_bulk_parallel_copy_ms=bulk_parallel,
            envelope_over_serial=expert/serial if serial else None,
            envelope_over_zero_launch_bulk=expert/bulk_parallel if bulk_parallel else None,
            envelope_over_parallel=expert/parallel if parallel else None))
    summary={key:bootstrap_mean([r[key] for r in episodes if r[key] is not None]) for key in
             ['serial_copy_ms','perfect_parallel_edge_copy_ms','zero_expert_envelope_ms',
              'zero_launch_bulk_parallel_copy_ms','envelope_over_serial',
              'envelope_over_parallel','envelope_over_zero_launch_bulk']}
    write(ROOT/'results/migration_roi.json',dict(summary=summary,episodes=episodes,invocations=results,
        expert_bytes=9*1024**2,P2P=cost and [dict(src=a,dst=b,ms=v) for (a,b),v in cost.items()],
        copy_accounting='hypothetical future-truth per-invocation mappings; all ownership changes charged',
        assumptions='serial isolated-edge sum vs unrealistically perfect all-edge parallel maximum; no allocation/coordination/cache cost',
        savings_domain='FIXED_OTHER_STAGES_ZERO_EXPERT_TIME_ENVELOPE; NOT actual oracle savings',
        no_measured_migration_or_E2E_gain=True))
    print('Copy/zero-expert envelope:',summary)


if __name__=='__main__':main()
