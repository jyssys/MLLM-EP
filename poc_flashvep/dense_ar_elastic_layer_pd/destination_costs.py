"""Measure legal destination batches separately from source batch size."""
import fcntl
from .common import *
from .run import execute
from .analyze import compare,records,percentile


def main():
    jobs=[]
    for destination in (8,16):
        for layers in (2,4,8,12):
            cfg=json.loads((ROOT/'configs/fresh_cost_B4_Last024812_N40_O2_v1.json').read_text())
            count=2*destination
            cfg.update(name=f'cost_Source4_Destination{destination}_Last{layers}_N{count}_v1',
                count=count,dataset='fresh_calibration.json',request_caps=[2]*count,arrival_offsets=[0.]*count,
                forced_layers=[layers],D_admission_batch=destination,D_token_budget=destination*8000,
                D_cohort_collect_ms=6000,hidden_destination_slots=16,max_reserved_requests=64,
                timeout=600,wall_timeout=1000,phase='FORCED_DESTINATION_SHAPE_COST_NOT_ONLINE_SERVING',
                scope='Collect real transferred requests for a legal destination shape. Artificial collection waiting is not a serving result or handoff cost.')
            save(cfg);jobs.append(cfg)
    results=[]
    try:
        for cfg in jobs:
            name=cfg['name']
            if not execute(name,restore=False):break
            output=compare('fresh_calibration_reference_Native_O16_N40_v1',name,True)
            folder=ROOT/'raw'/name
            frames=json.loads((folder/'CUDA_event_enclosures_decode.json').read_text())['rows']
            suffix=[r for r in frames if r['prefilling'] and r.get('cut')==36-cfg['forced_layers'][0]]
            selected=[r for r in suffix if len(r['requests'])==cfg['D_admission_batch']]
            result=dict(name=name,output=output,source_batch=4,destination_batch=cfg['D_admission_batch'],
                layers=cfg['forced_layers'][0],actual_suffix_batch_sizes=sorted({len(r['requests']) for r in suffix}),
                matching_samples=len(selected),suffix_event_p50_ms=percentile([r['elapsed_ms'] for r in selected],.5),
                suffix_event_p95_ms=percentile([r['elapsed_ms'] for r in selected],.95),
                shape_measured=bool(selected),evidence='MEASURED',
                scope='CUDA stream-event enclosures for actual completed suffix work. Exclude artificial collection waiting. Small samples do not establish a p99 bound.')
            results.append(result);write(ROOT/'state/destination_cost_progress.json',dict(results=results))
            print('DESTINATION_COST',name,output['passed'],result['actual_suffix_batch_sizes'],result['suffix_event_p50_ms'],flush=True)
            if not output['passed']:break
        write(ROOT/'analysis/destination_batch_costs.json',dict(results=results,
            scope='Source batch4 and destination batch8/16 are distinct dimensions. Unobserved shapes remain NOT_MEASURED, never zero.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
