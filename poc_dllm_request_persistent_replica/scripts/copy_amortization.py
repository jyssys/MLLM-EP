"""One request copy, measured directed P2P; byte and unvalidated time ROI separate."""
import json
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *


def main():
    out=ROOT/'results/copy_cost'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    p2p=json.loads((SHORT/'raw/physical_revalidation/p2p/p2p.json').read_text())
    expert=np.zeros((4,4));bw=np.zeros((4,4));raw=[]
    for row in p2p['rows']:
        ms=float(np.median(row['samples_ms']));raw.append({k:v for k,v in row.items() if k!='samples_ms'}|dict(median_ms=ms))
        if row['mode']=='actual_two_tensor_P2P':expert[row['src'],row['dst']]=ms
        if row['mode']=='contiguous_P2P' and row['bytes']==384*1024**2:bw[row['src'],row['dst']]=row['bytes']/ms
    pd.DataFrame(raw).to_csv(out/'p2p_summary.csv',index=False)
    allrows=[]
    for lane in ('short','long'):
        sim=ROOT/'results'/('simulation_'+lane+'_v2')
        df=pd.read_json(sim/'request_policy.jsonl',lines=True)
        blocks=pd.read_csv(sim/'block_policy.csv')
        baselines=pd.read_csv(ROOT/'results/long_requests.csv') if lane=='long' else pd.read_csv(COMM/'results/main_v2/requests.csv')
        # No proportional byte -> latency conversion is used for promotion.
        # Baseline D+C / copy is only an optimistic all-communication upper
        # ceiling, since replica routing speed is not measured in Stage A.
        base=baselines.set_index('request')
        for r in df.to_dict('records'):
            edges=np.array(r['copy_edges']);times=edges/EXPERT_BYTES*expert
            bulk=np.divide(edges,bw,out=np.zeros_like(edges,dtype=float),where=bw>0)
            serial=float(times.sum());parallel=float(max(times.sum(axis=0).max(),times.sum(axis=1).max()))
            b=blocks[(blocks.request==r['request'])&(blocks.policy==r['policy'])&
                     (blocks.budget_gib==r['budget_gib'])&(blocks.load_guard==r['load_guard'])].sort_values('block')
            future_saved=r['saved_bytes'];iterations=r['future_iterations']
            projected=r['copy_bytes']*iterations/future_saved if future_saved else None
            entire_dc=float(base.loc[r['request'],'comm_ms'])
            allrows.append({k:r[k] for k in ('lane','request','domain','split','policy','budget_gib','load_guard')}|dict(
                initial_copy_bytes=r['initial_copy_bytes'],update_bytes=0,total_copy_bytes=r['copy_bytes'],
                future_blocks=r['future_blocks'],future_iterations=iterations,
                avoided_dc_bytes=future_saved,byte_roi=r['byte_roi'],
                copy_serial_expert_ms=serial,copy_endpoint_parallel_est_ms=parallel,
                copy_serial_bulk_est_ms=float(bulk.sum()),
                copy_endpoint_parallel_bulk_est_ms=float(max(bulk.sum(axis=0).max(),bulk.sum(axis=1).max())),
                byte_break_even_iteration=r['byte_break_even_iteration'],byte_break_even_block=r['byte_break_even_block'],
                projected_byte_break_even_iterations=projected,
                projected_byte_break_even_remaining_fraction=projected/iterations if projected and iterations else None,
                timing_roi=None,timing_roi_status='UNRESOLVED_NO_MEASURED_REPLICA_D_C',
                all_baseline_communication_over_serial_copy_upper_ceiling=entire_dc/serial if serial else None,
                byte_roi_not_latency_ratio=True))
    pd.DataFrame(allrows).to_csv(out/'request_copy_roi.csv',index=False)
    new_json(out/'methodology.json',dict(source=str(SHORT/'raw/physical_revalidation/p2p/p2p.json'),source_sha256=sha(SHORT/'raw/physical_revalidation/p2p/p2p.json'),
        measured='6MiB actual two-tensor expert P2P and384MiB bulk,12 directed edges,30samples/edge',
        serial='sum exact expert count per edge x measured median expert copy time',
        endpoint_parallel='max summed incoming/outgoing edge time; optimistic estimate, not concurrent copy measurement',
        byte_break_even='actual cumulative avoided D+C bytes reaches total request copy bytes; if unreachable, no break-even observed',
        projection='constant observed saved bytes/refinement extrapolation, not extra executed denoising steps',
        time_roi='UNRESOLVED: no replica D+C runtime; total baseline D+C/copy ceiling is NOT a predicted saving',
        global_policy='prewarmed zero per-request preparation, matched resident memory',copy_lifetime='one request, no block reload'))


if __name__=='__main__':main()
