"""Cold copies are charged; distinguish byte ROI from unmeasured timing ROI."""
import json
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *


def main():
    out=ROOT/'results/copy_cost'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    p2p=json.loads((ROOT/'raw/physical_revalidation/p2p/p2p.json').read_text())
    expert_ms=np.zeros((4,4));bulk_bw=np.zeros((4,4));measurements=[]
    for row in p2p['rows']:
        median=float(np.median(row['samples_ms']));gbps=row['bytes']/median/1e6
        measurements.append({k:v for k,v in row.items() if k!='samples_ms'}|dict(median_ms=median,GB_per_s=gbps))
        if row['mode']=='actual_two_tensor_P2P':expert_ms[row['src'],row['dst']]=median
        if row['mode']=='contiguous_P2P' and row['bytes']==384*1024**2:bulk_bw[row['src'],row['dst']]=row['bytes']/median
    pd.DataFrame(measurements).to_csv(out/'p2p_summary.csv',index=False)
    blocks=pd.read_json(ROOT/'results/simulation/block_policy.jsonl',lines=True)
    costs=[]
    for b in blocks.to_dict('records'):
        edges=np.array(b['per_edge_copy_bytes'],dtype=np.float64)
        count=edges/EXPERT_BYTES
        times=count*expert_ms
        bulk=np.divide(edges,bulk_bw,out=np.zeros_like(edges),where=bulk_bw>0)
        # An optimistic endpoint-aware parallel estimate, not a measured
        # all-edge concurrency result or a physical network speedup claim.
        serial=float(times.sum());parallel=float(max(times.sum(axis=0).max(),times.sum(axis=1).max()))
        bulkserial=float(bulk.sum());bulkparallel=float(max(bulk.sum(axis=0).max(),bulk.sum(axis=1).max()))
        rate=b['saved_bytes']/b['remaining_iterations'] if b['remaining_iterations'] else 0.
        costs.append(dict(request=b['request'],request_id=b['request_id'],domain=b['domain'],block_length=b['block_length'],
            block=b['block'],policy=b['policy'],budget_gib=b['budget_gib'],copy_bytes=b['copy_bytes'],avoided_dc_bytes=b['saved_bytes'],
            byte_roi=b['saved_bytes']/b['copy_bytes'] if b['copy_bytes'] else 0.,
            remaining_iterations=b['remaining_iterations'],copy_serial_expert_ms=serial,
            copy_parallel_expert_est_ms=parallel,copy_serial_bulk_est_ms=bulkserial,copy_parallel_bulk_est_ms=bulkparallel,
            byte_break_even_iterations=b['copy_bytes']/rate if rate else None,
            break_even_remaining_ratio=(b['copy_bytes']/rate/b['remaining_iterations']) if rate and b['remaining_iterations'] else None,
            actual_saved_communication_ms=None,time_roi=None,
            timing_status='UNAVAILABLE: no measured replica dispatch/combine path; byte saving cannot infer stage latency'))
    pd.DataFrame(costs).to_csv(out/'block_copy_roi.csv',index=False)
    request=pd.DataFrame(costs).groupby(['request','policy','budget_gib'],as_index=False).agg(
        copy_serial_expert_ms=('copy_serial_expert_ms','sum'),copy_parallel_expert_est_ms=('copy_parallel_expert_est_ms','sum'),
        copy_serial_bulk_est_ms=('copy_serial_bulk_est_ms','sum'),copy_parallel_bulk_est_ms=('copy_parallel_bulk_est_ms','sum'),
        remaining_iterations=('remaining_iterations','sum'),copy_bytes=('copy_bytes','sum'),avoided_dc_bytes=('avoided_dc_bytes','sum'))
    request['byte_roi']=np.divide(request.avoided_dc_bytes,request.copy_bytes,out=np.zeros(len(request)),where=request.copy_bytes>0)
    request.to_csv(out/'request_copy_roi.csv',index=False)
    new_json(out/'methodology.json',dict(byte_roi='avoided destination-deduplicated D+C bytes / all copied exact layer-expert weight bytes',
        serial_expert='sum copy count × measured two-tensor expert-sized P2P median per directed edge',
        parallel_expert='maximum source outgoing / destination incoming sum of measured single-edge times; optimistic estimate, not concurrency measurement',
        bulk='bytes / measured384MiB-edge throughput; optimistic packed-buffer alternative, not actual scattered expert-copy implementation',
        byte_break_even='hold observed avoided bytes/remaining iteration constant; extrapolation, not actual extra denoising iterations',
        timing_roi='UNAVAILABLE; no byte-to-time speedup conversion',
        static_lifetime='headline charges cold copies each block for all policies; persistent-static alternative may amortize better and is not a claimed block-local gain'))


if __name__=='__main__':main()
