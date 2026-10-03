"""Policy-independent byte-ROI upper bound; not a latency bound.

One replicated expert can remove at most one destination D+C row (8192B)
per activation, even under fictitious perfect closure. Count every activation
including profiling iterations to favor replication. With uniform6MiB expert
size, any plan's avoided-byte/copy-byte ratio is bounded by the maximum
single(layer,source,expert) activation count times8192 /6291456.
"""
import json
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *
from poc_dllm_block_local_replica.scripts.simulate_replication import load_blocks,frequency


def main():
    manifest=json.loads((ROOT/'results/reused_trace_manifest.json').read_text())['requests']
    rows=[]
    for info in manifest:
        for meta,data in load_blocks(info):
            counts=frequency(data,meta['iterations'],0,meta['iterations'])
            for s in range(4):counts[:,s,s*64:(s+1)*64]=0
            layer,src,e=np.unravel_index(counts.argmax(),counts.shape);count=int(counts[layer,src,e])
            rows.append(dict(request=info['request'],block=meta['block'],iterations=meta['iterations'],
                most_recurrent_layer=layer+1,source=src,expert=e,max_remote_path_activations=count,
                any_policy_byte_roi_upper_bound=count*ROW_DC_BYTES/EXPERT_BYTES,
                necessary_activation_count_for_roi2=int(2*EXPERT_BYTES//ROW_DC_BYTES)))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/copy_roi_upper_bound.csv',index=False)
    new_json(ROOT/'results/copy_roi_upper_bound.json',dict(
        maximum_any_policy_byte_roi_bound=float(df.any_policy_byte_roi_upper_bound.max()),
        proof='Each removed destination-row can be charged to >=1 replicated selected expert. Sum per-expert counts×8192 upper-bounds all avoided bytes. Divide by count×6MiB. Includes profiling activations, so optimistic.',
        validity='only cold per-block copies, this B1 source-SP4 model/cohort, full exact BF16 routed expert tensors',
        not_a_timing_bound=True,not_a_cross_block_or_large_batch_bound=True))


if __name__=='__main__':main()
