"""Extract only existing physical live K8 paths; validate against old baseline."""
import json
import collections
from pathlib import Path
import numpy as np
import pandas as pd
from poc_dllm_block_local_replica.contract import *
from poc_dllm_block_local_replica.metrics import closure_groups
from poc_dllm_ep_comm.metrics import traffic


def extract(directory, baseline):
    traces=[json.loads((directory/f'trace.rank{r}.json').read_text()) for r in range(4)]
    blocks=collections.defaultdict(lambda:collections.defaultdict(dict))
    hashes={};total=0;validations=0
    for src,t in enumerate(traces):
        path=directory/f'trace.rank{src}.npz';hashes[path.name]=sha(path)
        hashes[f'trace.rank{src}.json']=sha(directory/f'trace.rank{src}.json')
        calls={e['call_id']:e for e in t['events']}
        with np.load(path) as z:
            for r in t['routes']:
                event=calls[r['call_id']]
                if event['phase']!='denoise':continue
                tag=r['tag'];idx=event['iteration']['index'];block=event['iteration']['block']
                live=(z[tag+'_positions']>=t['prompt_length']) & z[tag+'_before']
                ids=z[tag+'_ids'][live].astype(np.int16)
                positions=z[tag+'_positions'][live].astype(np.int32)
                weights=z[tag+'_weights'][live].astype(np.float32)
                assert ids.shape==(len(positions),8)
                # Preserve the recorded physical source partition. It may
                # change at cross-block calls; never derive src from token ID.
                blocks[int(block)][int(idx)][(int(r['layer']),src)]=(ids,positions,weights)
                total+=int(traffic(ids,src)['total'].sum())
                validations+=1
    assert total==int(baseline['B_live_only']), (directory.name,total,baseline['B_live_only'])
    rows={};metadata=[]
    for block,iterations in sorted(blocks.items()):
        order=sorted(iterations)
        for ordinal,idx in enumerate(order):
            for (layer,src),(ids,positions,weights) in sorted(iterations[idx].items()):
                tag=f'b{block}_t{ordinal}_l{layer}_s{src}'
                rows[tag+'_ids']=ids;rows[tag+'_positions']=positions;rows[tag+'_weights']=weights
        metadata.append(dict(block=block,iteration_indices=order,iterations=len(order)))
    return rows,metadata,dict(request=directory.name,route_files=hashes,live_bytes=total,
                              validations=validations,one_shot=int(baseline['B_one_shot']))


def main():
    dest=ROOT/'data/live_routes'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    baseline=pd.read_csv(SOURCE/'results/main_v2/requests.csv').set_index('request')
    manifest=[]
    for request,row in baseline.iterrows():
        arrays,blocks,audit=extract(SOURCE/'raw/main_v2'/request,row)
        np.savez_compressed(dest/(request+'.npz'),**arrays)
        manifest.append(dict(request=request,request_id=row['request_id'],domain=row['domain'],
                             block_length=int(row['block_length']),blocks=blocks,
                             B_live_only=int(row['B_live_only']),B_one_shot=int(row['B_one_shot']),
                             original_Raw_Amp=float(row['Raw_Amp']),audit=audit))
        print('EXTRACTED',request,len(blocks),flush=True)
    new_json(ROOT/'results/reused_trace_manifest.json',dict(source_commit=SOURCE_COMMIT,
        source='poc_dllm_ep_comm/raw/main_v2',requests=manifest,
        selected_rows='position>=prompt_length AND masked/live before actual physical forward',
        source_rank='recorded physical source-SP partition, not invented token mapping',
        raw_reuse_complete=True,additional_full_model_trace_requests=0))


if __name__=='__main__':main()
