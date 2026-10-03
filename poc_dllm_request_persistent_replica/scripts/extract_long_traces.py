"""Reuse full physical accounting validator; save only live IDs for new planners."""
import json
import numpy as np
import pandas as pd
from poc_dllm_request_persistent_replica.contract import *
from poc_dllm_ep_comm.scripts.analyze import request
from poc_dllm_block_local_replica.scripts.reuse_or_extract_traces import extract


def main():
    dest=ROOT/'data/live_routes'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    frozen=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    selections={r['request_id']:r for r in frozen['rows']}
    summaries=[];manifest=[];iterrows=[]
    for directory in sorted((ROOT/'raw/long_v1').glob('*_b32')):
        baseline,iterations,_,timing=request(directory)
        arrays,blocks,audit=extract(directory,baseline)
        # Exact source raw contains routing weights+positions+receive IDs;
        # planner duplicate stores only IDs needed for residency accounting.
        keep={k:v for k,v in arrays.items() if k.endswith('_ids')}
        archive=dest/(directory.name+'.npz');np.savez_compressed(archive,**keep)
        metadata=selections[baseline['request_id']]
        manifest.append(dict(request=directory.name,request_id=baseline['request_id'],domain=baseline['domain'],
            split=metadata['split'],lane='long',block_length=32,blocks=blocks,archive=str(archive),
            B_one_shot=baseline['B_one_shot'],B_live_only=baseline['B_live_only'],audit=audit,
            output_tokens=baseline['output_tokens'],finish=baseline['finish'],archive_sha256=sha(archive)))
        summaries.append(baseline|dict(split=metadata['split'],blocks=len(blocks)))
        iterrows.extend(iterations)
        print('VALIDATED_LONG',directory.name,baseline['output_tokens'],len(blocks),flush=True)
    assert len(manifest)==len(frozen['rows'])
    new_json(ROOT/'results/long_manifest.json',dict(requests=manifest,physical_source='raw/long_v1',
        all_source_partition_receive_buffer_checks_pass=True,live_scope_unchanged=True,
        whole_output_and_acceptance_timing_trace_parity=True))
    pd.DataFrame(summaries).to_csv(ROOT/'results/long_requests.csv',index=False)
    pd.DataFrame(iterrows).to_csv(ROOT/'results/long_iterations.csv',index=False)


if __name__=='__main__':main()
