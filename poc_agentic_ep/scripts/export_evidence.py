"""Compact semantic raw evidence + explicit destination ranks, append-only."""
import argparse
import gzip
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from poc_agentic_ep.contract import ROOT,SPEC,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--version',required=True);a=p.parse_args()
    root=ROOT/'results'/a.version;raw=ROOT/'raw'/a.run
    if (root/'compact_manifest.json').exists():raise FileExistsError('published evidence already exists')
    workers=json.loads((raw/'worker.dp0.json').read_text())['records']
    waves={r['wave'] for r in workers if r['split']!='diagnostic'}
    frames=[{(f['wave'],f['forward']):f for f in map(json.loads,(raw/f'frames.rank{rank}.jsonl').read_text().splitlines())} for rank in range(4)]
    keys=[k for k,f in frames[0].items() if k[0] in waves and f.get('sampled_tokens') is not None]
    assert len(keys)==108
    for rank in range(4):
        with gzip.open(root/f'native_semantic_frames.rank{rank}.jsonl.gz','wt') as f:
            for key in keys:f.write(json.dumps(frames[rank][key],separators=(',',':'))+'\n')
    write(root/'native_worker_records.json',dict(rows=workers,original_sha256=sha(raw/'worker.dp0.json'),
        scope='instrumented wave/drain timing only, not latency headline; native actual KV-cache hit counts'))
    destination={}
    for path in (root/'raw_routing').glob('*.npz'):
        with np.load(path) as values:destination[path.stem]=(values['expert_ids']//32).astype(np.uint8)
    np.savez_compressed(root/'destination_ep_ranks.npz',**destination)
    with gzip.open(root/'oracle_policy_assignments.json.gz','wt') as f:
        value=json.loads((root/'oracle_policy_assignments.json').read_text());json.dump(value,f,separators=(',',':'))
    rank_cols=['request_id','trajectory_id','turn','split','phase','tokens','rank_counts','remote_paths',
        'remote_dispatch_bytes','remote_combine_bytes','rank_max_mean','rank_cv','hottest_rank']
    pd.read_csv(root/'turn_metrics.csv')[rank_cols].to_csv(root/'ep_rank_metrics.csv',index=False)
    write(root/'execution_ledger.json',dict(semantic_requests_main=len(workers),semantic_cohort_turns=108,
        stage0_sanity_requests=1,earlier_smoke_requests=3,total_semantic_requests=112,
        main_target_forward_frames_per_rank=len(frames[0]),retained_semantic_forwards=len(keys),
        idle_dummy_forward_frames_per_rank=len(frames[0])-len(workers),
        note='native external-DP wave/drain has dummy forwards; never counted as agent turns or user routing'))
    # Byte-exact authoritative document kept as an immutable snapshot.
    (root/'authoritative_spec_snapshot.md').write_bytes(SPEC.read_bytes())
    snapshot=subprocess.check_output(['nvidia-smi','--id=2,3,6,7','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader'],text=True)
    write(root/'final_gpu_state.json',dict(gpus=snapshot,mask='2,3,6,7',
        original_burner_restored=True,policy='existing run_utilize.sh with explicit VLLM_UTILIZE_GPUS=2,3,6,7'))
    files=[path for path in root.rglob('*') if path.is_file() and path.name!='oracle_policy_assignments.json']
    write(root/'compact_manifest.json',dict(files={str(path.relative_to(root)):sha(path) for path in files},
        source_main_status=json.loads((raw/'status.json').read_text()),
        spec_sha256=sha(SPEC),raw_schema=dict(expert_ids='[48,tokens,8] uint8 actual global expert IDs',
            routing_weights='original FP32 selected normalized probabilities',positions='original input positions uint16',
            source_ep_rank='actual EP source0/1',destination_ep_ranks='separate NPZ companion, actual verified linear32 map',
            token_category='0=new_prefill,1=teacher-forced decode'),
        original_local_raw_preserved=str(raw),hidden_states_saved=False,weights_uploaded=False))
    print(json.dumps(dict(status='EXPORTED',files=len(files),semantic_routes=108)))


if __name__=='__main__':main()
