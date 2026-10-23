"""Record actual pins, upstream line ranges, devices, and prior evidence."""
import json
import subprocess
import time
from pathlib import Path
from .common import ROOT,FORK,REPO,MODEL,UPSTREAM,UUIDS,write,digest

def main():
    names=('vllm_rlt/config.py','vllm_rlt/request.py','vllm_rlt/core/scheduling_policy.py',
        'vllm_rlt/core/scheduler.py','vllm_rlt/worker/executor.py','vllm_rlt/worker/model_runner.py',
        'vllm_rlt/worker/cuda_graph.py','vllm_rlt/worker/async_state.py','vllm_rlt/worker/submission_queue.py',
        'vllm_rlt/core/kv_cache_manager.py','vllm_rlt/engine/stage_engine.py')
    diff=subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    patch=REPO/'poc_flashvep/ouro_elb/patches/grouped_decode.patch'
    patch.parent.mkdir(parents=True,exist_ok=True);patch.write_bytes(diff)
    write(ROOT/'source_lock.json',dict(captured_wall_s=time.time(),model_path=str(MODEL),model_revision=MODEL.name,
        runtime_path=str(FORK),runtime_pin=UPSTREAM,runtime_patch_sha256=digest(patch),
        files=[dict(path=n,sha256=digest(FORK/n)) for n in names],
        spec_sha256=digest(REPO/'poc_flashvep/reports/ouro_8gpu_elastic_loop_batching_decode_fusion_poc_spec.md'),
        parent_commit='b3c064d7fe65753b4e5a0f20c517083d27eac00c'))
    stats=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,memory.total','--format=csv,noheader'],text=True)
    write(ROOT/'gpu_uuid_map.json',dict(authorized=UUIDS,observed=stats,scope='8 independent single-GPU contexts'))
    for name in names:
        assert (FORK/name).exists()
    write(ROOT/'prior_evidence_ledger.json',dict(rows=[
        dict(commit='b3c064d7fe65753b4e5a0f20c517083d27eac00c',finding='Dual objective: NO_GO_INCREMENTAL. Actual C changes did not beat G-only. Batched fidelity unresolved.',action='Do not repeat prefill optimization.'),
        dict(commit='821b11653695d0a470bd88bfcc997ae1bbcf5bc9',finding='144/144 same Joint/G-only plans. No incremental optimizer claim.',action='Require changed executed decode plans and measured completed-token benefit.'),
        dict(commit='d6ba31c7275019867e2aca473818d42d9b1fcbb0',finding='Deadline predictor blocked. Issued depth differs from GPU completion.',action='Use actual producer events and delivered tokens. No deadline predictor required.'),
        dict(commit='c458f6a2a764f667a0da6784bcb2592b36626837',finding='Prefill group patch only. Native decode still one pass.',action='New isolated decode-only patch. Preserve previous worktree.')]))

if __name__=='__main__':main()
