"""Retry the graph qualification after correcting the installed class name."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .graph_qualification import check_state
from .analyze import metrics


def main():
    write(ROOT/'analysis/full_graph_retry_reason.json',dict(
        previous='qual_masked_FULL_graph_Colocated_N32_O64_v1',
        evidence='FAILED',cause='Import used a nonexistent CUDAGraphManager class. The installed runner uses ModelCudaGraphManager.',
        correction='Use the actual installed class. No graph or arithmetic option changed.'))
    try:
        cfg=base('qual_masked_FULL_graph_Colocated_N32_O64_v2','Colocated',count=32,cap=64,rate=3.,
            verified_ByteLevel_priming=True,ar_async_scheduling=True,chunk=512,token_budget=512,
            D_slots=16,correctness_only=True,clean_timing=False,full_prompt_state_witness=True,
            post_graph_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
            gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',
            explicit_cudagraph_capture_sizes=[1,2,4,8,16,256,512])
        name=save(cfg)
        if not execute(name,restore=False):
            write(ROOT/'analysis/full_graph_colocated_gate_v2.json',dict(name=name,passed=False,evidence='FAILED'))
            return
        token=against_previous(name);value=check_state(name,32)
        replay=sum(json.loads((ROOT/'raw'/name/f'GPU_phase_mask_{role}.json').read_text())['counters']['mixed_fullgraph_replays']
            for role in ('replica0','replica1'))
        passed=token['passed'] and value['passed'] and replay>0
        write(ROOT/'analysis/full_graph_colocated_gate_v2.json',dict(value,name=name,output=token,
            passed=passed,mixed_fullgraph_replays=replay,evidence='CORRECTNESS_ONLY'))
        if not passed:return
        for budget,rate in ((512,4.),(512,6.),(256,4.)):
            cfg=base(f'FULL_graph_C_B{budget}_O16_q{rate}_N128_v2','Colocated',count=128,cap=16,
                rate=rate,chunk=budget,token_budget=budget,D_slots=16,
                verified_ByteLevel_priming=True,gpu_phase_mask_norm=True,ar_async_scheduling=True,
                ar_cudagraph_mode='FULL',explicit_cudagraph_capture_sizes=[1,2,4,8,16,budget//2,budget])
            name=save(cfg);assert execute(name,restore=False);assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
