"""Qualify standard compiled full-model graphs before using them as baselines."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .graph_qualification import check_state
from .analyze import metrics


def main():
    gate=ROOT/'analysis/full_graph_colocated_gate_v2.json';limit=time.time()+10800
    while not gate.exists():
        assert time.time()<limit;time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    options=dict(verified_ByteLevel_priming=True,ar_async_scheduling=True,
        gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',ar_compile_mode=3,
        explicit_cudagraph_capture_sizes=[1,2,4,8,16,256,512])
    try:
        name='qual_masked_Compile3_FULL_Colocated_N32_O64_v1'
        cfg=base(name,'Colocated',count=32,cap=64,rate=3.,chunk=512,token_budget=512,D_slots=16,
            correctness_only=True,clean_timing=False,full_prompt_state_witness=True,
            post_graph_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
            startup_timeout=900,wall_timeout=2400,**options)
        if not execute(save(cfg),restore=False):
            write(ROOT/'analysis/compiled_full_graph_colocated_gate.json',dict(name=name,passed=False,evidence='FAILED'))
            return
        token=against_previous(name);state=check_state(name,32)
        replay=sum(json.loads((ROOT/'raw'/name/f'GPU_phase_mask_{role}.json').read_text())['counters']['mixed_fullgraph_replays']
            for role in ('replica0','replica1'))
        passed=token['passed'] and state['passed'] and replay>0
        write(ROOT/'analysis/compiled_full_graph_colocated_gate.json',dict(name=name,output=token,
            state=state,mixed_fullgraph_replays=replay,passed=passed,evidence='CORRECTNESS_ONLY'))
        if not passed:return
        for budget,rate in ((512,4.),(2048,5.)):
            opts=dict(options,explicit_cudagraph_capture_sizes=[1,2,4,8,16,budget//2,budget])
            cfg=base(f'Compile3_FULL_C_B{budget}_O16_q{rate}_N128_v1','Colocated',count=128,cap=16,
                rate=rate,chunk=budget,token_budget=budget,D_slots=16,startup_timeout=900,
                wall_timeout=2400,**opts)
            name=save(cfg);assert execute(name,restore=False);assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
