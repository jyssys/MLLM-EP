"""Tune larger qualified graph budgets for Balanced and raw objectives."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .graph_qualification import check_state
from .analyze import metrics


def main():
    assert json.loads((ROOT/'analysis/full_graph_colocated_gate_v2.json').read_text())['passed']
    try:
        for budget in (640,768,1024):
            options=dict(verified_ByteLevel_priming=True,ar_async_scheduling=True,
                gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',chunk=budget,
                token_budget=budget,D_slots=16,
                explicit_cudagraph_capture_sizes=[1,2,4,8,16,budget//2,budget])
            name=f'qual_FULL_graph_B{budget}_Colocated_N32_O64_v1'
            cfg=base(name,'Colocated',count=32,cap=64,rate=3.,correctness_only=True,
                clean_timing=False,full_prompt_state_witness=True,post_graph_state_witness=True,
                full_prompt_state_witness_roles=['replica0','replica1'],**options)
            if not execute(save(cfg),restore=False):
                write(ROOT/'analysis'/f'full_graph_B{budget}_state_gate.json',dict(name=name,passed=False,evidence='FAILED'))
                continue
            output=against_previous(name);state=check_state(name,32);passed=output['passed'] and state['passed']
            write(ROOT/'analysis'/f'full_graph_B{budget}_state_gate.json',dict(name=name,output=output,
                state=state,passed=passed,evidence='CORRECTNESS_ONLY'))
            if not passed:continue
            name=f'frontier_FULL_graph_B{budget}_O16_q5.0_N128_v1'
            cfg=base(name,'Colocated',count=128,cap=16,rate=5.,**options)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('GRAPH_OBJECTIVE_TUNING',budget,value['TTFT_p95_ms'],value['ITL_p99_ms'],
                {k:v['Joint_attainment'] for k,v in value['SLO_regimes'].items()},flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
