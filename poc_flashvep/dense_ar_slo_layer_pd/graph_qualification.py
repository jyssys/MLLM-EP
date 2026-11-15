"""Qualify a stronger existing full-graph Colocated execution path."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .analyze import metrics


def check_state(name,count):
    expected=independent();checks={}
    for role in ('replica0','replica1'):
        states=json.loads((ROOT/'raw'/name/f'full_prompt_state_{role}.json').read_text())['requests']
        for rid,value in states.items():checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
    return dict(checks=checks,passed=len(checks)==count and all(all(v.values()) for v in checks.values()))


def main():
    limit=time.time()+7200
    gate=ROOT/'analysis/phase_mask_norm_GPU_gate.json'
    while not gate.exists():
        assert time.time()<limit;time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    common=dict(verified_ByteLevel_priming=True,ar_async_scheduling=True,
        chunk=512,token_budget=512,D_slots=16,correctness_only=True,clean_timing=False,
        full_prompt_state_witness=True,post_graph_state_witness=True,
        full_prompt_state_witness_roles=['replica0','replica1'])
    try:
        cfg=base('qual_post_runner_witness_Eager_N8_v1','Colocated',count=8,cap=16,rate=3.,
            direct_phase_norm=True,**common)
        name=save(cfg);assert execute(name,restore=False)
        assert against_previous(name)['passed'];value=check_state(name,8);assert value['passed']
        write(ROOT/'analysis/post_runner_witness_crosscheck.json',dict(name=name,**value,evidence='CORRECTNESS_ONLY'))
        cfg=base('qual_masked_FULL_graph_Colocated_N32_O64_v1','Colocated',count=32,cap=64,rate=3.,
            gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',
            explicit_cudagraph_capture_sizes=[1,2,4,8,16,256,512],**common)
        name=save(cfg)
        if not execute(name,restore=False):
            write(ROOT/'analysis/full_graph_colocated_gate.json',dict(name=name,passed=False,evidence='FAILED'))
            return
        token=against_previous(name);value=check_state(name,32)
        replay=sum(json.loads((ROOT/'raw'/name/f'GPU_phase_mask_{role}.json').read_text())['counters']['mixed_fullgraph_replays']
            for role in ('replica0','replica1'))
        passed=token['passed'] and value['passed'] and replay>0
        result=dict(value,name=name,output=token,passed=passed,
            mixed_fullgraph_replays=replay,evidence='CORRECTNESS_ONLY')
        write(ROOT/'analysis/full_graph_colocated_gate.json',result)
        if not passed:return
        for budget,rate in ((512,4.),(512,6.),(256,4.)):
            cfg=base(f'FULL_graph_C_B{budget}_O16_q{rate}_N128_v1','Colocated',count=128,cap=16,
                rate=rate,chunk=budget,token_budget=budget,D_slots=16,
                verified_ByteLevel_priming=True,gpu_phase_mask_norm=True,
                ar_async_scheduling=True,ar_cudagraph_mode='FULL',
                explicit_cudagraph_capture_sizes=[1,2,4,8,16,budget//2,budget])
            name=save(cfg);assert execute(name,restore=False);assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
