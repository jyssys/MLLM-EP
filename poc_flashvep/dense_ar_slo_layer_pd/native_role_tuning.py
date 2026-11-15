"""Retain a fast qualified Decode path while testing compiled Native Prefill."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .analyze import metrics


def main():
    assert json.loads((ROOT/'analysis/native_compiled_full_state_gate.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/async_D_full_state_gate.json').read_text())['passed']
    options=dict(verified_ByteLevel_priming=True,gpu_phase_mask_norm=True,opaque_unfused_norm=True,
        direct_phase_norm=True,phase_mask_norm_roles=['prefill'],ar_compile_mode=0,
        ar_compile_mode_by_role={'prefill':3,'decode':0},async_decode=True,
        ar_execution_family='NATIVE_P3_D0_ASYNC_EXACT_NORM',nixl_manual_coalesce=True,
        nixl_descriptor_sort=False,startup_timeout=900,wall_timeout=2400)
    try:
        name='qual_Native_P3_D0_async_N32_O16_v1'
        cfg=base(name,'Native',count=32,cap=16,rate=4.8,full_prompt_state_witness=True,
            post_graph_state_witness=True,full_prompt_state_witness_roles=['prefill'],
            correctness_only=True,clean_timing=False,**options)
        if not execute(save(cfg),restore=False):
            write(ROOT/'analysis/native_role_tuned_state_gate.json',dict(name=name,passed=False,evidence='FAILED'))
            return
        token=against_previous(name);states=json.loads((ROOT/'raw'/name/'full_prompt_state_prefill.json').read_text())['requests']
        expected=independent();checks={rid:dict(KV_equal=value['KV']==expected[rid]['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256']) for rid,value in states.items()}
        passed=token['passed'] and len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/native_role_tuned_state_gate.json',dict(name=name,passed=passed,
            output=token,checks=checks,evidence='CORRECTNESS_ONLY'))
        if not passed:return
        name='Native_P3_D0_async_N128_O16_q4.8_v1'
        cfg=base(name,'Native',count=128,cap=16,rate=4.8,**options)
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
        print('NATIVE_ROLE_TUNING',value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
