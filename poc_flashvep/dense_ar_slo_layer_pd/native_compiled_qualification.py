"""Apply the restored full-model compile path to Native, not just Colocated."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .analyze import metrics


def main():
    assert json.loads((ROOT/'analysis/compiled_opaque_norm_gate.json').read_text())['passed']
    options=dict(verified_ByteLevel_priming=True,gpu_phase_mask_norm=True,opaque_unfused_norm=True,
        ar_compile_mode=3,ar_execution_family='COMPILE3_OPAQUE_ORIGINAL_UNFUSED_NORM',
        nixl_manual_coalesce=True,nixl_descriptor_sort=False,startup_timeout=900,wall_timeout=2400)
    try:
        name='qual_Native_Compile3_opaque_norm_N32_O16_v1'
        cfg=base(name,'Native',count=32,cap=16,rate=4.8,full_prompt_state_witness=True,
            post_graph_state_witness=True,full_prompt_state_witness_roles=['prefill'],
            correctness_only=True,clean_timing=False,**options)
        if not execute(save(cfg),restore=False):
            write(ROOT/'analysis/native_compiled_full_state_gate.json',dict(name=name,passed=False,evidence='FAILED'))
            return
        token=against_previous(name);states=json.loads((ROOT/'raw'/name/'full_prompt_state_prefill.json').read_text())['requests']
        expected=independent();checks={rid:dict(KV_equal=value['KV']==expected[rid]['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256']) for rid,value in states.items()}
        passed=token['passed'] and len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/native_compiled_full_state_gate.json',dict(name=name,passed=passed,
            output=token,checks=checks,evidence='CORRECTNESS_ONLY',
            scope='Original full36 Native model with original exact normalization calls. Qualified transport is unchanged. This is generic baseline tuning, not a Layer migration gain.'))
        if not passed:return
        for rate in (3.2,4.8):
            name=f'Native_Compile3_opaque_norm_N128_O16_q{rate}_v1'
            cfg=base(name,'Native',count=128,cap=16,rate=rate,**options)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('NATIVE_COMPILED',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
