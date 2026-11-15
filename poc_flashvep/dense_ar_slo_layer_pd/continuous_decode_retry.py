"""Retest cached-query ingress after qualifying standard async Decode."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .analyze import metrics


def main():
    assert json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
    options=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,async_decode=True,
        nixl_manual_coalesce=True,nixl_descriptor_sort=False,native_continuous_D_ingress=True,
        D_admission_batch=1,phase='GENERIC_ASYNC_CACHED_QUERY_INGRESS_DIAGNOSTIC_NOT_DYNAMIC_GAIN')
    try:
        name='qual_async_continuous_Native_N32_full_state_v1'
        cfg=base(name,'Native',count=32,cap=16,rate=4.8,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill'],
            imported_native_state_witness=True,correctness_only=True,clean_timing=False,**options)
        assert execute(save(cfg),restore=False);output=against_previous(name);assert output['passed']
        folder=ROOT/'raw'/name;expected=independent()
        source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
        imported=json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests']
        checks={rid:dict(KV_equal=value['KV']==expected[rid]['KV'],
            Imported_KV_equal=imported[rid]['KV']==value['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
            for rid,value in source.items()}
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/async_continuous_Native_full_state_gate.json',dict(name=name,
            passed=passed,checks=checks,output=output,evidence='CORRECTNESS_ONLY',
            scope='Native cached queries only. No recurrent or layer suffix overlaps Decode. This is a generic baseline test, not a Dynamic layer result.'))
        assert passed
        for cap,rate in ((16,4.8),(64,3.),(200,1.5)):
            name=f'async_continuous_Native_N128_O{cap}_q{rate}_v1'
            cfg=base(name,'Native',count=128,cap=cap,rate=rate,**options)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('ASYNC_CONTINUOUS_NATIVE',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],
                value['output_tokens_s'],{k:v['Joint_attainment'] for k,v in value['SLO_regimes'].items()},flush=True)
        write(ROOT/'state/async_continuous_Native_diagnostic_complete.json',dict(time=time.time()))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
