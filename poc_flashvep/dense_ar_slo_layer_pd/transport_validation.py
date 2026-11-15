"""Qualify byte-range transport and then test its serving cost."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import records,metrics
from .stages import state_gate
from .generic_qualification import independent


def main():
    common=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,
        nixl_manual_coalesce=True,nixl_descriptor_sort=False)
    cfg=base('qual_coalesced_all_cuts_N32_full_state_v1','FORCED_QUALIFICATION',count=32,
        cap=16,request_caps=[2]*32,arrival_offsets=[0.]*32,arrival_rate=None,
        forced_layers=[0,2,4,8,12,0,4,8],verify_layer_execution=True,
        hidden_verify_hash=True,full_prompt_state_witness=True,
        full_prompt_state_witness_roles=['prefill','decode'],verify_native_prefix_KV=True,
        imported_native_state_witness=True,correctness_only=True,clean_timing=False,
        negotiation_timeout_s=120,phase='COALESCED_PARTIAL_KV_FULL_STATE_CORRECTNESS',**common)
    try:
        name=save(cfg)
        assert execute(name,restore=False)
        token=against_previous(name);work=state_gate(name,32)
        assert token['passed'] and work['passed']
        folder=ROOT/'raw'/name;expected=independent()
        source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
        dest=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
        checks={}
        for rid in expected:
            value=source[rid] if work['cuts'][rid]==36 else dest[rid]
            checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
                Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        requests=records(folder/'requests.jsonl')
        native_cache=all(r['D_cached_prompt_tokens']==8000 for r in requests if r['actual_layer_cut']==36)
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values()) and native_cache
        write(ROOT/'analysis/coalesced_all_cuts_full_state_gate.json',dict(name=name,
            checks=checks,native_cache_complete=native_cache,passed=passed,evidence='CORRECTNESS_ONLY'))
        assert passed
        # A/B/A keeps workload, output, and exact generic corrections fixed.
        for i,enabled in enumerate((False,True,False)):
            test=base(f'transport_ABA_Native_N128_q4p8_O16_{i}_v2','Native',count=128,rate=4.8,
                verified_ByteLevel_priming=True,direct_phase_norm=True,
                nixl_manual_coalesce=enabled,nixl_descriptor_sort=False,
                phase='TRANSPORT_ABA_DISCOVERY_NOT_SUSTAINABLE_CAPACITY')
            name=save(test);assert execute(name,restore=False)
            assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        for cap in (16,64):
            test=base(f'qual_continuous_coalesced_Native_N64_O{cap}_v1','Native',count=64,cap=cap,rate=3.,
                native_continuous_D_ingress=True,D_admission_batch=1,**common)
            name=save(test);assert execute(name,restore=False)
            assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        write(ROOT/'state/transport_validation_complete.json',dict(time=time.time(),passed=True))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
