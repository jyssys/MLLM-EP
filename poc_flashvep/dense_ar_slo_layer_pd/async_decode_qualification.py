"""Qualify the existing asynchronous Decode worker before timing its effect."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .method_discovery_v2 import method_settings
from .stages import state_gate
from .analyze import records,metrics


def main():
    settings=method_settings()
    settings.update(async_decode=True,cost_model='coalesced_cost_model_v2.json')
    try:
        name='qual_async_D_Last024812_full_state_N32_v1'
        cfg=base(name,'FORCED_QUALIFICATION',count=32,cap=16,request_caps=[2]*32,
            arrival_offsets=[0.]*32,arrival_rate=None,forced_layers=[0,2,4,8,12,0,4,8],
            verify_layer_execution=True,hidden_verify_hash=True,verify_native_prefix_KV=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            correctness_only=True,clean_timing=False,negotiation_timeout_s=120,**settings)
        assert execute(save(cfg),restore=False);token=against_previous(name);assert token['passed']
        work=state_gate(name,32);assert work['passed'];expected=independent();folder=ROOT/'raw'/name
        source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
        dest=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
        imports={r['external_request_id']:r for p in folder.glob('extension_decode_*.jsonl')
            for r in records(p) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
        checks={}
        for rid,value in expected.items():
            native=work['cuts'][rid]==36;actual=source[rid] if native else dest[rid]
            checks[rid]=dict(KV_equal=actual['KV']==value['KV'],
                Hidden_equal=actual['final_hidden_SHA256']==value['final_hidden_SHA256'],
                Native_import_equal=imports.get(rid,{}).get('passed',False) if native else True)
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/async_D_full_state_gate.json',dict(name=name,passed=passed,
            work=work,output=token,checks=checks,evidence='CORRECTNESS_ONLY'))
        assert passed
        cfg=base('qual_async_D_online_DIRECT_N32_O16_v1','DIRECT',count=32,cap=16,rate=6.4,
            separate_nominal_objective=True,verify_layer_execution=True,correctness_only=True,
            clean_timing=False,**settings)
        name=save(cfg);assert execute(name,restore=False);assert against_previous(name)['passed']
        work=state_gate(name,32);assert work['passed']
        write(ROOT/'analysis/async_D_online_work_gate.json',dict(name=name,work=work,passed=True,evidence='CORRECTNESS_ONLY'))
        for arm,layers,rate in (('STATIC',12,4.8),('NATIVE',0,3.2)):
            name=f'async_D_{arm}_L{layers}_N128_O16_q{rate}_v1'
            cfg=base(name,arm,count=128,cap=16,rate=rate,static_layers=layers,migration_fraction=1.,**settings)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('ASYNC_D',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
