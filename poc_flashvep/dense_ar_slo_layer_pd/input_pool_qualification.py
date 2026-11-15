"""Qualify immutable Host prompt sharing without reducing GPU prompt work."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent


def main():
    data=json.loads((ROOT/'data/discovery.json').read_text())[:8]
    path=ROOT/'data/repeated_eight_prompt_pool_qualification32.json';rows=[data[i%8] for i in range(32)]
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
    name='qual_intern_immutable_pool_FULL_graph_N32_O16_v1'
    cfg=base(name,'Colocated',count=32,cap=16,rate=3.,dataset=path.name,
        intern_immutable_prompt_pool=True,verified_ByteLevel_priming=True,ar_async_scheduling=True,
        gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',chunk=512,token_budget=512,D_slots=16,
        explicit_cudagraph_capture_sizes=[1,2,4,8,16,256,512],correctness_only=True,clean_timing=False,
        full_prompt_state_witness=True,post_graph_state_witness=True,
        full_prompt_state_witness_roles=['replica0','replica1'])
    try:
        assert execute(save(cfg),restore=False);output=against_previous(name);assert output['passed']
        expected=independent();checks={};tokens={f'q{i}':0 for i in range(32)}
        for role in ('replica0','replica1'):
            values=json.loads((ROOT/'raw'/name/f'full_prompt_state_{role}.json').read_text())['requests']
            for rid,value in values.items():
                original=expected[f'q{int(rid[1:])%8}']
                checks[rid]=dict(KV_equal=value['KV']==original['KV'],
                    Hidden_equal=value['final_hidden_SHA256']==original['final_hidden_SHA256'])
            for p in (ROOT/'raw'/name).glob('native_'+role+'_*.json'):
                for span in json.loads(p.read_text())['prompt_spans']:
                    rid=span['request_id'].split('-')[0]
                    if rid in tokens:tokens[rid]+=span['count']
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values()) and all(v==8000 for v in tokens.values())
        write(ROOT/'analysis/immutable_prompt_pool_GPU_gate.json',dict(name=name,passed=passed,
            output=output,checks=checks,actual_scheduled_prompt_rows=tokens,
            distinct_source_documents=8,request_count=32,evidence='CORRECTNESS_ONLY',
            scope='Only immutable Host token-list storage is shared. Every unique request still schedules all8000 prompt rows. Prefix caching stays disabled. Same generated tokens, all36 valid KV planes and final Hidden. This is not measured GPU speedup or proof that prior delivery spikes came from GC.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
