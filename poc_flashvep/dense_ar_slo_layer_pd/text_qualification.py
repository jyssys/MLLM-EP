"""Qualify generic exact text preparation before retuning the baseline."""
from .common import *
from .configuration import base
from .frontier import run
from .analyze import records


def main():
    assert json.loads((ROOT/'analysis/verified_ByteLevel_text_CPU_gate.json').read_text())['passed']
    cfg=base('qual_Colocated_fast_text_N32_O200_full_state_v1','Colocated',count=32,cap=200,rate=3.,
        chunk=8192,token_budget=8192,D_slots=1,verified_ByteLevel_priming=True,verify_primed_text=True,
        full_prompt_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
        clean_timing=False,correctness_only=True,phase='EXACT_STREAMED_TEXT_AND_FULL_STATE_NOT_TIMING')
    run([cfg]);folder=ROOT/'raw'/cfg['name'];checks={};parity=[]
    reference=LEGACY/'raw/qual_D16_independent_full_state_N32_v2'
    expected={r['external_request_id']:r for p in reference.glob('extension_prefill_*.jsonl')
        for r in records(p) if r['event']=='VALID_QUERY_STATE' and r.get('query_tokens')==8000
        and r['external_request_id'].startswith('q')}
    for role in ('replica0','replica1'):
        states=json.loads((folder/f'full_prompt_state_{role}.json').read_text())['requests']
        for rid,r in states.items():checks[rid]=dict(KV_equal=r['KV']==expected[rid]['KV'],
            Hidden_equal=r['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        for p in folder.glob('extension_'+role+'_*.jsonl'):
            parity.extend(r for r in records(p) if r['event']=='VERIFIED_BYTELEVEL_TEXT_PARITY'
                and r.get('external_request_id','').startswith('q'))
    passed=len(checks)==32 and all(all(v.values()) for v in checks.values()) and len(parity)==6400
    write(ROOT/'analysis/fast_text_GPU_full_state_gate.json',dict(name=cfg['name'],checks=checks,
        streamed_text_comparisons=len(parity),passed=passed,evidence='CORRECTNESS_ONLY'))
    assert passed,'Exact text or full-state gate failed'
    jobs=[]
    for budget,slots in ((8192,1),(512,16),(384,16),(256,16)):
        jobs.append(base(f'textv2_Colocated_B{budget}_S{slots}_N64_O16_q3_v1','Colocated',
            chunk=budget,token_budget=budget,D_slots=slots,verified_ByteLevel_priming=True,
            phase='SLO_FRONTIER_AFTER_GENERIC_EXACT_TEXT_PREPARATION'))
    run(jobs)


if __name__=='__main__':main()
