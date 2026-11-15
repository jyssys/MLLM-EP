"""Qualify common runtime improvements and actual large destination cohorts."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import records,metrics
from .stages import state_gate


def independent():
    reference=LEGACY/'raw/qual_D16_independent_full_state_N32_v2'
    return {r['external_request_id']:r for p in reference.glob('extension_prefill_*.jsonl')
        for r in records(p) if r['event']=='VALID_QUERY_STATE' and r.get('query_tokens')==8000
        and r['external_request_id'].startswith('q')}


def run_one(cfg):
    if not execute(save(cfg),restore=False):raise RuntimeError('Generic runtime qualification failed: '+cfg['name'])
    gate=against_previous(cfg['name']);assert gate['passed'],gate


def main():
    initialize();expected=independent();assert len(expected)==32
    common=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_descriptor_sort=True,
        clean_timing=False,correctness_only=True,full_prompt_state_witness=True,
        phase='GENERIC_RUNTIME_FULL_STATE_CORRECTNESS_NOT_TIMING')
    jobs=[base('qual_direct_norm_Colocated_B256_N32_O64_v1','Colocated',count=32,cap=64,rate=3.,
        chunk=256,token_budget=256,D_slots=16,full_prompt_state_witness_roles=['replica0','replica1'],**common),
        base('qual_sorted_Native_N32_full_state_v1','Native',count=32,cap=16,rate=4.8,
            full_prompt_state_witness_roles=['prefill'],imported_native_state_witness=True,**common),
        base('qual_source_macro16_Last8_Native_N32_v1','FORCED_QUALIFICATION',count=32,cap=16,
            arrival_offsets=[0.]*32,arrival_rate=None,P_group_size=16,D_wait_for_source_macro=True,
            hidden_source_slots=32,D_admission_batch=16,forced_layers=[8,0],
            verify_layer_execution=True,hidden_verify_hash=True,verify_native_prefix_KV=True,
            negotiation_timeout_s=120,full_prompt_state_witness_roles=['prefill','decode'],**common)]
    results=[]
    try:
        for index,cfg in enumerate(jobs):
            run_one(cfg);folder=ROOT/'raw'/cfg['name'];checks={}
            if index==0:
                for role in ('replica0','replica1'):
                    states=json.loads((folder/f'full_prompt_state_{role}.json').read_text())['requests']
                    for rid,v in states.items():checks[rid]=dict(KV_equal=v['KV']==expected[rid]['KV'],
                        Hidden_equal=v['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
                mixed=sum(json.loads((folder/f'phase_stable_norm_{r}.json').read_text())['counters']['prepared_mixed_measurement_batches'] for r in ('replica0','replica1'))
                assert mixed>0,'The short-Prefill operation did not run in a measured mixed batch'
            elif index==1:
                states=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
                imports=json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests']
                for rid,v in states.items():checks[rid]=dict(KV_equal=v['KV']==expected[rid]['KV'],
                    Imported_KV_equal=imports[rid]['KV']==v['KV'],
                    Hidden_equal=v['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
            else:
                work=state_gate(cfg['name'],32);assert work['passed'],work
                source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
                destination=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
                native_imports={r['external_request_id']:r for p in folder.glob('extension_decode_*.jsonl')
                    for r in records(p) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
                for rid in expected:
                    v=source[rid] if work['cuts'][rid]==36 else destination[rid]
                    checks[rid]=dict(KV_equal=v['KV']==expected[rid]['KV'],
                        Hidden_equal=v['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'],
                        Native_import_equal=native_imports.get(rid,{}).get('passed',False) if work['cuts'][rid]==36 else True)
            passed=len(checks)==32 and all(all(v.values()) for v in checks.values())
            value=dict(name=cfg['name'],checks=checks,passed=passed,evidence='CORRECTNESS_ONLY')
            results.append(value);write(ROOT/'state/generic_qualification_progress.json',dict(results=results))
            assert passed,cfg['name']
        write(ROOT/'analysis/generic_runtime_full_state_gate.json',dict(results=results,passed=True,
            scope='Exact tokens, valid KV and final Hidden. The macro test also checks actual layer visits and normal lifetimes. No Serving performance or physical idle claim.'))
        print('GENERIC_RUNTIME_FULL_STATE_PASS',len(results),flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
