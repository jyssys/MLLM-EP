"""Qualify variable-length state before running a heterogeneous policy."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import records,compare
from .stages import state_gate


def configs():
    common=dict(count=32,cap=16,dataset='variable_qualification.json',length=12288,
        chunk=12288,token_budget=32000,variable_input=True,warmup_batch_sizes=[1,2],
        arrival_offsets=[0.]*32,arrival_rate=None,verified_ByteLevel_priming=True,
        full_prompt_state_witness=True,clean_timing=False,correctness_only=True,
        phase='VARIABLE_INPUT_EXACT_STATE_NOT_SERVING_PERFORMANCE')
    return [base('qual_variable_Full_reference_DP2_N32_v1','Colocated',D_slots=4,
                full_prompt_state_witness_roles=['replica0','replica1'],**common),
        base('qual_variable_Forced_Last024812_N32_v1','FORCED_QUALIFICATION',P_group_size=1,
            forced_layers=[0,2,4,8,12,0,4,8],verify_layer_execution=True,
            hidden_verify_hash=True,verify_native_prefix_KV=True,negotiation_timeout_s=120,
            full_prompt_state_witness_roles=['prefill','decode'],**common)]


def main():
    assert json.loads((ROOT/'data/variable_manifest.json').read_text())
    try:
        jobs=configs()
        for cfg in jobs:
            if not execute(save(cfg),restore=False):raise RuntimeError('Variable-input runtime qualification failed')
        reference,candidate=[c['name'] for c in jobs];output=compare(reference,candidate)
        expected={}
        for role in ('replica0','replica1'):
            expected.update(json.loads((ROOT/'raw'/reference/f'full_prompt_state_{role}.json').read_text())['requests'])
        source=json.loads((ROOT/'raw'/candidate/'full_prompt_state_prefill.json').read_text())['requests']
        dest=json.loads((ROOT/'raw'/candidate/'full_prompt_state_decode.json').read_text())['requests']
        work=state_gate(candidate,32);checks={}
        Native_imports={r['external_request_id']:r for p in (ROOT/'raw'/candidate).glob('extension_decode_*.jsonl')
            for r in records(p) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
        for rid,value in expected.items():
            actual=source[rid] if work['cuts'][rid]==36 else dest[rid]
            checks[rid]=dict(queries=value['queries'],KV_equal=actual['KV']==value['KV'],
                Hidden_equal=actual['final_hidden_SHA256']==value['final_hidden_SHA256'],
                Native_import_equal=Native_imports.get(rid,{}).get('passed',False) if work['cuts'][rid]==36 else True)
        passed=output['passed'] and work['passed'] and len(checks)==32 and all(
            all(v for k,v in r.items() if k!='queries') for r in checks.values())
        write(ROOT/'analysis/variable_input_full_state_gate.json',dict(output=output,work=work,checks=checks,
            passed=passed,evidence='CORRECTNESS_ONLY',scope='Full valid state at each actual prompt length. Independent full-model reference. Native final Hidden stays on P. Suffix final Hidden is computed on D. This does not calibrate variable-length costs.'))
        assert passed,'Variable-input exact state gate failed'
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
