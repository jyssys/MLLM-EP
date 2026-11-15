"""Qualify serial destination batches from one real source macro."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .stages import state_gate
from .analyze import records


def main():
    cfg=base('qual_source16_destination8_Last8_Native_N32_v1','FORCED_QUALIFICATION',
        count=32,cap=16,request_caps=[2]*32,arrival_offsets=[0.]*32,arrival_rate=None,
        P_group_size=16,D_wait_for_source_macro=True,D_admission_batch=8,
        hidden_source_slots=32,forced_layers=[8,0],negotiation_timeout_s=120,
        verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_manual_coalesce=True,
        nixl_descriptor_sort=False,verify_layer_execution=True,hidden_verify_hash=True,
        verify_native_prefix_KV=True,full_prompt_state_witness=True,
        full_prompt_state_witness_roles=['prefill','decode'],correctness_only=True,clean_timing=False,
        phase='SOURCE_MACRO_WITH_TWO_SERIAL_DESTINATION_COHORTS_CORRECTNESS')
    try:
        name=save(cfg);assert execute(name,restore=False)
        token=against_previous(name);work=state_gate(name,32)
        assert token['passed'] and work['passed']
        folder=ROOT/'raw'/name;expected=independent()
        source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
        destination=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
        checks={}
        for rid in expected:
            value=source[rid] if work['cuts'][rid]==36 else destination[rid]
            checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
                Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        batches={r['cohort']:r['cohort_size'] for r in records(folder/'D_host_admission.jsonl')}
        passed=len(checks)==32 and all(all(v.values()) for v in checks.values()) and list(batches.values())==[8]*4
        write(ROOT/'analysis/source16_destination8_full_state_gate.json',dict(name=name,
            checks=checks,actual_destination_cohorts=batches,passed=passed,evidence='CORRECTNESS_ONLY',
            scope='All actual prefix/suffix layers, Full36 Decode, final Hidden, valid KV, exact tokens, ACK and normal retirement. No performance claim.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
