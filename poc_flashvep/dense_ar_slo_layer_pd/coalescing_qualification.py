"""Test physical-range merging without replacing native KV ownership."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import records
from .generic_qualification import independent


def main():
    cfg=base('qual_manual_coalescing_Native_N32_full_state_v2','Native',count=32,cap=16,rate=4.8,
        verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_manual_coalesce=True,
        nixl_descriptor_sort=False,full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill'],
        imported_native_state_witness=True,correctness_only=True,clean_timing=False,
        phase='PHYSICAL_BYTE_RANGE_COALESCING_CORRECTNESS_NOT_CLEAN_TIMING')
    try:
        if not execute(save(cfg),restore=False):raise RuntimeError('Physical-range coalescing qualification failed')
        name=cfg['name'];folder=ROOT/'raw'/name;output=against_previous(name)
        expected=independent();source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
        dest=json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests'];checks={}
        for rid,value in source.items():checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
            Imported_KV_equal=dest[rid]['KV']==value['KV'],
            Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        tx=json.loads((folder/'manual_KV_coalescing.json').read_text())['rows']
        actual_reduction=bool(tx) and any(v['submitted_descriptors']<v['input_descriptors'] for v in tx)
        passed=output['passed'] and len(checks)==32 and all(all(v.values()) for v in checks.values())
        result=dict(name=name,output=output,checks=checks,passed=passed,
            actual_reduction=actual_reduction,transactions=len(tx),
            evidence='CORRECTNESS_ONLY',scope='Exact full-model state and original ACK. Transaction geometry does not establish a Serving improvement.')
        write(ROOT/'analysis/manual_KV_coalescing_full_state_gate.json',result)
        assert passed,'Coalesced ranges changed state or output'
        print('MANUAL_COALESCING_STATE',passed,'REDUCED',actual_reduction,flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
