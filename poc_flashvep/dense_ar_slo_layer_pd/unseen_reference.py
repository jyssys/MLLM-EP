"""Qualify unseen greedy outputs without using reference latency for tuning."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import records,compare


def main():
    jobs=[]
    for repeat in (1,2):
        jobs.append(base(f'reference_unseen_Native_N128_O200_repeat{repeat}','Native',count=128,cap=200,
            rate=4.8,dataset='unseen_confirmation.json',seed=2026111491,
            verified_ByteLevel_priming=True,clean_timing=False,correctness_only=True,
            full_prompt_state_witness=repeat==1,full_prompt_state_witness_roles=['prefill'],
            phase='UNSEEN_OUTPUT_REFERENCE_NOT_PERFORMANCE_SELECTION'))
    jobs.append(base('reference_unseen_independent_Full_DP2_N32_O200_v1','Colocated',count=32,cap=200,
        rate=2.,dataset='unseen_confirmation.json',seed=2026111491,
        chunk=8192,token_budget=8192,D_slots=16,verified_ByteLevel_priming=True,
        full_prompt_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
        clean_timing=False,correctness_only=True,phase='INDEPENDENT_UNSEEN_FULL_MODEL_REFERENCE_NOT_PERFORMANCE'))
    try:
        for cfg in jobs:
            if not execute(save(cfg),restore=False):raise RuntimeError('Unseen reference run failed')
        names=[c['name'] for c in jobs]
        repeat=compare(names[0],names[1]);independent=compare(names[0],names[2],prefix=True)
        source=json.loads((ROOT/'raw'/names[0]/'full_prompt_state_prefill.json').read_text())['requests']
        full={}
        for role in ('replica0','replica1'):
            full.update(json.loads((ROOT/'raw'/names[2]/f'full_prompt_state_{role}.json').read_text())['requests'])
        states={rid:dict(KV_equal=value['KV']==source[rid]['KV'],
            Hidden_equal=value['final_hidden_SHA256']==source[rid]['final_hidden_SHA256']) for rid,value in full.items()}
        passed=repeat['passed'] and independent['passed'] and len(states)==32 and all(all(v.values()) for v in states.values())
        write(ROOT/'analysis/unseen_reference_correctness.json',dict(reference=names[0],repeat=repeat,
            independent=independent,state_checks=states,passed=passed,evidence='CORRECTNESS_ONLY',
            scope='Reference outputs and full-state checks only. Do not aggregate these reference latencies for policy or baseline selection. Confirmation arrival traces remain separate.'))
        assert passed,'Unseen Native reference is not qualified'
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
