"""Test the installed engine's asynchronous scheduler as a strong baseline option."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .generic_qualification import independent
from .analyze import metrics


def main():
    common=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,ar_async_scheduling=True)
    cfg=base('qual_async_Colocated_B512_N32_O64_full_state_v1','Colocated',count=32,cap=64,rate=3.,
        chunk=512,token_budget=512,D_slots=16,correctness_only=True,clean_timing=False,
        full_prompt_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
        phase='STANDARD_ASYNC_SCHEDULER_OPTION_CORRECTNESS_NOT_PERFORMANCE',**common)
    try:
        name=save(cfg)
        if not execute(name,restore=False):
            write(ROOT/'analysis/asynchronous_colocated_qualification.json',dict(name=name,
                passed=False,evidence='FAILED',scope='Do not use this execution option for a performance claim. Preserve its runtime failure.'))
            return
        output=against_previous(name);expected=independent();checks={}
        for role in ('replica0','replica1'):
            states=json.loads((ROOT/'raw'/name/f'full_prompt_state_{role}.json').read_text())['requests']
            for rid,value in states.items():checks[rid]=dict(KV=value['KV']==expected[rid]['KV'],
                Hidden=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        passed=output['passed'] and len(checks)==32 and all(all(v.values()) for v in checks.values())
        write(ROOT/'analysis/asynchronous_colocated_qualification.json',dict(name=name,
            output=output,checks=checks,passed=passed,evidence='CORRECTNESS_ONLY'))
        if not passed:return
        for budget,slots,cap,rate in ((8192,1,16,3.),(512,16,16,4.),(256,16,64,2.)):
            cfg=base(f'async_C_B{budget}_S{slots}_O{cap}_q{rate}_N128_v1','Colocated',
                count=128,cap=cap,rate=rate,chunk=budget,token_budget=budget,D_slots=slots,**common)
            name=save(cfg);assert execute(name,restore=False)
            assert against_previous(name)['passed']
            write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
