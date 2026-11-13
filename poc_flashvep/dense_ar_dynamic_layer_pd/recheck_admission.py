"""Recheck observed-state decisions after removing a redundant cut gate."""
import fcntl
from .common import *
from .online_pilots import serving_config
from .run import execute
from .analyze import records,compare


def main():
    initialize();results=[]
    try:
        for mode in ('COST','DEADLINE','IDLE'):
            c=serving_config(f'actual_online_{mode}_N32_qualification_v2',cap=16,count=32,
                policy=mode,verify_layer_execution=True,
                phase='OBSERVED_STATE_N32_AFTER_REDUNDANT_PENDING_CUT_REJECTION_FIX')
            if not execute(save(c),restore=False):return
            ref='coarse_O16_N64_q4.777_native16_s1_v1'
            gate=compare(ref,c['name'],True);commits=records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')
            layers=sorted({r['offloaded_layers'] for r in commits})
            row=dict(name=c['name'],output=gate,actual_layers=layers,
                state_aware_adaptation_demonstrated=any(layers) and len(layers)>1,
                forced=False,scope='No future arrivals. A one-positive-cut outcome is reported as its actual0/m distribution, not as all-candidate adaptation.')
            results.append(row);write(ROOT/'state/admission_fix_N32_progress.json',dict(results=results))
            print('RECHECK_ADMISSION',mode,gate['passed'],layers,flush=True)
            if not gate['passed']:return
        write(ROOT/'state/post_admission_fix_N32_qualified.json',dict(results=results,passed=True))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
