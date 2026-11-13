"""Cover every actual cut without diagnostic hash time causing cancellation."""
import argparse,fcntl
from .common import *
from .qualify import dynamic_config,state_gate
from .analyze import records,compare
from .run import execute


def main():
    initialize()
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=3);a=p.parse_args()
    c=dynamic_config(f'qual_all_candidates_mixed_N32_v{a.version}',32)
    c.update(dataset='discovery.json',negotiation_timeout_s=90,
        native_ACK_hold=True,
        phase='ALL_CANDIDATE_NUMERICAL_MIXED_QUALIFICATION_NOT_SERVING',
        scope='A long diagnostic negotiation timeout permits the independent full-reference hashes. It is not the online timeout or a performance setting.')
    try:
        name=save(c)
        if not execute(name,restore=False):return
        ref=next(p.stem for p in (ROOT/'configs').glob('coarse_O16_N64_*native16_s1_v1.json'))
        output=compare(ref,name,True);state=state_gate(name,32,True)
        commits=records(ROOT/'raw'/name/'actual_cut_commits.jsonl')
        actual={r['offloaded_layers'] for r in commits}
        passed=output['passed'] and state and actual=={0,2,4,8,12}
        write(ROOT/'state/all_candidates_mixed32_qualified.json',dict(name=name,time=time.time(),passed=passed,
            actual_layers=sorted(actual),output=output,state_pass=state,
            scope='Forced state qualification only. The earlier2s diagnostic also validated safe pre-prefix timeout fallback. Neither run demonstrates state-aware online adaptation.'))
        assert passed
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
