"""Repeat the fixed last12 work/geometry contrast with new arrival seeds."""
import fcntl
import random
import time
from .common import ROOT,QUIET_LOCK,resources,write,digest
from .axis_configuration import base
from .discovery import job
from .token_axis import matched_token_plan


def main():
    seeds=(2026111760,2026111761,2026111762)
    contract=dict(count=144,cap=16,QPS=6.4,seeds=list(seeds),layer_cut='Last12',
        token_plan=matched_token_plan(12,144),source_group=4,destination_group=4,
        SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        scope='Selected fixed-work contrast. Fresh arrival seeds, but the original64-document discovery pool is cycled. This is not held-out-prompt generalization or best-versus-best token tuning. Native uses the normal strong16-slot admission, not the forced4-request geometry.')
    path=ROOT/'state/matched_confirmation_freeze.json'
    if not path.exists():
        write(path,dict(time=time.time(),contract=contract))
    values=[]
    try:
        for seed in seeds:
            methods=['Native','Static','Token'];random.Random(seed).shuffle(methods)
            for method in methods:
                cfg=base(f'matched_confirm_{method}_Last12_N144_q6.4_s{seed}_v1',
                    method=method,count=144,cap=16,rate=6.4,seed=seed,layers=12,
                    matched=method!='Native',hidden_source_slots=32,hidden_destination_slots=32,
                    phase='INDEPENDENT_ARRIVAL_MATCHED_WORK_GEOMETRY')
                if method=='Native':
                    cfg.update(D_admission_batch=16,D_wait_for_source_macro=False,D_combine_ready_macros=True)
                elif method=='Token':
                    cfg['token_suffix_by_request']=matched_token_plan(12,144)
                value=job(cfg)
                if value:
                    values.append(value);write(ROOT/'analysis/matched_arrival_confirmation_results.json',values)
        write(ROOT/'state/matched_arrival_confirmation_complete.json',dict(names=[r['name'] for r in values]))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
