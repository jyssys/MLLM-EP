"""Separate economic ranking from unchanged upper-bound safety predicates."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records
from .stages import state_gate


def one(count,cap,rate,qualify=False):
    name=f'method3_DIRECT_nominal_O{cap}_q{rate}_N{count}_v1'
    settings=method_settings()
    settings.update(separate_nominal_objective=True,cost_model='coalesced_cost_model_v2.json',
        phase='NOMINAL_OBJECTIVE_UNCHANGED_SAFETY_DISCOVERY')
    cfg=base(name,'DIRECT',count=count,cap=cap,rate=rate,correctness_only=qualify,
        clean_timing=not qualify,verify_layer_execution=qualify,**settings)
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    if qualify:
        work=state_gate(name,count);assert work['passed']
        migrated=sum(r['completed_D_prefill_layers'] for r in records(ROOT/'raw'/name/'requests.jsonl'))
        assert migrated>0
        write(ROOT/'analysis/nominal_policy_online_gate.json',dict(name=name,passed=True,
            work=work,completed_migrated_layers=migrated,evidence='CORRECTNESS_ONLY'))
    value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
    print('METHOD3',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)


def main():
    try:
        one(32,16,6.4,True)
        one(128,16,4.8)
        one(128,64,3.)
        one(128,200,1.5)
        write(ROOT/'state/method3_complete.json',dict(time=time.time(),evidence='MEASURED',
            scope='Discovery only. Nominal economic ranking does not relax the safety upper-bound or any SLO.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
