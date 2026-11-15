"""Test positive own-request benefit without fabricating Source queue pressure."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records
from .stages import state_gate


def main():
    replay=json.loads((ROOT/'analysis/zero_queue_personal_benefit_replay.json').read_text())
    assert replay['accepted_requests']>0
    settings=method_settings()
    settings.update(separate_nominal_objective=True,allow_zero_queue_personal_benefit=True,
        cost_model='coalesced_cost_model_v2.json',phase='POSITIVE_PERSONAL_BENEFIT_ZERO_QUEUE_DISCOVERY')
    try:
        for count,rate,qualify in ((32,6.4,True),(128,4.8,False)):
            name=f'method4_DIRECT_personal_O16_q{rate}_N{count}_v1'
            cfg=base(name,'DIRECT',count=count,cap=16,rate=rate,correctness_only=qualify,
                clean_timing=not qualify,verify_layer_execution=qualify,**settings)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            if qualify:
                work=state_gate(name,count);assert work['passed']
                migrated=sum(r['completed_D_prefill_layers'] for r in records(ROOT/'raw'/name/'requests.jsonl'))
                assert migrated>0
                write(ROOT/'analysis/personal_benefit_policy_online_gate.json',dict(name=name,
                    work=work,completed_migrated_layers=migrated,passed=True,evidence='CORRECTNESS_ONLY'))
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('METHOD4',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
