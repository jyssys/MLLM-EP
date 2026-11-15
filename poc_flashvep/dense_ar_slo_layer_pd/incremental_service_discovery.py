"""Ablate transient waiting and selected-group queue double counting."""
import fcntl
from .common import *
from .configuration import base
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records
from .stages import state_gate


def main():
    assert json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
    options=method_settings()
    options.update(async_decode=True,separate_nominal_objective=True,cost_model='coalesced_cost_model_v2.json',
        pipeline_objective_excludes_existing_wait=True,phase='INCREMENTAL_SERVICE_OBJECTIVE_ABLATION_DISCOVERY')
    try:
        for external,count,qualify in ((False,32,True),(False,128,False),(True,128,False)):
            name=f'incremental_DIRECT_external{int(external)}_ASYNC_D_N{count}_O16_q4.8_v1'
            cfg=base(name,'DIRECT',count=count,cap=16,rate=4.8,
                queue_benefit_external_only=external,correctness_only=qualify,
                clean_timing=not qualify,verify_layer_execution=qualify,**options)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            if qualify:
                work=state_gate(name,count);assert work['passed']
                migrated=sum(r.get('completed_D_prefill_layers',0) for r in records(ROOT/'raw'/name/'requests.jsonl'))
                assert migrated>0,'The unforced policy must exercise actual migration before qualification'
                write(ROOT/'analysis/incremental_service_policy_work_gate.json',dict(name=name,
                    passed=True,work=work,completed_migrated_layers=migrated,evidence='CORRECTNESS_ONLY'))
            value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
            print('INCREMENTAL_SERVICE',name,value['TTFT_p95_ms'],value['ITL_p99_ms'],value['output_tokens_s'],flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
