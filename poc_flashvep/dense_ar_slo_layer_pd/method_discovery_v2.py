"""Compare strong compatible-ready batching before attributing policy gains."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records
from .stages import state_gate
from .dynamic_discovery import settings


def method_settings(group=8,wait=False):
    c=settings(group);c.update(D_wait_for_source_macro=wait,D_combine_ready_macros=True,
        D_admission_batch=8,bounded_destination_batches=True,frontier_completion_model=True,
        static_request_block_size=group,phase='STRONG_READY_BATCHING_FRONTIER_POLICY_DISCOVERY')
    return c


def one(arm,group=8,wait=False,layers=12,cap=16,rate=4.8,count=128,qualify=False):
    name=f'method2_{arm}_G{group}_D8_wait{int(wait)}_L{layers}_O{cap}_q{rate}_N{count}_v1'
    cfg=base(name,arm,count=count,cap=cap,rate=rate,static_layers=layers,migration_fraction=1.,
        verify_layer_execution=qualify,correctness_only=qualify,clean_timing=not qualify,
        **method_settings(group,wait))
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    if qualify:
        work=state_gate(name,count);assert work['passed']
        requests=records(ROOT/'raw'/name/'requests.jsonl')
        migrated=sum(r['completed_D_prefill_layers'] for r in requests)
        assert migrated>0,'The actual policy did not exercise migration. Do not label it online-qualified.'
        write(ROOT/'analysis/frontier_policy_online_qualification.json',dict(name=name,
            passed=True,work=work,completed_migrated_layers=migrated,evidence='CORRECTNESS_ONLY',
            scope='Actual unforced cut selection, exact tokens, actual work and normal lifetime. State arithmetic uses separately full-state-qualified executors.'))
    value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
    print('METHOD2',name,round(value['TTFT_p95_ms'],2),round(value['ITL_p99_ms'],2),round(value['output_tokens_s'],2),flush=True)
    return value


def main():
    gate=ROOT/'analysis/combined_ready_macros_full_state_gate.json';limit=time.time()+7200
    while not gate.exists():
        assert time.time()<limit,'Wait for correct ready-work batching before comparison'
        time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    try:
        one('DIRECT',rate=6.4,count=32,qualify=True)
        for group,wait in ((4,False),(8,False),(16,False),(8,True)):
            one('STATIC',group,wait)
        for arm,layers in (('STATIC',8),('STATIC',4),('DIRECT',0),('NATIVE',0)):
            one(arm,layers=layers)
        one('DIRECT',wait=True,layers=0)
        for cap,rate in ((64,3.),(200,1.5)):
            for arm,layers in (('STATIC',2),('STATIC',8),('DIRECT',0)):
                one(arm,wait=True,layers=layers,cap=cap,rate=rate)
        write(ROOT/'state/method2_discovery_complete.json',dict(time=time.time(),
            scope='Discovery only. Source-ready batching is common to all method controls. No independent Dynamic gain is assumed.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
