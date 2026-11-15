"""Test pipeline economics without forcing any online cut choice."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics,records


def settings(group=4):
    return dict(verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_manual_coalesce=True,
        nixl_descriptor_sort=False,cost_model='coalesced_cost_model_v1.json',
        P_group_size=group,D_wait_for_source_macro=True,hidden_source_slots=32,
        track_active_preparation=True,direct_objective='PIPELINE_RATE',
        direct_wait_limit_ms=2500.,objective_SLO='Decode_priority',
        safety_margin_ms=110.,safety_margin_by_layers_ms={str(k):110. for k in (2,4,8,12)},
        phase='PIPELINE_RATE_AND_DIRECT_DEADLINE_DISCOVERY_NOT_HELD_OUT')


def one(arm,group,cap,rate,layers=8):
    name=f'geometry_{arm}_G{group}_L{layers}_O{cap}_q{rate}_N128_v1'
    cfg=base(name,arm,count=128,cap=cap,rate=rate,static_layers=layers,migration_fraction=1.,**settings(group))
    assert execute(save(cfg),restore=False)
    assert against_previous(name)['passed']
    value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
    print('GEOMETRY_DISCOVERY',name,round(value['TTFT_p95_ms'],2),round(value['ITL_p99_ms'],2),
        round(value['output_tokens_s'],2),flush=True)
    return value


def main():
    limit=time.time()+5400
    while not (ROOT/'configs/coalesced_cost_model_v1.json').exists():
        assert time.time()<limit,'Calibration did not finish. Do not substitute an unqualified cost model.'
        time.sleep(10)
    try:
        # These share the actual source/destination geometry. A larger batch
        # is not attributed to the Dynamic selection rule.
        for group in (4,8,16):
            for arm in ('NATIVE','STATIC','DIRECT'):
                one(arm,group,16,4.8)
        one('STATIC',16,16,4.8,layers=12)
        for cap,rate in ((64,3.),(200,1.5)):
            for arm in ('NATIVE','STATIC','DIRECT'):one(arm,16,cap,rate)
        write(ROOT/'state/pipeline_dynamic_discovery_complete.json',dict(time=time.time(),
            scope='Actual online choices, not forced cut counts. Finite-cohort discovery only. Matched-budget confirmation remains separate.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
