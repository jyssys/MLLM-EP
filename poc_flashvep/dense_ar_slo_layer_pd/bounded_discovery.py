"""Separate source grouping from a Decode-safe destination batch cap."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .qualification import against_previous
from .analyze import metrics
from .dynamic_discovery import settings


def main():
    limit=time.time()+7200
    gate=ROOT/'analysis/source16_destination8_full_state_gate.json'
    while not (ROOT/'configs/coalesced_cost_model_v1.json').exists() or not gate.exists():
        assert time.time()<limit,'Do not proceed without shape and cost qualification'
        time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    try:
        for cap,rate in ((16,4.8),(64,3.),(200,1.5)):
            for arm,layers in (('NATIVE',0),('STATIC',8),('STATIC',12),('DIRECT',0)):
                options=settings(16);options.update(D_admission_batch=8,bounded_destination_batches=True,
                    static_request_block_size=16,phase='BOUNDED_DESTINATION_PIPELINE_DISCOVERY')
                name=f'bounded8_{arm}_L{layers}_G16_O{cap}_q{rate}_N128_v1'
                cfg=base(name,arm,count=128,cap=cap,rate=rate,static_layers=layers,
                    migration_fraction=1.,**options)
                assert execute(save(cfg),restore=False)
                assert against_previous(name)['passed']
                value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value)
                print('BOUNDED_DESTINATION',name,round(value['TTFT_p95_ms'],2),
                    round(value['ITL_p99_ms'],2),round(value['output_tokens_s'],2),flush=True)
        write(ROOT/'state/bounded_destination_discovery_complete.json',dict(time=time.time(),
            scope='The destination cap and group geometry also apply to Controller Zero and Strong Static. They are not isolated Dynamic-policy gains.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
