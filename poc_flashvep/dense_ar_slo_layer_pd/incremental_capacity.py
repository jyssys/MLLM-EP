"""Check whether the improved incremental objective survives a long run."""
import fcntl
from .common import *
from .method_discovery_v2 import method_settings
from .residual_guard_study import calibrate
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics
from .prediction_validation import analyze


def main():
    assert json.loads((ROOT/'analysis/incremental_service_policy_work_gate.json').read_text())['passed']
    margin=calibrate()['total_margin_ms'];options=method_settings()
    options.update(async_decode=True,separate_nominal_objective=True,cost_model='coalesced_cost_model_v2.json',
        pipeline_objective_excludes_existing_wait=True,queue_benefit_external_only=True,
        safety_margin_ms=margin,safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)},
        phase='INCREMENTAL_RESIDUAL_BOUND_LONG_DISCOVERY')
    rates=[4.8];runs=[]
    try:
        for index in range(2):
            rate=rates[index];name=f'incremental_guard_{int(margin)}_DIRECT_O16_q{rate}_s2026111458_W600_v1'
            cfg=duration_config(name,'DIRECT',rate,seed=2026111458,warm=60.,measure=600.,follow=90.,**options)
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
            write(ROOT/'analysis'/(name+'_prediction.json'),analyze(name));runs.append(name)
            if index==0:rates.append(5.2 if value['SLO_regimes']['Decode_priority']['qualified'] else 4.4)
            print('INCREMENTAL_CAPACITY',name,value['raw_completed_output_tokens_s'],
                {k:v['qualified'] for k,v in value['SLO_regimes'].items()},flush=True)
        write(ROOT/'analysis/incremental_capacity_complete.json',dict(runs=runs,time=time.time()))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
