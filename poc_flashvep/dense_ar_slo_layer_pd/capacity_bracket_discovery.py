"""Bracket qualified fixed-window rates before any held-out selection."""
import fcntl
from .common import *
from .sustainable import duration_config,assess
from .method_discovery_v2 import method_settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def one(arm,rate):
    options=method_settings()
    options.update(async_decode=True,separate_nominal_objective=True,
        cost_model='coalesced_cost_model_v2.json',static_layers=12,migration_fraction=1.,
        phase='CAPACITY_BRACKET_DISCOVERY_NOT_HELD_OUT')
    name=f'bracket_{arm}_ASYNC_D_O16_q{rate}_s2026111455_W600_v1'
    cfg=duration_config(name,arm,rate,seed=2026111455,warm=60.,measure=600.,follow=90.,**options)
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    result=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    print('CAPACITY_BRACKET',name,result['raw_completed_output_tokens_s'],
        {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in result['SLO_regimes'].items()},flush=True)
    return result


def main():
    assert json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
    results=[]
    try:
        for arm in ('STATIC','DIRECT'):
            first=one(arm,4.8);results.append(first['name'])
            second=one(arm,5.2 if first['SLO_regimes']['Decode_priority']['qualified'] else 4.4)
            results.append(second['name'])
        write(ROOT/'analysis/capacity_bracket_discovery.json',dict(runs=results,time=time.time(),
            scope='One predeclared branch per method. These are discovery operational screens, not maximum sustainable capacity or held-out confirmation.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
