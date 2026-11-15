"""Measure streaming strong Static and Dynamic with a fixed long window."""
import argparse,fcntl
from .common import *
from .method_discovery_v2 import method_settings
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['STATIC','DIRECT','NATIVE'])
    p.add_argument('rate',type=float);p.add_argument('--layers',type=int,default=12)
    p.add_argument('--seed',type=int,default=2026111453);p.add_argument('--measure',type=float,default=600.)
    p.add_argument('--async-decode',action='store_true')
    p.add_argument('--tag',default='v1');a=p.parse_args()
    options=method_settings()
    options.update(cost_model='coalesced_cost_model_v2.json',static_layers=a.layers,
        migration_fraction=1.,separate_nominal_objective=True,
        phase='STRONG_READY_BATCHING_TEN_MINUTE_DISCOVERY')
    if a.async_decode:
        assert json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
        options['async_decode']=True
    name=f'rate_method3_{a.arm}_L{a.layers}_O16_q{a.rate}_s{a.seed}_W{a.measure}_{a.tag}'
    if a.async_decode:name+='_ASYNC_D'
    cfg=duration_config(name,a.arm,a.rate,seed=a.seed,measure=a.measure,follow=90.,**options)
    try:
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        result=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        print('METHOD_RATE',name,result['raw_completed_output_tokens_s'],
            {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in result['SLO_regimes'].items()},flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
