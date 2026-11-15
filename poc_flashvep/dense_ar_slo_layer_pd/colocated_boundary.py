"""Refine the observed Colocated capacity boundary with both legal routers."""
import fcntl
from .common import *
from .objective_rate_discovery import graph_options
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    try:
        names=[]
        for placement in ('ROUND_ROBIN','LEAST_OUTSTANDING'):
            name=f'boundary_C_FULL_B512_S16_{placement}_O16_q5.2_s2026111459_W600_v1'
            cfg=duration_config(name,'Colocated',5.2,seed=2026111459,warm=60.,measure=600.,follow=90.,
                placement=placement,phase='SLO_SPECIFIC_BOUNDARY_ROUTING_DISCOVERY',**graph_options(512))
            assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
            value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name));names.append(name)
            print('COLOCATED_BOUNDARY',name,value['raw_completed_output_tokens_s'],
                {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in value['SLO_regimes'].items()},flush=True)
        write(ROOT/'state/colocated_boundary_complete.json',dict(time=time.time(),names=names))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
