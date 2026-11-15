"""Run long fixed-window baseline screens before selecting a capacity peak."""
import fcntl
from .common import *
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    common=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,
        nixl_manual_coalesce=True,nixl_descriptor_sort=False,
        phase='TEN_MINUTE_FIXED_WINDOW_DISCOVERY_NOT_HELD_OUT_CONFIRMATION')
    jobs=[duration_config('sustain_discovery_Native_O16_q2p8_s51_v1','Native',2.8,
        seed=2026111451,measure=600.,follow=90.,**common),
        duration_config('sustain_discovery_C_B8192_S1_O16_q2p5_s51_v1','Colocated',2.5,
            seed=2026111451,measure=600.,follow=90.,chunk=8192,token_budget=8192,D_slots=1,**common),
        duration_config('sustain_discovery_C_B512_S16_O16_q2p5_s51_v1','Colocated',2.5,
            seed=2026111451,measure=600.,follow=90.,chunk=512,token_budget=512,D_slots=16,**common)]
    try:
        for cfg in jobs:
            name=save(cfg);assert execute(name,restore=False)
            assert against_previous(name)['passed']
            result=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
            print('SUSTAINED_DISCOVERY',name,{k:(v['qualified'],v['arrival_cohort_joint_attainment'])
                for k,v in result['SLO_regimes'].items()},flush=True)
        write(ROOT/'state/sustained_baseline_discovery_complete.json',dict(time=time.time(),
            scope='A finite ten-minute stability screen. Independent peak confirmation and an upper neighboring rate remain required.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
