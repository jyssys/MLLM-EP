"""Check SLO-specific graph frontiers and a separate raw-throughput arm."""
import fcntl
from .common import *
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def graph_options(budget):
    gate='full_graph_colocated_gate_v2.json' if budget==512 else f'full_graph_B{budget}_state_gate.json'
    assert json.loads((ROOT/'analysis'/gate).read_text())['passed']
    return dict(verified_ByteLevel_priming=True,ar_async_scheduling=True,
        gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',chunk=budget,
        token_budget=budget,D_slots=16,
        explicit_cudagraph_capture_sizes=[1,2,4,8,16,budget//2,budget])


def one(label,rate,options,cap=16):
    name=f'objective_{label}_O{cap}_q{rate}_s2026111456_W600_v1'
    cfg=duration_config(name,'Colocated',rate,cap=cap,seed=2026111456,
        warm=60.,measure=600.,follow=90.,phase='SLO_SPECIFIC_OBJECTIVE_DISCOVERY',**options)
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    result=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    print('OBJECTIVE_RATE',name,result['raw_completed_output_tokens_s'],
        {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in result['SLO_regimes'].items()},flush=True)
    return result


def main():
    try:
        runs=[]
        runs.append(one('BALANCED_FULL_B640',5.6,graph_options(640))['name'])
        runs.append(one('DECODE_FULL_B512',5.4,graph_options(512))['name'])
        # This is a raw objective, not a Decode-safe Colocated configuration.
        runs.append(one('RAW_B8192_S32',6.4,dict(verified_ByteLevel_priming=True,
            direct_phase_norm=True,ar_async_scheduling=True,chunk=8192,
            token_budget=8192,D_slots=32))['name'])
        # Full36 Decode remains identical. Only the output cap changes.
        for cap,rate in ((64,3.),(200,1.5)):
            runs.append(one('LONG_DECODE_FULL_B512',rate,graph_options(512),cap=cap)['name'])
        write(ROOT/'analysis/objective_rate_discovery.json',dict(runs=runs,time=time.time(),
            scope='Discovery only. Raw and SLO-specific objectives are distinct. These sparse finite-window tests do not establish a global maximum.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
