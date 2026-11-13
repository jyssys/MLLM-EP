"""Compare qualified settings on viewed prompts before confirmation."""
import argparse,fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def jobs(group):
    result=[]
    if group=='colocated':
        for cap in (200,64,16):
            budgets=(256,128) if cap==200 else (8192,2048,1024,512,256,128)
            for budget in budgets:
                result.append(base(f'tune_Continuous_B{budget}_O{cap}_N64_q4p8_v1',arm='Colocated',cap=cap,
                    token_budget=budget,chunk=min(budget,8000),phase='CONTINUOUS_TOKEN_BUDGET_DISCOVERY'))
    elif group=='static':
        for cap in (16,64,200):
            for layers,fraction in ((2,1.),(4,1.),(8,1.),(12,1.),(12,.5),(8,.5)):
                result.append(base(f'tune_Static{layers}_f{round(100*fraction)}_O{cap}_N64_q4p8_v1',
                    arm='STATIC',cap=cap,static_layers=layers,migration_fraction=fraction,
                    phase='STATIC_CUT_AND_FRACTION_DISCOVERY'))
    elif group=='runtime':
        for cap in (16,64):
            for arm in ('Native','NATIVE','Revised','STATIC'):
                label='ControllerZero' if arm=='NATIVE' else arm
                result.append(base(f'runtime_D8_{label}_O{cap}_N64_q4p8_v1',arm=arm,cap=cap,
                    D_admission_batch=8,static_layers=12,migration_fraction=1.,
                    phase='COHORT_GEOMETRY_AND_CONTROLLER_ZERO_ABLATION'))
        result.append(base('runtime_D16_ControllerZero_O64_N64_q4p8_v1',arm='NATIVE',cap=64,
            phase='CONTROLLER_NO_MIGRATION_ABLATION'))
        for cap in (16,32):
            for rate in (4.,6.,8.):
                result.append(base(f'cut_discovery_Revised_O{cap}_N64_q{rate:g}_v1',arm='Revised',cap=cap,rate=rate,
                    phase='OBSERVED_STATE_SMALLER_CUT_OPPORTUNITY_DISCOVERY'))
    else:raise ValueError(group)
    for cfg in result:save(cfg)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--set',required=True,choices=['colocated','static','runtime'])
    group=parser.parse_args().set;initialize();results=[]
    try:
        for cfg in jobs(group):
            name=cfg['name']
            if not execute(name,restore=False):
                write(ROOT/'state'/(group+'_discovery_progress.json'),dict(results=results,failed=name,reason='RUNTIME_FAILURE'));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            commits=records(ROOT/'raw'/name/'actual_cut_commits.jsonl')
            work={str(n):sum(len(row['requests']) for row in commits if row['offloaded_layers']==n) for n in (0,2,4,8,12)}
            result=dict(name=name,output=output,metrics=metrics(name),completed_layer_distribution=work)
            results.append(result);write(ROOT/'state'/(group+'_discovery_progress.json'),dict(results=results))
            print('DISCOVERY',group,name,output['passed'],work,
                *[round(result['metrics'][key],2) for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens_s')],flush=True)
            if not output['passed']:return
        write(ROOT/'analysis'/(group+'_discovery_results.json'),dict(results=results,
            scope='Viewed prompt discovery. One-start point estimates are not confirmation or independent-start confidence intervals.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
