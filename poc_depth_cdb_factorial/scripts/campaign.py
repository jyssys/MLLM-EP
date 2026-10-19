"""Run every required cell. No gain-based early termination exists."""
import argparse
import subprocess
from poc_depth_cdb_factorial.contract import *

def launch(plan,output,gpus,extra=()):
    destination=ROOT/'outputs'/output
    if destination.exists():
        assert (destination/'status.json').exists() and read(destination/'status.json')['success'],destination
        print('ALREADY_COMPLETE',output,flush=True)
        return
    command=[REFERENCE_PYTHON,'-m','poc_depth_cdb_factorial.scripts.gpu_session','--gpus',gpus,
        '--output','outputs/'+output,'--timeout','18000','--',REFERENCE_PYTHON,
        '-m','poc_depth_cdb_factorial.scripts.run','--plan',str(ROOT/'artifacts'/plan),*extra]
    print('RUN',command,flush=True)
    subprocess.run(command,cwd=PROJECT,check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['speed','extra','quality','trace','sensitivity','all'],default='all')
    args=p.parse_args()
    if args.phase in ['speed','all']:
        for repeat in range(3):
            order=['A0','DP4'] if repeat%2==0 else ['DP4','A0']
            for group in order:
                launch(f'speed_{group}_r{repeat}_plan.json',f'speed_{group}_r{repeat}_v1','4' if group=='A0' else MASK)
    if args.phase in ['quality','all']:
        launch('quality_A0_plan.json','quality_A0_v1','4')
        if not (ROOT/'artifacts/quality_DP4_plan.json').exists():
            subprocess.run([REFERENCE_PYTHON,'-m','poc_depth_cdb_factorial.scripts.prepare_plans',
                '--quality-followup',str(ROOT/'outputs/quality_A0_v1')],cwd=PROJECT,check=True)
        launch('quality_DP4_plan.json','quality_DP4_v1',MASK)
    if args.phase in ['trace','all']:
        launch('trace_A0_full_plan.json','trace_A0_v1','4',('--trace',))
        launch('trace_DP4_full_plan.json','trace_DP4_v1',MASK,('--trace',))
    if args.phase in ['sensitivity','all']:
        launch('sensitivity_v3_matched_DP4_plan.json','sensitivity_DP4_v1',MASK)
    if args.phase in ['extra','all'] and (ROOT/'artifacts/repeat_extension.json').exists():
        for repeat in [3,4]:
            order=['A0','DP4'] if repeat%2==0 else ['DP4','A0']
            for group in order:
                launch(f'speed_{group}_r{repeat}_plan.json',f'speed_{group}_r{repeat}_v1','4' if group=='A0' else MASK)
        if (ROOT/'artifacts/sensitivity_shuffled_DP4_plan.json').exists():
            launch('sensitivity_shuffled_DP4_plan.json','order_control_DP4_v1',MASK)

if __name__=='__main__':main()
