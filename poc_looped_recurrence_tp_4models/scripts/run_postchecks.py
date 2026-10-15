"""Queue bounded diagnostics after the primary campaign releases its GPUs."""
import subprocess
import sys
import time
import argparse
from poc_looped_recurrence_tp_4models.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v2');args=parser.parse_args()
    deadline=time.monotonic()+7200
    while True:
        progress=ROOT/'results/campaign_progress.json'
        try:ready=progress.exists() and len(json.loads(progress.read_text()))==24
        except json.JSONDecodeError:ready=False
        if ready:break
        if time.monotonic()>deadline:raise TimeoutError('Main campaign did not finish within the diagnostic wait budget')
        time.sleep(20)
    jobs=[('ouro',family,4,TP,TP==1) for TP in [1,4,2] for family in ['native','tied']]+[('qwen3','native',1,1,False),('qwen25','native',1,1,False)]
    jobs += [('ouro',family,2,4,True) for family in ['native','tied']]
    records=[]
    for model,family,R,TP,focus in jobs:
        name=f'{model}_{family}_R{R}_TP{TP}_{args.tag}';directory=ROOT/'results/diagnostics'/name
        if directory.exists():
            records.append(dict(name=name,status='PRESERVED_EXISTING'));continue
        command=[sys.executable,'-m','poc_looped_recurrence_tp_4models.scripts.gpu_session',
            '--gpus',','.join(map(str,GPUS[:TP])),'--timeout','1800','--output','results/diagnostics/'+name,
            '--',PYTHON,'-m','poc_looped_recurrence_tp_4models.scripts.run_control_diagnostic',
            '--model',model,'--family',family,'--r',str(R),'--tp',str(TP),'--output',str(directory)]
        if focus:command.append('--repeat-source-grid')
        result=subprocess.run(command,cwd=PROJECT,check=False)
        records.append(dict(name=name,exit_code=result.returncode))
        write(ROOT/f'results/diagnostic_progress_{args.tag}.json',records)
        print(name,result.returncode,flush=True)
        if result.returncode:
            print('Preserve first failure and stop diagnostics. No automatic retry.',flush=True);break


if __name__=='__main__':main()
