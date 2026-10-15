"""Bounded sequential native runs. Each run restores unused-device burner policy."""
import subprocess
import sys
from poc_looped_recurrence_tp_4models.contract import *


def run(model,family,R,TP,smoke=False):
    name=f'{model}_{family}_R{R}_TP{TP}'+('_smoke_v1' if smoke else '_v1')
    directory=ROOT/'results/raw'/name
    if directory.exists():
        status=directory/'status.json'
        return status.exists() and json.loads(status.read_text()).get('success',False)
    command=[sys.executable,'-m','poc_looped_recurrence_tp_4models.scripts.gpu_session',
        '--gpus',','.join(map(str,GPUS[:TP])),'--timeout','2400','--output','results/raw/'+name,'--',PYTHON,
        '-m','poc_looped_recurrence_tp_4models.scripts.run_config','--model',model,'--family',family,
        '--r',str(R),'--tp',str(TP),'--output',str(directory)]+(['--smoke'] if smoke else [])
    result=subprocess.run(command,cwd=PROJECT,check=False)
    print(name,result.returncode,flush=True)
    return result.returncode==0 and (directory/'status.json').exists()


def main():
    smoke=run('ouro','native',4,1,True)
    tied=smoke and run('ouro','tied',4,1,True)
    write(ROOT/'results/control_feasibility.json',dict(native_smoke=smoke,tied_smoke=tied,
        tolerance=PARITY,scope='Eager smoke. Compiled same-TP/R parity remains mandatory for every tied arm.'))
    # Both real AR checkpoints execute early. Native/tied pairs share adjacent sessions.
    jobs=[('qwen3','native',1,1),('qwen25','native',1,4),
          ('ouro','native',4,1),('ouro','tied',4,1),('ouro','native',4,4),('ouro','tied',4,4),
          ('qwen3','native',1,4),('qwen25','native',1,1),('ouro','native',4,2),('ouro','tied',4,2),
          ('qwen3','native',1,2),('qwen25','native',1,2)]
    for R,tps in [(1,[2,1,4]),(2,[4,2,1])]:
        for TP in tps:jobs.extend([('ouro','native',R,TP),('ouro','tied',R,TP)])
    records=[]
    for model,family,R,TP in jobs:
        if family=='tied' and not tied:
            records.append(dict(model=model,family=family,R=R,TP=TP,status='TIED_UNROLL_RUNTIME_BLOCKED'))
        else:
            success=run(model,family,R,TP)
            records.append(dict(model=model,family=family,R=R,TP=TP,status='COMPLETE' if success else 'FAILED_PRESERVED'))
        write(ROOT/'results/campaign_progress.json',records)


if __name__=='__main__':main()
