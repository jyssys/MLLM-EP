"""Resume-safe CPU closeout. This program never launches a GPU experiment."""
import json
import os
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write


def main():
    state=ROOT/'state/postprocess.json'
    while True:
        p=ROOT/'state/residual_followup.json'
        if p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE':break
        write(state,dict(status='WAITING_FOR_RESIDUAL_PROBES',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    commands=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        report=folder/'timeline.nsys-rep'
        if not report.exists():continue
        if not (folder/'timeline.sqlite').exists():
            commands.append(['nsys','export','--type','sqlite','--output',str(folder/'timeline.sqlite'),str(report)])
        commands.append([PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.nsys_analysis','--run',folder.name])
    for module in ('summarize','slowdown','capture_analysis','finalize_analysis','artifacts','validation'):
        commands.append([PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.'+module])
    commands.append([PYTHON,'-m','pytest','-q','poc_flashvep/looped_pd_b_numerical_diagnosis/test_diagnostics.py'])
    results=[]
    for index,command in enumerate(commands):
        write(state,dict(status='RUNNING',command=command,index=index,total=len(commands),pid=os.getpid(),time=time.time()))
        log=ROOT/'logs'/f'postprocess_{index}.log'
        with log.open('a') as stream:
            value=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        results.append(dict(command=command,returncode=value.returncode,log=str(log.relative_to(ROOT))))
        if value.returncode:
            write(state,dict(status='FAILED',results=results,time=time.time()));return
    write(state,dict(status='COMPLETE',results=results,time=time.time()))


if __name__=='__main__':main()
