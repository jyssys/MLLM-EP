"""Analyze completed phases on CPU without taking a GPU lease."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,execution_allowed

def run():
    tasks=[('c0_campaign_complete_v2.json','c0_phase_analysis_complete_v2.json',[
        ['-m','poc_flashvep.dense_ar_output_aware_v2.c0_analysis'],
        ['-m','poc_flashvep.dense_ar_output_aware_v2.c0_selector_diagnostic','evaluate'],
        ['-m','poc_flashvep.dense_ar_output_aware_v2.path_analysis','--plan','c0_output_matrix_v2.json']]),
        ('c3_continuous_iso8_output_bridge_complete_v2.json','c3_iso8_phase_analysis_complete_v2.json',[
            ['-c',"from poc_flashvep.dense_ar_output_aware_v2.c3_main import analyze; analyze('c3_continuous_iso8_output_bridge_v2.json')"]])]
    while tasks and execution_allowed():
        for task in list(tasks):
            predecessor,marker,commands=task
            if not (ROOT/'state'/predecessor).exists():continue
            if (ROOT/'state'/marker).exists():tasks.remove(task);continue
            results=[]
            for index,command in enumerate(commands):
                log=ROOT/'logs'/f'{marker.removesuffix(".json")}_{index}.log'
                assert not log.exists()
                with log.open('w') as stream:
                    value=subprocess.run([str(PYTHON)]+command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
                results.append(dict(command=command,returncode=value.returncode))
                if value.returncode:break
            write(ROOT/'state'/marker,dict(time=time.time(),passed=all(r['returncode']==0 for r in results),results=results,
                scope='CPU analysis of completed immutable runs. This marker is not a scientific performance or safety gate.'))
            print('PHASE_CPU_ANALYSIS',marker,results,flush=True);tasks.remove(task)
        time.sleep(15)
    if tasks:write(ROOT/'state/phase_cpu_analysis_pending_v2.json',dict(time=time.time(),tasks=tasks,status='BUDGET_LIMITED_PARTIAL'))

if __name__=='__main__':run()
