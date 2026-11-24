"""Recalculate terminal evidence after both frozen GPU schedules finish."""
import argparse
import json
import os
import subprocess
import time
from .common import ROOT,REPO,PYTHON,SOURCE,write,digest,preserve


def run():
    preserve();receipts=('main400_native_relative_complete_v2.json','heldout192_native_relative_complete_v1.json')
    while not all((ROOT/'state'/name).exists() for name in receipts):time.sleep(10)
    commands=[(name,[]) for name in ('failure_analysis','request_trace','policy_trace','causal_analysis',
        'decode_path_analysis','gate_overlap_analysis','runtime_bridge_analysis','reused_controls','work_audit',
        'intervention_analysis','profile_analysis')]
    commands += [('verification',[action]) for action in ('cpu','datasets','arrivals')]
    commands += [('verification',[action,'--final']) for action in ('inventory','metrics','budget')]
    commands += [('verification',['figures'])]
    results=[]
    for module,args in commands:
        command=[str(PYTHON),'-B','-m','poc_flashvep.dense_ar_native_ttft_slo.'+module,*args]
        before=time.time();process=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        row=dict(module=module,args=args,command=command,start_epoch=before,end_epoch=time.time(),
            returncode=process.returncode,passed=process.returncode==0,stdout=process.stdout,stderr=process.stderr)
        results.append(row)
        write(ROOT/'state/terminal_analysis_progress_v1.json',dict(time=time.time(),results=results))
        print('TERMINAL_ANALYSIS',module,args,process.returncode,flush=True)
        if not row['passed']:break
    value=dict(time=time.time(),complete=True,passed=len(results)==len(commands) and all(r['passed'] for r in results),
        results=results,source_files_SHA256={str(path.relative_to(REPO)):digest(path) for path in SOURCE.glob('*.py')},
        terminal_receipts_SHA256={name:digest(ROOT/'state'/name) for name in receipts},
        scope='Read and recalculate completed GPU evidence only. No new GPU work,scientific verdict,report approval,resource closeout or publication is automated here. '
            'The first CPU failure stops this pipeline. Keep its output before a targeted reader correction.')
    write(ROOT/'state/terminal_analysis_complete_v1.json',value);preserve();return value


def launch():
    path=ROOT/'state/terminal_analysis_controller_v1.json';assert not path.exists()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=['taskset','-c','160-175',str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.terminal_analysis','run']
    with (ROOT/'logs/terminal_analysis_v1.log').open('w') as stream:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(path,dict(time=time.time(),pid=process.pid,command=command,scope='CPU-only postprocessing after both GPU schedule receipts.'))
    return process.pid


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','run'));args=parser.parse_args()
    result=launch() if args.action=='launch' else run()
    print(result if isinstance(result,int) else result['passed'],flush=True)
