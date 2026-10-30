"""Finite Huginn adapter-correction gates. No serving-capacity measurements."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from .common import ROOT,write
from .resources import base
from .run import execute
from .qualification import compare


def gate(count,rate,seed,dataset,devices):
    names=[]
    try:
        for index,arm in enumerate(('PD_G1','PD_G1','B')):
            name=f'huginn_down_norm_adapter_qual_N{count}_seed{seed}_{arm}_{index}'
            job=dict(name=name,model='huginn',variant='down_norm_adapter',arm=arm,workload='diagnostic',
                     dataset=dataset,count=count,cap=128,rate=rate,seed=seed,gpus=devices,capacity=8,
                     timeout=3600,diagnostic=True)
            if execute(job).get('status')!='COMPLETE':return dict(status='BLOCKED_RUNTIME',failed=name)
            names.append(name)
            if index==1:
                repeated=compare(names)
                if not all(r['N']==count and r['exact']==count and r['complete'] for r in repeated):
                    return dict(status='NO_GO_CORRECTNESS',runs=names,comparisons=repeated)
        comparisons=compare(names)
        checks=[]
        for name in names:
            folder=ROOT/'raw'/name
            records=[r for line in (folder/'requests.jsonl').read_text().splitlines()
                     if (r:=json.loads(line)) and not r.get('warmup')]
            before=json.loads((folder/'before_measured.json').read_text())
            after=json.loads((folder/'after_measured.json').read_text())
            issued=sum(v.get('issued_loop_rows',0)-before.get(k,{}).get('issued_loop_rows',0) for k,v in after.items())
            expected=32*sum(r['input_tokens']+r['output_tokens']-1 for r in records)
            checks.append(dict(run=name,issued=issued,expected=expected,passed=issued==expected and
                all(len(r['exit_depths'])==r['output_tokens'] and all(d==32 for d in r['exit_depths']) for r in records)))
        passed=all(r['N']==count and r['exact']==count and r['complete'] for r in comparisons) and all(r['passed'] for r in checks)
        result=dict(status='FINITE_QUALIFICATION_PASS' if passed else 'NO_GO_CORRECTNESS',count=count,seed=seed,
                    rate=rate,GPUs=devices,runs=names,comparisons=comparisons,full_depth_work=checks,
                    scope='Internal fixed-R32 output parity. Not original-HF fidelity or answer accuracy.')
        write(ROOT/'state'/f'huginn_adapter_gate_N{count}_seed{seed}.json',result)
        return result
    finally:
        base.restore(devices)


def main():
    state=ROOT/'state/huginn_adapter_qualification.json'
    probes=[ROOT/'raw'/n/'result.json' for n in ('teacher_huginn_sample85_adapter_v9','teacher_huginn_sample56_adapter_v9')]
    while not all(p.exists() for p in probes):
        write(state,dict(status='WAITING_FOR_ADAPTER_REPLAY',pid=os.getpid(),time=time.time()))
        time.sleep(10)
    values=[json.loads(p.read_text()) for p in probes]
    if any(v['first_observed_difference'] or any(r['status']!='COMPLETE' for r in v['outputs']) or
           len({r['top1'] for r in v['outputs']})!=1 for v in values):
        write(state,dict(status='NO_GO_OPERATOR_REPLAY',probes=[str(p) for p in probes],time=time.time()));return
    time.sleep(5)
    devices=list(range(8))
    try:
        base.stop(devices)
        write(state,dict(status='RUNNING_SMALL_GATE',pid=os.getpid(),time=time.time()))
        small=gate(8,.2,2026103001,'W1',[0,1])
        if small['status']!='FINITE_QUALIFICATION_PASS':
            write(state,dict(status=small['status'],checks=[small],time=time.time()));return
        # gate restores its burners. Acquire that pair again before the larger plan.
        base.stop([0,1])
        jobs=[(200,.2,2026103001,'W1',[0,1]),(200,.2,2026109001,'W1',[2,3]),
              (400,0.,2026109001,'W1_repair_cycles',[4,5]),(400,0.,2026109002,'W1_repair_cycles',[6,7])]
        write(ROOT/'configs/huginn_adapter_qualification_plan.json',dict(small=small,jobs=jobs,
              scope='Four independent physical pairs. Numerical qualification only. Latency is not a clean serving benchmark.'))
        write(state,dict(status='RUNNING_EXPANDED_GATES',small=small,jobs=jobs,pid=os.getpid(),time=time.time()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(gate,*job) for job in jobs]
            results=[future.result() for future in futures]
        passed=all(r['status']=='FINITE_QUALIFICATION_PASS' for r in results)
        write(state,dict(status='FINITE_QUALIFICATION_PASS' if passed else 'NO_GO_CORRECTNESS',checks=[small,*results],
            time=time.time(),original_HF_fidelity='NOT_ESTABLISHED',downstream_accuracy='NOT_MEASURED',
            serving_capacity='NOT_RUN'))
    finally:
        base.restore(devices)


if __name__=='__main__':main()
