"""Finite correctness gates. Stop expansion after any failed output comparison."""
import argparse
import atexit
import json
import os
import time
from .common import ROOT,setup,write
from .run import execute


def compare(names):
    from .slowdown import summarize
    data=[]
    for name in names:
        path=ROOT/'raw'/name/'requests.jsonl'
        rows={r['sample_id']:r for line in path.read_text().splitlines()
              if (r:=json.loads(line)) and not r.get('warmup')}
        data.append(rows)
    output=[]
    for other in data[1:]:
        common=sorted(data[0].keys()&other.keys())
        changed=[]
        for i in common:
            x,y=data[0][i]['output_ids'],other[i]['output_ids']
            if x!=y:
                n=next((j for j,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)))
                changed.append(dict(sample_id=i,first_divergence=n))
        output.append(dict(N=len(common),exact=len(common)-len(changed),differences=changed,
                           complete=len(common)==len(data[0])==len(other)))
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=('ouro','huginn'),required=True)
    p.add_argument('--variant',default='down_norm')
    p.add_argument('--gpus',required=True)
    p.add_argument('--wait-for',default='')
    args=p.parse_args();setup()
    ledger=ROOT/'state'/f'{args.model}_{args.variant}_qualification.json'
    while args.wait_for:
        paths=[ROOT/'raw'/name/'status.json' for name in args.wait_for.split(',')]
        if all(path.exists() and json.loads(path.read_text()).get('status') in ('COMPLETE','FAILED') for path in paths):break
        write(ledger,dict(status='WAITING_FOR_EXISTING_REPRO',pid=os.getpid(),time=time.time()))
        time.sleep(5)
    devices=list(map(int,args.gpus.split(',')))
    from .resources import base
    atexit.register(base.restore,devices)
    # Correctness counts are finite. No arrival-rate or serving-capacity search.
    plan=([(8,128,1.,2026103001,'W1'),(200,128,1.,2026103001,'W1'),
           (400,128,1.,2026107001,'W1_repair_cycles'),(400,128,1.,2026109001,'W1_repair_cycles')]
          if args.model=='ouro' else
          [(8,128,.2,2026103001,'W1'),(200,128,.2,2026103001,'W1'),
           (200,128,.2,2026109001,'W1')])
    checks=[]
    for count,cap,rate,seed,dataset in plan:
        names=[]
        for index,arm in enumerate(('PD_G1','PD_G1','B')):
            name=f'{args.model}_{args.variant}_qual_N{count}_seed{seed}_{arm}_{index}'
            job=dict(name=name,model=args.model,variant=args.variant,arm=arm,workload='diagnostic',
                     dataset=dataset,count=count,cap=cap,rate=rate,seed=seed,gpus=devices,
                     capacity=8,timeout=2400,diagnostic=True)
            if execute(job).get('status')!='COMPLETE':
                write(ledger,dict(status='BLOCKED_RUNTIME',checks=checks,failed=name,time=time.time()));return
            names.append(name)
        comparisons=compare(names)
        work=[]
        expected_depth=4 if args.model=='ouro' else 32
        for name in names:
            records=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()
                     if line.strip()]
            records=[r for r in records if not r.get('warmup')]
            before=json.loads((ROOT/'raw'/name/'before_measured.json').read_text())
            after=json.loads((ROOT/'raw'/name/'after_measured.json').read_text())
            issued=sum(v.get('issued_loop_rows',0)-before.get(k,{}).get('issued_loop_rows',0) for k,v in after.items())
            expected=expected_depth*sum(r['input_tokens']+r['output_tokens']-1 for r in records)
            work.append(dict(run=name,issued=issued,expected=expected,passed=issued==expected and
                all(len(r['exit_depths'])==r['output_tokens'] and all(d==expected_depth for d in r['exit_depths']) for r in records)))
        passed=all(r['N']==count and r['exact']==count and r['complete'] for r in comparisons) and all(r['passed'] for r in work)
        check=dict(count=count,seed=seed,runs=names,comparisons=comparisons,full_depth_work=work,passed=passed)
        checks.append(check)
        write(ledger,dict(status='PASS_NEXT_GATE' if passed else 'NO_GO_CORRECTNESS',checks=checks,time=time.time()))
        print('GATE',json.dumps(check),flush=True)
        if not passed:return
    write(ledger,dict(status='FINITE_QUALIFICATION_PASS',checks=checks,time=time.time(),
                     original_HF_fidelity='UNRESOLVED',downstream_accuracy='NOT_MEASURED',
                     serving_capacity='NOT_RUN'))


if __name__=='__main__':main()
