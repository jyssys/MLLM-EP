"""Bounded diagnostic runs with fresh namespaces and owned child sessions."""
import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from .common import ROOT,PRIOR,REPO,PYTHON,setup,write,environment,digest


def execute(job):
    setup()
    name=job['name']
    dest=ROOT/'raw'/name
    status=dest/'status.json'
    path=ROOT/'configs'/f'{name}.json'
    if path.exists() and json.loads(path.read_text())!=job: raise ValueError('Run contract conflict')
    if status.exists():
        value=json.loads(status.read_text())
        if value.get('status') in ('COMPLETE','FAILED'):
            print('REUSE',name,value['status'],flush=True)
            return value
        raise RuntimeError('An unfinished namespace requires review: '+name)
    write(path,job)
    with zipfile.ZipFile(ROOT/'configs'/f'{name}_diagnostic_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for source in __import__('pathlib').Path(__file__).parent.glob('*.py'):
            archive.write(source,source.name)
    dataset=job['dataset']+'.json'
    if not (ROOT/'data'/dataset).exists():
        shutil.copy2(PRIOR/'data'/dataset,ROOT/'data'/dataset)
        write(ROOT/'configs'/f'{job["dataset"]}_provenance.json',dict(source=str(PRIOR/'data'/dataset),
              sha256=digest(ROOT/'data'/dataset),copied_read_only_input=True))
    from .resources import base
    occupied=[r for r in base.processes() if r['gpu'] in job['gpus']]
    leased=set(base.burners())&set(job['gpus'])
    if occupied or leased:
        value=dict(status='FAILED',error='GPU lease preflight refused occupied or burner-reserved devices',
                   occupied=occupied,burner_reserved=sorted(leased),time=time.time())
        write(status,value)
        raise RuntimeError(value['error'])
    env=environment(job['gpus'])
    env.update(job.get('env',{}))
    env.update(B_NUMERICAL_RUN=name,B_NUMERICAL_VARIANT=job.get('variant','original'),
               ABC_ASYNC_EVIDENCE='1',ABC_LANE='isolated_quiet_diagnosis' if job.get('quiet') else 'parallel_numerical_diagnosis_not_capacity')
    cmd=[PYTHON,'-m','poc_flashvep.looped_pd_b_numerical_diagnosis.worker',str(path)]
    if job.get('nsys'):
        dest.mkdir(parents=True,exist_ok=True)
        env.update(B_NUMERICAL_NVTX='1',NSYS_NVTX_PROFILER_REGISTER_ONLY='0')
        cmd=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
             '--capture-range=nvtx','--nvtx-capture=diagnostic_measured','--capture-range-end=stop',
             '--cuda-graph-trace=node','-o',str(dest/'timeline')]+cmd
    with (ROOT/'logs'/f'{name}.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'state'/f'{name}.json',dict(pid=process.pid,command=cmd,job=job,started=time.time()))
        print('START',name,process.pid,flush=True)
        try: code=process.wait(timeout=job.get('timeout',1200)+400)
        except subprocess.TimeoutExpired:
            from ..looped_pd_b_slo_serving.jobs import clean_session
            clean_session(process.pid)
            code=process.wait(timeout=30)
        if code or not status.exists():
            write(status,dict(status='FAILED',returncode=code,time=time.time()))
    from ..looped_pd_b_slo_serving.jobs import clean_session
    clean_session(process.pid)
    result=json.loads(status.read_text())
    print('END',name,result,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=('ouro','huginn'),required=True)
    parser.add_argument('--variant',default='original')
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--cap',type=int,default=128)
    parser.add_argument('--rate',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=2026103001)
    parser.add_argument('--dataset',default='W1')
    parser.add_argument('--tag',required=True)
    parser.add_argument('--arms',default='PD_G1,PD_G1,B')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--kv-verify',action='store_true')
    parser.add_argument('--queued-compute',action='store_true')
    parser.add_argument('--nsys',action='store_true')
    parser.add_argument('--capacity',type=int,default=8)
    parser.add_argument('--capture-seed',type=int)
    parser.add_argument('--preserve-source-seeds',action='store_true')
    args=parser.parse_args()
    for index,arm in enumerate(args.arms.split(',')):
        job=dict(name=f'{args.tag}_{arm}_{index}',model=args.model,arm=arm,variant=args.variant,
                 workload='diagnostic',dataset=args.dataset,count=args.count,rate=args.rate,seed=args.seed,
                 gpus=list(map(int,args.gpus.split(','))),cap=args.cap,capacity=args.capacity,timeout=1800,diagnostic=True,
                 nsys=args.nsys,
                 env={**({'B_NUMERICAL_TRANSFER_PROBE':'1'} if args.probe else {}),
                      **({'ABC_VERIFY_KV':'1'} if args.kv_verify else {}),
                      **({'B_NUMERICAL_CAPTURE_SEED':str(args.capture_seed)} if args.capture_seed is not None else {}),
                      **({'B_NUMERICAL_PRESERVE_SOURCE_SEEDS':'1'} if args.preserve_source_seeds else {}),
                      **({'B_NUMERICAL_QUEUED_COMPUTE':'1'} if args.queued_compute else {})})
        if execute(job).get('status')!='COMPLETE': break


if __name__=='__main__': main()
