"""Resume-safe dispatcher with a hard wall-clock cutoff and owned-resource lease."""
import argparse
import json
import os
import signal
import subprocess
import time
import zipfile
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('configs',nargs='+');p.add_argument('--parallel',type=int,default=1)
    p.add_argument('--quiet',action='store_true');p.add_argument('--keep-quiet',action='store_true');a=p.parse_args();setup()
    if a.keep_quiet and not a.quiet:raise ValueError('Persistent quiet lease requires --quiet')
    frozen=ROOT/'configs/frozen_execution_hashes.json'
    if frozen.exists():
        for source,expected in json.loads(frozen.read_text())['hashes'].items():
            if digest(source)!=expected:raise RuntimeError('Frozen execution source changed: '+source)
    from .resources import base as resources
    pending=[]
    for name in a.configs:
        path=ROOT/'configs'/name;cfg=json.loads(path.read_text());dest=ROOT/'raw'/cfg['name']
        if dest.exists():
            status=dest/'status.json'
            if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':continue
            raise FileExistsError('Preserved existing namespace: '+str(dest))
        pending.append((path,cfg))
    if not pending:return
    devices=list(range(8)) if a.quiet else sorted({g for _,c in pending for g in c['gpus']})
    resources.stop(devices);active=[];failed=[]
    try:
        while pending or active:
            occupied={g for _,c,_ in active for g in c['gpus']}
            for path,cfg in list(pending):
                if len(active)>=a.parallel:break
                if occupied.intersection(cfg['gpus']):continue
                if remaining()<max(3600,cfg.get('estimated_seconds',300)+3600):
                    append(ROOT/'state/skipped.jsonl',dict(name=cfg['name'],reason='NOT_RUN_TIME_BUDGET',time=time.time()))
                    pending.remove((path,cfg));continue
                dest=ROOT/'raw'/cfg['name'];dest.mkdir(exist_ok=False)
                with zipfile.ZipFile(dest/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                    for folder in ('adaptive_loop_token_pd_2gpu','adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis'):
                        for source in (REPO/'poc_flashvep'/folder).glob('*.py'):archive.write(source,str(source.relative_to(REPO)))
                    archive.writestr('rlt_dirty.patch',subprocess.check_output(['git','diff','--binary'],cwd=RLT))
                write(dest/'source_provenance.json',dict(
                    repo_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
                    RLT_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
                    archive_SHA256=digest(dest/'source.zip'),config_SHA256=digest(path),
                    time=time.time(),interpretation='HEAD plus archived process-local overlays. The worktree is not assumed clean.'))
                env=environment(cfg['gpus']);env.update(ARC_CONFIG=str(path),ARC_RUN=cfg['name'],LTP_WORKER='1',ABC_LANE='quiet_pair' if a.quiet else 'parallel_correctness')
                env['LTP_SOURCE_HASHES']=json.dumps({str(SOURCE/name):digest(SOURCE/name)
                    for name in ('common.py','worker.py','connector.py','policy.py')})
                handle=(ROOT/'logs'/f'{cfg["name"]}.log').open('w')
                command=[str(PYTHON),'-m','poc_flashvep.adaptive_loop_token_pd_2gpu.worker',str(path)]
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                append(ROOT/'commands.jsonl',dict(time=time.time(),name=cfg['name'],command=command,pid=child.pid,gpus=cfg['gpus'],quiet=a.quiet,
                    environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','ARC_CONFIG','ARC_RUN','LTP_WORKER','ABC_LANE','PYTHONPATH','ABC_HUGINN_REFERENCE','OMP_NUM_THREADS')}))
                active.append((child,cfg,handle));pending.remove((path,cfg));occupied.update(cfg['gpus'])
                print('START',cfg['name'],child.pid,flush=True)
            for child,cfg,handle in list(active):
                if remaining()<300 and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                if child.poll() is None:continue
                handle.close();active.remove((child,cfg,handle));status=ROOT/'raw'/cfg['name']/'status.json'
                if child.returncode or not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':failed.append(cfg['name'])
                print('END',cfg['name'],child.returncode,flush=True)
            write(ROOT/'state'/f'dispatcher_{os.getpid()}.json',dict(time=time.time(),
                active=[dict(name=c['name'],pid=p.pid,gpus=c['gpus']) for p,c,h in active],
                pending=[c['name'] for _,c in pending],failed=failed))
            budget(status='GPU_RUNS' if active else 'BETWEEN_RUNS',active=[c['name'] for _,c,_ in active],
                   current_gpu_ownership=[dict(name=c['name'],pid=p.pid,gpus=c['gpus']) for p,c,h in active],
                   pending=[c['name'] for _,c in pending],failed=failed)
            time.sleep(1.)
    finally:
        if not active and not a.keep_quiet:resources.restore(devices)
        else:append(ROOT/'state/dispatcher_interruption.jsonl',dict(time=time.time(),active=[dict(name=c['name'],pid=p.pid) for p,c,h in active]))
    if failed:raise RuntimeError('Failed runs: '+str(failed))

if __name__=='__main__':main()
