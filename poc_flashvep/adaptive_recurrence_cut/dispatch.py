"""Resume completed runs. Lease only validated, owned burner sessions."""
import argparse
import json
import os
import subprocess
import time
import zipfile
from .common import ROOT,REPO,PYTHON,RLT,environment,write,append,setup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('configs',nargs='+')
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args();setup()
    from . import resources
    pending=[]
    for name in args.configs:
        path=ROOT/'configs'/name
        cfg=json.loads(path.read_text())
        dest=ROOT/'raw'/cfg['name']
        status=dest/'status.json'
        if status.exists():
            value=json.loads(status.read_text())
            if value['status']=='COMPLETE':continue
            if dest.exists():raise RuntimeError(f'Refuse to overwrite failed or active namespace {dest}')
        pending.append((path,cfg))
    if not pending:return
    devices=sorted({g for _,cfg in pending for g in cfg['gpus']})
    leased=list(range(8)) if args.quiet else devices
    resources.base.stop(leased)
    active=[];failures=[]
    try:
        while pending or active:
            occupied={g for proc,cfg,handle in active for g in cfg['gpus']}
            for path,cfg in list(pending):
                if len(active)>=args.parallel:break
                if occupied.intersection(cfg['gpus']):continue
                dest=ROOT/'raw'/cfg['name'];dest.mkdir(parents=True,exist_ok=False)
                with zipfile.ZipFile(dest/'adaptive_source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                    for source in (REPO/'poc_flashvep/adaptive_recurrence_cut').glob('*.py'):
                        archive.write(source,'poc_flashvep/adaptive_recurrence_cut/'+source.name)
                    archive.writestr('rlt_dirty.patch',subprocess.check_output(['git','diff','--binary'],cwd=RLT))
                env=environment(cfg['gpus'])
                env.update(ARC_CONFIG=str(path),ARC_RUN=cfg['name'])
                env['ABC_LANE']='quiet_pair' if args.quiet and args.parallel==1 else 'parallel_correctness'
                handle=(ROOT/'logs'/f'{cfg["name"]}.log').open('w')
                command=[str(PYTHON),'-m','poc_flashvep.adaptive_recurrence_cut.worker',str(path)]
                proc=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                append(ROOT/'commands.jsonl',dict(time=time.time(),name=cfg['name'],command=command,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','ARC_CONFIG','ARC_RUN','PYTHONPATH')},pid=proc.pid,quiet=args.quiet))
                active.append((proc,cfg,handle));pending.remove((path,cfg));occupied.update(cfg['gpus'])
                print('START',cfg['name'],proc.pid,flush=True)
            for proc,cfg,handle in list(active):
                if proc.poll() is None:continue
                handle.close();active.remove((proc,cfg,handle))
                status=ROOT/'raw'/cfg['name']/'status.json'
                if proc.returncode or not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':
                    failures.append(cfg['name'])
                print('END',cfg['name'],proc.returncode,flush=True)
            time.sleep(.25)
    finally:
        # Do not restore a burner into a GPU while a surviving experiment owns it.
        if active:
            append(ROOT/'state/dispatcher_interruptions.jsonl',dict(time=time.time(),active=[dict(name=cfg['name'],pid=p.pid,gpus=cfg['gpus']) for p,cfg,h in active]))
        else:resources.base.restore(leased)
    if failures:raise RuntimeError(f'Failed namespaces: {failures}')


if __name__=='__main__':main()
