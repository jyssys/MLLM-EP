"""Resume-safe, ownership-checked launch with a twelve-hour campaign boundary."""
import argparse
import signal
import subprocess
import zipfile
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--parallel',type=int,default=1)
    p.add_argument('--nonclean',action='store_true');p.add_argument('--keep-quiet',action='store_true');a=p.parse_args();setup()
    from .bootstrap import preserve
    preserve()
    from .resources import base as resources
    todo=[]
    for name in a.names:
        path=ROOT/'configs'/name;cfg=json.loads(path.read_text());folder=ROOT/'raw'/cfg['name']
        if cfg.get('phase') in ('crossover','focused_crossover','critical_repeat','primary','repeat','protocol_control','static_tuning'):
            slo=ROOT/'configs'/('frozen_SLO_qwen.json' if cfg['model']=='qwen' else 'frozen_SLO.json')
            assert slo.exists() and cfg['model'] in json.loads(slo.read_text()),'Freeze the model SLO before held-out serving'
        if folder.exists():
            status=folder/'status.json'
            if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':continue
            raise FileExistsError('Preserve failed or active namespace: '+str(folder))
        todo.append((path,cfg))
    if not todo:return
    devices=sorted({g for _,cfg in todo for g in cfg['gpus']}) if a.nonclean else list(range(8))
    resources.stop(devices);running=[];failed=[];terminating={};interrupted=[False]
    def request_stop(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        while todo or running:
            if remaining()<1800 or interrupted[0]:
                for child,cfg,handle in running:
                    if child.poll() is None and os.getsid(child.pid)==child.pid:
                        if child.pid not in terminating:
                            os.killpg(child.pid,signal.SIGTERM);terminating[child.pid]=time.monotonic()
                            append(ROOT/'state/not_run.jsonl',dict(name=cfg['name'],reason='STOPPED_TIME_BUDGET' if not interrupted[0] else 'CONTROLLED_INTERRUPT',time=time.time()))
                        elif time.monotonic()-terminating[child.pid]>20:os.killpg(child.pid,signal.SIGKILL)
                todo.clear()
            occupied={g for _,cfg,_ in running for g in cfg['gpus']}
            for path,cfg in list(todo):
                if len(running)>=a.parallel:break
                if occupied.intersection(cfg['gpus']):continue
                if remaining()<max(3600,cfg.get('estimated_seconds',600)+2700):
                    append(ROOT/'state/not_run.jsonl',dict(name=cfg['name'],reason='NOT_RUN_TIME_BUDGET',time=time.time()));todo.remove((path,cfg));continue
                folder=ROOT/'raw'/cfg['name'];folder.mkdir(exist_ok=False)
                with zipfile.ZipFile(folder/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                    for package in ('looped_pd_bottleneck_crossover','adaptive_loop_token_pd_2gpu','adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis'):
                        for source in (REPO/'poc_flashvep'/package).glob('*.py'):archive.write(source,str(source.relative_to(REPO)))
                    archive.writestr('RLT_private_diff.patch',subprocess.check_output(['git','diff','--binary'],cwd=RLT))
                env=environment(cfg['gpus']);env.update(ARC_CONFIG=str(path),ARC_RUN=cfg['name'],BC_WORKER='1',ABC_LANE='nonclean_gate' if a.nonclean else 'isolated_service')
                frozen_sources=list(SOURCE.glob('*.py'))+list((REPO/'poc_flashvep/adaptive_loop_token_pd_2gpu').glob('*.py'))
                env['LTP_SOURCE_HASHES']=json.dumps({str(s):digest(s) for s in frozen_sources})
                command=[str(PYTHON),'-m','poc_flashvep.looped_pd_bottleneck_crossover.'+cfg['worker'],str(path)]
                handle=(ROOT/'logs'/f'{cfg["name"]}.log').open('w')
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                append(ROOT/'commands.jsonl',dict(name=cfg['name'],pid=child.pid,time=time.time(),command=command,gpus=cfg['gpus'],clean=not a.nonclean,
                    environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','ARC_CONFIG','ARC_RUN','PYTHONPATH','ABC_HUGINN_REFERENCE')},
                    source_archive_SHA256=digest(folder/'source.zip'),repo_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
                running.append((child,cfg,handle));todo.remove((path,cfg));occupied.update(cfg['gpus'])
                print('START',cfg['name'],child.pid,flush=True)
            for child,cfg,handle in list(running):
                if child.poll() is None:continue
                handle.close();running.remove((child,cfg,handle));status=ROOT/'raw'/cfg['name']/'status.json'
                write(ROOT/'raw'/cfg['name']/'dispatcher_result.json',dict(returncode=child.returncode,time=time.time(),pid=child.pid))
                if child.returncode or not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':failed.append(cfg['name'])
                print('END',cfg['name'],child.returncode,flush=True)
            budget(status='GPU_RUNS' if running else 'BETWEEN_RUNS',active=[c['name'] for _,c,_ in running],failed=failed,
                owned_worker_sessions=[dict(pid=child.pid,name=c['name'],gpus=c['gpus']) for child,c,_ in running])
            time.sleep(1)
    finally:
        if not running and not a.keep_quiet:resources.restore(devices)
    if failed:raise RuntimeError('Preserved failed runs: '+str(failed))

if __name__=='__main__':main()
