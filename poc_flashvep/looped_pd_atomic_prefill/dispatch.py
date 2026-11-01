"""Run one isolated job. Preserve failures and restore verified owned burners."""
import argparse
import fcntl
import signal
import sys
import subprocess
import zipfile
from .common import *


def pressure(name):
    try:return Path('/proc/pressure',name).read_text()
    except OSError:return 'NOT_AVAILABLE'


def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--keep-quiet',action='store_true');a=p.parse_args()
    setup();path=ROOT/'configs'/a.name;cfg=json.loads(path.read_text())
    qualification=cfg.get('phase') in ('correctness','long_correctness','chunk_correctness','final_correctness')
    lease=(ROOT/'state/gpu_lease.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_SH if qualification else fcntl.LOCK_EX)
    from .bootstrap import preserve
    preserve();folder=ROOT/'raw'/cfg['name']
    if folder.exists():
        status=folder/'status.json'
        if status.exists() and json.loads(status.read_text())['status']=='COMPLETE':return
        raise FileExistsError('Preserve failed or incomplete namespace: '+cfg['name'])
    from .resources import inherited as resources
    resources.stop(cfg['gpus'] if qualification else list(range(8)));folder.mkdir(exist_ok=False)
    write(folder/'isolation_before.json',dict(time=time.time(),qualification_only=qualification,
        GPU_processes=resources.processes(),owned_burners=resources.burners(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        host_load_average=os.getloadavg(),scope='GPU ownership and burner isolation. Shared-host CPU work is observed, not forcibly terminated.'))
    write(folder/'configuration.json',cfg)
    with zipfile.ZipFile(folder/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for package in ('looped_pd_atomic_prefill','looped_pd_bottleneck_crossover','adaptive_loop_token_pd_2gpu',
            'adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis'):
            for source in (REPO/'poc_flashvep'/package).rglob('*.py'):z.write(source,str(source.relative_to(REPO)))
        z.writestr('RLT_private_diff.patch',subprocess.check_output(['git','diff','--binary'],cwd=RLT))
    env=environment(cfg['gpus']);env.update(ARC_CONFIG=str(path),ARC_RUN=cfg['name'],ABC_LANE='isolated_atomic_pd')
    if cfg['worker']=='native_mixed_worker':
        env.pop('ATOMIC_PD_WORKER',None);env['ATOMIC_NATIVE_COLOCATED']='1'
        env['PYTHONPATH']=str(SOURCE/'native_mixed_bootstrap')+os.pathsep+env['PYTHONPATH']
    runtime_names={'common.py','ar_worker.py','ar_capacity.py'} if cfg['model']=='qwen8' else {'common.py','runtime.py','rlt_worker.py','capacity_worker.py'}
    runtime_names.add(cfg['worker']+'.py')
    if cfg['worker'] in ('capacity_worker','reference_worker'):runtime_names.add('local_metrics.py')
    sources=[p for p in SOURCE.glob('*.py') if p.name in runtime_names]
    if cfg['worker']=='native_mixed_worker':sources.append(SOURCE/'native_mixed_bootstrap/sitecustomize.py')
    env['ATOMIC_SOURCE_HASHES']=json.dumps({str(p):digest(p) for p in sources})
    command=[str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.'+cfg['worker'],str(path)]
    if cfg.get('nsys'):
        command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace='+cfg.get('nsys_graph_trace','node'),
            '--trace-fork-before-exec=true','--force-overwrite=false','--output='+str(folder/'nsys_report')]+command
    child=None;interrupted=[False]
    def request_stop(*args):interrupted[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        with (ROOT/'logs'/(cfg['name']+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            append(ROOT/'commands.jsonl',dict(time=time.time(),name=cfg['name'],pid=child.pid,command=command,gpus=cfg['gpus'],
                repo_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                source_archive_SHA256=digest(folder/'source.zip'),config_SHA256=digest(path),clean=not qualification,
                isolation='DISJOINT_GPU_QUALIFICATION_NOT_CLEAN_TIMING' if qualification else 'EXCLUSIVE_ALL_GPU_QUIET_LANE'))
            print('START',cfg['name'],child.pid,flush=True);deadline=time.monotonic()+cfg.get('timeout',3600)+900
            stopped=None;host_sample=0.;failed_since=None
            while child.poll() is None:
                if time.monotonic()-host_sample>=1:
                    host_sample=time.monotonic()
                    append(folder/'host_resource_samples.jsonl',dict(time=host_sample,load_average=os.getloadavg(),
                        CPU_pressure=pressure('cpu'),memory_pressure=pressure('memory'),IO_pressure=pressure('io')))
                worker_status=folder/'status.json'
                if worker_status.exists():
                    try:failed=json.loads(worker_status.read_text()).get('status')=='FAILED'
                    except json.JSONDecodeError:failed=False
                    if failed and failed_since is None:failed_since=time.monotonic()
                    if failed_since is not None and time.monotonic()-failed_since>10:interrupted[0]=True
                if interrupted[0] or time.monotonic()>deadline:
                    assert os.getsid(child.pid)==child.pid
                    if stopped is None:os.killpg(child.pid,signal.SIGTERM);stopped=time.monotonic()
                    elif time.monotonic()-stopped>30:os.killpg(child.pid,signal.SIGKILL)
                time.sleep(.5)
            write(folder/'dispatcher_result.json',dict(time=time.time(),pid=child.pid,returncode=child.returncode,watchdog_or_signal=stopped is not None))
            print('END',cfg['name'],child.returncode,flush=True)
            assert child.returncode==0,'Job failed. Preserve its complete log.'
            status=folder/'status.json'
            assert status.exists() and json.loads(status.read_text())['status']=='COMPLETE','Missing valid terminal worker status'
    finally:
        if child is not None and child.poll() is None:
            error=repr(sys.exc_info()[1]);assert os.getsid(child.pid)==child.pid
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            write(folder/'dispatcher_result.json',dict(time=time.time(),pid=child.pid,returncode=child.returncode,
                dispatcher_aborted=True,error=error,scope='Only the registered owned child session was stopped. Raw evidence is preserved.'))
        if child is None or child.poll() is not None:
            if not a.keep_quiet and not resources.processes():resources.restore(list(range(8)))
        lease.close()


if __name__=='__main__':main()
