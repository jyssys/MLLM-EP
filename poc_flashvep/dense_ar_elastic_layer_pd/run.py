"""Run one bounded GPU job with frozen source and ownership checks."""
import argparse,fcntl,json,os,signal,subprocess,time,zipfile
from .common import *


def execute(name,restore=True):
    initialize();preserve()
    for path,expected in json.loads((ROOT/'state/frozen_predecessor_hashes.json').read_text()).items():
        assert digest(REPO/path)==expected,'Frozen predecessor changed: '+path
    installed=ROOT/'state/installed_runtime_sources.json'
    if installed.exists():
        for path,expected in json.loads(installed.read_text())['hashes'].items():
            assert digest(path)==expected,'Installed Runtime changed: '+path
    assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    frozen_runtime=ROOT/'state/heldout_runtime_freeze.json'
    if name.startswith(('heldout_','enclosure_')):
        assert frozen_runtime.exists(),'Freeze execution source before confirmation'
        for path,expected in json.loads(frozen_runtime.read_text())['hashes'].items():
            assert digest(REPO/path)==expected,'Confirmation execution source changed: '+path
    cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text())
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if receipt.exists():return json.loads(receipt.read_text()).get('complete',False)
    assert not folder.exists(),'Preserve incomplete or failed namespace'
    with QUIET_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'GPU work closed while this job waited for its lease'
        owner=resources();owner.stop(list(range(8)))
        folder.mkdir();write(folder/'configuration.json',cfg)
        files=[]
        for package in ('dense_ar_elastic_layer_pd','dense_ar_dynamic_layer_pd','dense_ar_layerwise_pd','looped_pd_atomic_prefill','looped_pd_abc'):
            files.extend(p for p in (REPO/'poc_flashvep'/package).rglob('*.py') if '__pycache__' not in p.parts)
        archive=folder/'source.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,str(p.relative_to(REPO)))
        frozen=folder/'worker_source';frozen.mkdir()
        with zipfile.ZipFile(archive) as z:z.extractall(frozen)
        hashes={str(frozen/p.relative_to(REPO)):digest(frozen/p.relative_to(REPO)) for p in files}
        write(folder/'source_hashes.json',hashes)
        write(folder/'isolation_before.json',dict(time=time.time(),GPU_processes=owner.processes(),
            owned_burners=owner.burners(),host_load=os.getloadavg(),CPU_affinity=sorted(os.sched_getaffinity(0)),
            scope='All approved burners are quiet for isolated measurement. Unknown user processes are not terminated.'))
        env=dict(os.environ,PYTHONPATH=str(frozen),DENSE_FROZEN_ROOT=str(frozen),DENSE_CONFIG=str(cfg_path),
            PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
            DENSE_SOURCE_HASHES=json.dumps(hashes),VLLM_BATCH_INVARIANT='1',
            CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
            VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for key in ('ATOMIC_PD_WORKER','ABC_SOURCE_HASHES','ATOMIC_SOURCE_HASHES','BC_WORKER'):env.pop(key,None)
        module=cfg.get('module','native');command=[str(PYTHON),'-u','-m','poc_flashvep.dense_ar_elastic_layer_pd.'+module,str(cfg_path)]
        preflight_command=[str(PYTHON),'-m','unittest',
            'poc_flashvep.dense_ar_elastic_layer_pd.test_phase_stable_norm',
            'poc_flashvep.dense_ar_elastic_layer_pd.test_revised_policy',
            'poc_flashvep.dense_ar_elastic_layer_pd.test_execution_cut',
            'poc_flashvep.dense_ar_elastic_layer_pd.test_serial_handoff']
        preflight=subprocess.run(preflight_command,cwd=frozen,env=env,capture_output=True,text=True,timeout=30)
        write(folder/'CPU_preflight.json',dict(command=preflight_command,returncode=preflight.returncode,
            stdout=preflight.stdout,stderr=preflight.stderr,scope='Run tests against the exact immutable worker source before creating any model worker.'))
        if preflight.returncode:
            write(receipt,dict(time=time.time(),complete=False,status='FAILED_PREFLIGHT',returncode=preflight.returncode,
                normal_worker_exit=False,timing_isolation_passed=True,scope='No model worker started. Not a GPU performance result.'))
            if restore and not owner.processes():owner.restore(list(range(8)))
            return False
        child=None;started=time.time();interrupted=[False]
        def stop(*a):interrupted[0]=True
        old_signal={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with (ROOT/'logs'/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(folder/'command.json',dict(time=started,pid=child.pid,command=command,cwd=str(frozen),
                    config_SHA256=digest(cfg_path),source_archive_SHA256=digest(archive),publication_parent=BASE,
                    worktree_HEAD=git('rev-parse','HEAD'),GPU_mapping=cfg['gpus'],arrival_seed=cfg['arrival_seed']))
                print('START',name,child.pid,flush=True);term=None
                while child.poll() is None:
                    timed_out=time.time()>started+cfg.get('wall_timeout',900)
                    if interrupted[0] or timed_out:
                        assert os.getsid(child.pid)==child.pid
                        if term is None:os.killpg(child.pid,signal.SIGTERM);term=time.time()
                        elif time.time()-term>20:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(.5)
                terminal=folder/'status.json';state=json.loads(terminal.read_text()) if terminal.exists() else {}
                exits=folder/'worker_exits.json';normal=json.loads(exits.read_text()).get('passed',False) if exits.exists() else False
                observers=[json.loads(p.read_text()) for p in folder.glob('NVML-summary-*.json')]
                isolation=all(v.get('isolation_passed',True) for v in observers)
                complete=child.returncode==0 and state.get('status')=='COMPLETE' and normal and isolation
                write(receipt,dict(time=time.time(),complete=complete,returncode=child.returncode,
                    status='COMPLETE' if complete else 'FAILED',watchdog=term is not None,elapsed_seconds=time.time()-started,
                    normal_worker_exit=normal,timing_isolation_passed=isolation))
                print('END',name,child.returncode,complete,flush=True);return complete
        finally:
            if child is not None and child.poll() is None:
                assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for s,handler in old_signal.items():signal.signal(s,handler)
            preserve()
            if restore and not owner.processes():owner.restore(list(range(8)))


def main():
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');args=p.parse_args()
    try:
        for n in args.names:
            if not execute(n,restore=False):break
    finally:
        # Restoration must use the same lease as startup. Another launcher can
        # acquire the measurement lease before its first CUDA context exists.
        # An unlocked empty NVML query would then start Burners inside that job.
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()

