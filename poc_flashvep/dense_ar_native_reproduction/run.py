"""Dispatch only Native/reference jobs under the existing quiet-lane lock."""
import fcntl
import json
import os
import signal
import subprocess
import time
import zipfile
from .common import ROOT,SOURCE,REPO,PYTHON,BASE,QUIET_LOCK,write,digest,resources,initialize,preserve


def execute(name):
    initialize();preserve()
    stop=ROOT/'state/STOP_REQUESTED.json'
    assert not stop.exists(),'Campaign dispatch is stopped'
    cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text())
    assert cfg.get('method','Native') in ('Native','Agg') and not cfg.get('token_axis') and cfg.get('static_layers',0)==0
    assert cfg['entry_module'].startswith('poc_flashvep.dense_ar_native_reproduction.')
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if receipt.exists():return json.loads(receipt.read_text())['complete']
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not stop.exists(),'Campaign dispatch is stopped after lease acquisition'
        assert not folder.exists(),'Keep failed/incomplete evidence; use a new job name'
        preserve();folder.mkdir();write(folder/'configuration.json',cfg)
        if cfg.get('dataset'):
            dataset=ROOT/'data'/cfg['dataset'];assert dataset.exists()
            if cfg.get('dataset_SHA256'):assert digest(dataset)==cfg['dataset_SHA256']
            write(folder/'input_provenance.json',dict(dataset=str(dataset),dataset_SHA256=digest(dataset),
                model_revision='b968826d9c46dd6066d109eabc6255188de91218',config_SHA256=digest(cfg_path)))
        files=list(SOURCE.glob('*.py'))+[REPO/'poc_flashvep/__init__.py']
        files.extend((REPO/'poc_flashvep/dense_ar_slo_layer_pd').glob('common.py'))
        archive=folder/'source.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
            for p in files:
                if p.exists():bundle.write(p,str(p.relative_to(REPO)))
        frozen=folder/'worker_source';frozen.mkdir()
        with zipfile.ZipFile(archive) as bundle:bundle.extractall(frozen)
        write(folder/'source_hashes.json',{str(p.relative_to(REPO)):digest(p) for p in files if p.exists()})
        env={k:v for k,v in os.environ.items() if not k.startswith(('DENSE_','DUETSERVE_'))}
        env.update(PYTHONPATH=str(frozen),CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),
            PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
            VLLM_BATCH_INVARIANT=str(int(cfg['batch_invariant'])),VLLM_ENABLE_V1_MULTIPROCESSING='0',
            OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if cfg.get('kv_layout'):env['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
        owner=resources();owner.stop(list(range(8)));assert not owner.processes()
        command=[str(PYTHON),'-u','-m',cfg['entry_module'],str(cfg_path)]
        interrupted=[False];child=None;started=time.time()
        def requested(*_):interrupted[0]=True
        handlers={s:signal.signal(s,requested) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with (ROOT/'logs'/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(folder/'command.json',dict(command=command,pid=child.pid,GPUs=cfg['gpus'],
                    source_archive_SHA256=digest(archive),config_SHA256=digest(cfg_path),publication_parent=BASE,
                    environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','VLLM_ENABLE_V1_MULTIPROCESSING','OMP_NUM_THREADS')}))
                write(ROOT/'state/GPU_lease_current.json',dict(name=name,pid=child.pid,GPUs=cfg['gpus'],status='RUNNING'))
                print('START',name,child.pid,flush=True);term=None
                while child.poll() is None:
                    if interrupted[0] or time.time()-started>cfg.get('wall_timeout',900):
                        if term is None:
                            write(stop,dict(time=time.time(),reason='Signal or watchdog. Do not dispatch another job.'))
                            os.killpg(child.pid,signal.SIGINT);term=time.time()
                        elif time.time()-term>75:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(.5)
                state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                exits=folder/'worker_exits.json'
                exit_gate=json.loads(exits.read_text()).get('passed',False) if exits.exists() else cfg['entry_module']!='poc_flashvep.dense_ar_native_reproduction.serving'
                passed=child.returncode==0 and state.get('status')=='COMPLETE' and exit_gate
                write(receipt,dict(complete=passed,returncode=child.returncode,elapsed_seconds=time.time()-started,
                    interrupted=interrupted[0],source_SHA256=digest(archive)))
                print('END',name,passed,flush=True)
                if interrupted[0]:raise SystemExit('STOP_REQUESTED: do not continue the controller loop')
                return passed
        finally:
            if child is not None and child.poll() is None:
                assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for signum,handler in handlers.items():signal.signal(signum,handler)
            write(ROOT/'state/GPU_lease_current.json',dict(name=name,status='RELEASED',time=time.time()))
            preserve()
            if not owner.processes():owner.restore(list(range(8)))
