"""Dispatch gated BI0 jobs under the existing quiet-lane lock."""
import fcntl
import hashlib
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
    assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Campaign is closed'
    cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text())
    assert cfg.get('method','Native') in ('Native','Agg','Layer','Token')
    if cfg.get('axis_kind'):
        qualification_tag=cfg.get('runtime_qualification_tag','shared128_v2')
        if cfg.get('isolated_continuation_cost'):
            assert cfg.get('serial') and not cfg.get('state_capture') and not cfg.get('axis_state_capture')
            gate=json.loads((ROOT/'analysis'/f'BI0_axis_state_qualification_{qualification_tag}.json').read_text())
            assert any(r['axis']==cfg['axis_kind'] and r['cut']==cfg['axis_cut'] and r['passed'] for r in gate['results'])
            assert cfg.get('KV_block_tokens')==128 and cfg.get('restore_official_UCX_init')
        elif cfg.get('qualified_controlled_performance'):
            assert cfg.get('controlled_pipeline') and not cfg.get('serial')
            assert not cfg.get('state_capture') and not cfg.get('axis_state_capture')
            burst_tag=cfg.get('burst_gate_tag','shared128_v2')
            burst=json.loads((ROOT/'state'/f'controlled_axis_structural_gate_{burst_tag}.json').read_text())
            assert burst['passed']
            gate=json.loads((ROOT/'analysis'/f'BI0_axis_state_qualification_{qualification_tag}.json').read_text())
            assert any(r['axis']==cfg['axis_kind'] and r['cut']==cfg['axis_cut'] and r['passed'] for r in gate['results'])
            assert cfg.get('KV_block_tokens')==128 and cfg.get('restore_official_UCX_init')
        else:
            assert cfg.get('correctness_only') and (cfg.get('serial') or cfg.get('controlled_pipeline')), 'Online offloading is not yet qualified'
        assert cfg.get('execution_family')=='MATCHED_NO_COMPILE'
        async_qualified=False
        if cfg.get('qualified_async_runtime'):
            assert qualification_tag=='fast128_v3' and cfg.get('KV_block_tokens')==128 and cfg.get('restore_official_UCX_init')
            assert cfg.get('nixl_num_threads',4)==4
            async_qualified=json.loads((ROOT/'state/fast_native_control_state_v3_complete.json').read_text())['passed']
        assert not cfg.get('async_scheduling',True) or (cfg.get('correctness_only') and cfg.get('async_qualification_only')) or async_qualified, 'Async offload requires separate qualification'
        if cfg['axis_kind']=='Token' and not cfg.get('correctness_only'):
            teacher_tag=cfg.get('token_teacher_gate_tag','v1')
            teacher=json.loads((ROOT/'analysis'/f'BI0_Token_same_geometry_gate_{teacher_tag}.json').read_text())
            assert any(r['cut']==cfg['axis_cut'] and r['passed'] for r in teacher['results'])
    assert cfg['entry_module'].startswith('poc_flashvep.dense_ar_bi0_shortanswer_pd.')
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if receipt.exists():return json.loads(receipt.read_text())['complete']
    budget=json.loads((ROOT/'state/budget.json').read_text())
    assert time.time()<budget['last_large_dispatch_epoch'],'No new experiment dispatch in the reserved closeout hour'
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not stop.exists(),'Campaign dispatch is stopped after lease acquisition'
        assert not (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Campaign closed while waiting for the lease'
        assert time.time()<budget['last_large_dispatch_epoch'],'Budget changed while waiting for the GPU lease'
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
        with zipfile.ZipFile(archive) as bundle:
            hashes={member:hashlib.sha256(bundle.read(member)).hexdigest() for member in bundle.namelist()}
        write(folder/'source_hashes.json',hashes)
        env={k:v for k,v in os.environ.items() if not k.startswith(('DENSE_','DUETSERVE_'))}
        env.update(PYTHONPATH=str(frozen),CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),
            PATH=str(PYTHON.parent)+os.pathsep+os.environ.get('PATH',''),
            VLLM_BATCH_INVARIANT=str(int(cfg['batch_invariant'])),VLLM_ENABLE_V1_MULTIPROCESSING='0',
            OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if cfg.get('kv_layout'):env['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
        if cfg.get('restore_official_UCX_init'):env.setdefault('UCX_RCACHE_MAX_UNRELEASED','1024')
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
                exit_gate=json.loads(exits.read_text()).get('passed',False) if exits.exists() else cfg['entry_module']!='poc_flashvep.dense_ar_bi0_shortanswer_pd.serving'
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
