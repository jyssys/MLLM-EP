"""Run the newly requested bounded attention-head PoC under a pair lease."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile
from .common import ROOT,REPO,SOURCE,PYTHON,MODEL_REVISION,write,digest,bind,preserve,resources,dispatch_allowed,execution_allowed

POC=ROOT/'attention_poc'


def prepare():
    bind();preserve()
    for name in ('configs','raw','logs','analysis','figures','state','references'):(POC/name).mkdir(parents=True,exist_ok=True)
    data_path=ROOT/'data/longbench_balanced400_v2.json';data=json.loads(data_path.read_text())
    selected=[]
    for target in (8192,16384,32768):
        row=min(data,key=lambda value:(abs(value['input_tokens']-target),value['prompt_id']))
        selected.append(dict(prompt_id=row['prompt_id'],input_tokens=row['input_tokens'],input_ids_SHA256=row['input_ids_SHA256'],target_label=target))
    parent=json.loads((ROOT/'configs/PC_C3_EqualReplay_s0_OriginalLayer.json').read_text())
    source=ROOT/'references/source_capsules'/parent['source_capsule']
    with zipfile.ZipFile(source) as bundle:files={name:bundle.read(name) for name in bundle.namelist()}
    for name in ('common.py','__init__.py','attention_capture.py','attention_benchmark.py'):
        files['poc_flashvep/dense_ar_p_critical_path/'+name]=(SOURCE/name).read_bytes()
    archive=POC/'references/attention_poc_source_v1.zip';assert not archive.exists()
    for name,value in files.items():
        if name.endswith('.py'):compile(value,name,'exec')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as stream:
        for name,value in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;stream.writestr(item,value)
    cfg=dict(name='Capture_v1',pair=[0,1],prompts=selected,layers=[0,1,2,3],layer_counts=[1,2,4],remote_KV_heads=[2,4],
        dataset_SHA256=digest(data_path),model_revision=MODEL_REVISION,warmup_iterations=5,iterations=30,
        source_archive=str(archive.relative_to(ROOT)),source_archive_SHA256=digest(archive),
        scope='Added at user request. A bounded isolated attention-head feasibility PoC,not a replacement for the ongoing P-critical-path campaign. '
            'Select the closest real input to8K/16K/32K by length and prompt ID only. No padding,cropping,output-based selection or dataset change. '
            'Capture real full36 Qwen BF16/BI0 input tensors. Preserve all GQA groups and actual B8192 chunks. '
            'Measure local-full,local-split and peer-offloaded attention call sequences at1/2/4 selected layers. '
            'D is idle. No unsafe active-decode insertion. Full-model candidate greedy/task-quality and online serving remain NOT_RUN. '
            'Transport uses torch peer copies,not NIXL/NCCL. Report payload,packing,joining and real two-device waiting. '
            'Three benchmark process starts are not independent serving starts. No sustainable capacity claim.')
    path=POC/'configs/attention_poc_v1.json';assert not path.exists();write(path,cfg)
    write(POC/'references/source_manifest_v1.json',dict(SHA256=digest(archive),parent_SHA256=digest(source),
        files={name:hashlib.sha256(value).hexdigest() for name,value in files.items()}))
    log=POC/'logs/controller_v1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.attention_poc','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(POC/'state/controller_v1.json',dict(time=time.time(),pid=child.pid));return child.pid


def child_run(module,name,visible,extra=(),version=1):
    from .stable_source import source_tree
    cfgpath=POC/'configs'/f'attention_poc_v{version}.json';cfg=json.loads(cfgpath.read_text());source=ROOT/cfg['source_archive']
    pair=cfg['pair'];affinity=list(range(pair[0]//2*40,(pair[0]//2+1)*40))
    assert digest(source)==cfg['source_archive_SHA256'];frozen=source_tree(source)
    folder=POC/'raw'/name;assert not folder.exists();folder.mkdir()
    env={key:value for key,value in os.environ.items() if not key.startswith(('DENSE_','DUETSERVE_'))}
    env.update(CUDA_VISIBLE_DEVICES=visible,PYTHONPATH=str(frozen),PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'],
        PYTHONDONTWRITEBYTECODE='1',VLLM_BATCH_INVARIANT='0',
        VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',VLLM_KV_CACHE_LAYOUT='HND',MAX_JOBS='4')
    command=[str(PYTHON),'-B','-u','-m','poc_flashvep.dense_ar_p_critical_path.'+module,str(cfgpath)]+list(extra)
    lease=ROOT/'state'/f'pair_lease_{pair[0]}_{pair[1]}.json';child=None;started=time.time()
    try:
        with (POC/'logs'/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:os.sched_setaffinity(0,set(affinity)))
            write(folder/'command.json',dict(command=command,pid=child.pid,GPUs=pair,source_archive_SHA256=digest(source),
                config_SHA256=digest(cfgpath),CUDA_VISIBLE_DEVICES=visible,start_epoch=started,CPU_affinity=affinity))
            write(lease,dict(name='AttentionPoC_'+name,pid=child.pid,GPUs=pair,status='RUNNING',time=time.time()))
            stopped=None
            while child.poll() is None:
                if not execution_allowed() or time.time()-started>1800:
                    if stopped is None:os.killpg(child.pid,signal.SIGINT);stopped=time.time()
                    elif time.time()-stopped>60:os.killpg(child.pid,signal.SIGKILL)
                time.sleep(.5)
            status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            ok=child.returncode==0 and status.get('status')=='COMPLETE'
            write(folder/'dispatcher_result.json',dict(complete=ok,returncode=child.returncode,elapsed_seconds=time.time()-started))
            return ok
    finally:
        if child is not None and child.poll() is None:
            assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(lease,dict(name='AttentionPoC_'+name,GPUs=pair,status='RELEASED',time=time.time()))
        preserve()


def run(version=1):
    bind();owner=resources();result=dict(time=time.time(),jobs=[],status='NOT_RUN')
    cfg=json.loads((POC/'configs'/f'attention_poc_v{version}.json').read_text())
    pair=cfg['pair']
    if cfg.get('parent_completion'):
        path=POC/'state'/cfg['parent_completion']
        while not path.exists():
            if not dispatch_allowed():
                result['reason']='DISPATCH_BUDGET_BEFORE_PARENT';write(POC/'state'/f'completion_v{version}.json',result);return
            time.sleep(5)
        if json.loads(path.read_text())['status']!='COMPLETE':
            result['reason']='PARENT_CAPTURE_OR_BENCHMARK_FAILED';write(POC/'state'/f'completion_v{version}.json',result);return
        write(POC/'state/active_protocol.json',dict(version=version,config=f'attention_poc_v{version}.json',completion=f'completion_v{version}.json'))
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            guardian=json.loads((ROOT/'state/guardian.json').read_text())
            assert guardian['status']=='ACTIVE' and Path('/proc',str(guardian['pid']),'cmdline').exists()
            if not dispatch_allowed():result['reason']='DISPATCH_BUDGET';return
            capture=cfg['name'];owner.stop(pair)
            if cfg.get('reuse_capture'):
                manifest=POC/'raw'/cfg['capture_name']/'capture_manifest.json'
                value=json.loads(manifest.read_text());assert value['passed'] and value['dataset_SHA256']==cfg['dataset_SHA256']
                result['capture_manifest_SHA256']=digest(manifest)
                result['jobs'].append(dict(name=cfg['capture_name'],status='REUSED_CORRECTNESS_ONLY_CAPTURE'))
            else:
                assert child_run('attention_capture',capture,str(pair[0]),version=version)
                result['jobs'].append(dict(name=capture,status='COMPLETE'))
            for index in range(3):
                name=cfg.get('benchmark_prefix','Benchmark')+'_s'+str(index)
                if not dispatch_allowed():
                    result['jobs'].append(dict(name=name,status='NOT_RUN',reason='DISPATCH_BUDGET'));continue
                assert child_run('attention_benchmark',name,','.join(map(str,pair)),('--start',str(index)),version=version)
                result['jobs'].append(dict(name=name,status='COMPLETE'))
            result['status']='COMPLETE'
        except Exception as error:result.update(status='FAILED',error=repr(error))
        finally:
            owner.restore(pair);preserve();write(POC/'state'/f'completion_v{version}.json',result)
    print('ATTENTION_POC_COMPLETE',result['status'],flush=True)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[2]) if len(sys.argv)>2 else 1),flush=True)
