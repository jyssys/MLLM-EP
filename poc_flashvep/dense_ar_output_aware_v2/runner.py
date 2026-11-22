"""Dispatch immutable v2 jobs with explicit composition proofs and pair leases."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile
from .common import ROOT,REPO,PYTHON,BASE,write,digest,initialize,preserve,resources,dispatch_allowed,execution_allowed

VARIABLE_FIELDS={'name','dataset','dataset_SHA256','count','arrival_offsets','qps','seed','gpus','output_cap','ignore_eos',
    'correctness_only','axis_state_capture','state_capture','numerical_probe','numerical_probe_sync','gpu_state_digest',
    'verify_GPU_digest_with_CPU','serving_timeout','wall_timeout','source_capsule','source_capsule_SHA256','entry_module',
    'serial','scope','evidence','discovery_only','confirmation','mechanism_input_bucket','mechanism_output_tokens',
    'qualification_gate_v2','inherited_state_proofs','request_holdout_scope','teacher_eager_all_prompt',
    'teacher_eager_prefix_compiled_suffix','teacher_mode0_IR_norm_priority','token_teacher_cut','teacher_suffix_fraction',
    'reference_single','suffix_by_request','aggregate_work_audit'}
VARIABLE_FIELDS.update({'axis_hash_workers','axis_capture_hidden_last_only','axis_import_hash_after_model',
    'gpu_digest_selftest','quality_contract','kernel_activity','phase','mechanism_capacity_reference'})
VARIABLE_FIELDS.update({'diagnostic_controlled_output_replay','replay_output_lengths'})
VARIABLE_FIELDS.update({'kernel_activity_roles','collector_namespace','runtime_api_activity','activity_copies',
    'operator_correlation','graph_enclosures','activity_delay_seconds','activity_ROI_seconds','observer_scope'})

def signature(cfg):
    # Every unrecognized runtime flag enters the signature. A new optimization
    # cannot accidentally inherit a proof that did not exercise that flag.
    return {k:v for k,v in cfg.items() if k not in VARIABLE_FIELDS}

def save(cfg):
    assert cfg['batch_invariant'] is False and cfg['token_budget']==8192 and cfg['max_num_seqs']==1024
    assert cfg['gpus'] in ([0,1],[2,3],[4,5],[6,7])
    assert not cfg.get('controlled_pipeline') or not cfg.get('continuous_source_credit_limit',0), 'Do not mix controlled and continuous coordinator windows'
    if cfg.get('axis_kind')=='Token' and (not cfg.get('correctness_only') or cfg.get('suffix_by_request')):
        from .work_contract import validate
        validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
    cfg=dict(cfg,entry_module='poc_flashvep.dense_ar_longbench400_pd.serving')
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg['name']

def execute(name):
    preserve();assert dispatch_allowed()
    cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text());pair=cfg['gpus']
    if cfg.get('axis_kind')=='Token' and (not cfg.get('correctness_only') or cfg.get('suffix_by_request')):
        from .work_contract import validate
        validate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
    guardian=json.loads((ROOT/'state/guardian.json').read_text());assert guardian['status']=='ACTIVE'
    words=Path('/proc',str(guardian['pid']),'cmdline').read_bytes()
    assert b'poc_flashvep.dense_ar_output_aware_v2.bridge' in words and b'guardian' in words
    if not cfg.get('correctness_only'):
        proof=json.loads((ROOT/'state'/cfg['qualification_gate_v2']).read_text())
        assert proof['passed'] and proof['signature']==signature(cfg)
        assert proof['source_capsule_SHA256']==cfg['source_capsule_SHA256']
        if not cfg.get('controlled_pipeline'):assert proof['full400_passed'] and proof['quality_passed']
    folder=ROOT/'raw'/name
    if (folder/'dispatcher_result.json').exists():return json.loads((folder/'dispatcher_result.json').read_text())['complete']
    assert not folder.exists(),'Failed or interrupted evidence remains immutable; use a new version'
    folder.mkdir();write(folder/'configuration.json',cfg)
    dataset=ROOT/'data'/cfg['dataset'];assert digest(dataset)==cfg['dataset_SHA256']
    source=ROOT/'references/source_capsules'/cfg['source_capsule'];assert digest(source)==cfg['source_capsule_SHA256']
    archive=folder/'source.zip';shutil.copyfile(source,archive)
    frozen=folder/'worker_source';frozen.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in bundle.namelist())
        write(folder/'source_hashes.json',{n:hashlib.sha256(bundle.read(n)).hexdigest() for n in bundle.namelist()})
        bundle.extractall(frozen)
    write(folder/'input_provenance.json',dict(dataset_SHA256=digest(dataset),base=BASE,
        model_revision='b968826d9c46dd6066d109eabc6255188de91218',source_archive_SHA256=digest(archive)))
    env={k:v for k,v in os.environ.items() if not k.startswith(('DENSE_','DUETSERVE_'))}
    env.update(PYTHONPATH=str(frozen),PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'],
        VLLM_BATCH_INVARIANT='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',UCX_RCACHE_MAX_UNRELEASED='1024')
    if cfg.get('kv_layout'):env['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
    command=[str(PYTHON),'-u','-m',cfg['entry_module'],str(cfg_path)]
    owner=resources();active_gpus=pair[:1] if cfg.get('reference_single') else pair
    assert not [r for r in owner.processes() if r['gpu'] in active_gpus]
    if cfg.get('reference_single'):owner.restore(pair[1:])
    child=None;began=time.time();complete=False;leasepath=ROOT/'state'/f'pair_lease_{pair[0]}_{pair[1]}.json'
    try:
        with (ROOT/'logs'/(name+'.log')).open('w') as log:
            affinity=list(range(pair[0]//2*40,(pair[0]//2+1)*40))
            child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:os.sched_setaffinity(0,set(affinity)))
            write(folder/'command.json',dict(command=command,pid=child.pid,GPUs=pair,start_epoch=began,
                CPU_affinity=affinity,source_archive_SHA256=digest(archive),config_SHA256=digest(cfg_path)))
            write(leasepath,dict(name=name,pid=child.pid,GPUs=pair,status='RUNNING',time=time.time()))
            term=None
            while child.poll() is None:
                if time.time()-began>cfg.get('wall_timeout',4500) or not execution_allowed():
                    if term is None:os.killpg(child.pid,signal.SIGINT);term=time.time()
                    elif time.time()-term>75:os.killpg(child.pid,signal.SIGKILL)
                time.sleep(.5)
            status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            exits=json.loads((folder/'worker_exits.json').read_text()) if (folder/'worker_exits.json').exists() else {}
            complete=child.returncode==0 and status.get('status')=='COMPLETE' and exits.get('passed',False)
            write(folder/'dispatcher_result.json',dict(complete=complete,returncode=child.returncode,
                elapsed_seconds=time.time()-began,source_SHA256=digest(archive)))
            return complete
    finally:
        if child is not None and child.poll() is None:
            assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(leasepath,dict(name=name,GPUs=pair,status='RELEASED',time=time.time(),complete=complete))
        preserve();assert not [r for r in owner.processes() if r['gpu'] in active_gpus]

def lane(plan_name,lane_index):
    initialize();plan=json.loads((ROOT/'jobs'/plan_name).read_text());jobs=plan['lanes'][lane_index]
    pair=[2*lane_index,2*lane_index+1];owner=resources();results=[]
    if not jobs:
        write(ROOT/'state'/f'{Path(plan_name).stem}_lane{lane_index}_complete.json',dict(time=time.time(),results=[],complete=True))
        return
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        while not (ROOT/'state/guardian.json').exists():
            assert dispatch_allowed();time.sleep(5)
        owner.stop(pair)
        try:
            from .bridge import bind
            bind()
            from ..dense_ar_longbench400_pd.metrics import summarize
            for name in jobs:
                if not dispatch_allowed():
                    results.append(dict(name=name,status='NOT_RUN',reason='TIME_BUDGET_OR_SAFETY_STOP'));continue
                try:
                    owner.stop(pair)
                    ok=execute(name);metric=summarize(name) if ok else None
                    result=dict(name=name,status='COMPLETE' if ok else 'FAILED',
                        metrics={k:metric[k] for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','output_tokens')} if metric else None)
                except Exception as error:
                    result=dict(name=name,status='FAILED',error=repr(error))
                results.append(result)
                write(ROOT/'state'/f'{Path(plan_name).stem}_lane{lane_index}_progress.json',dict(time=time.time(),results=results))
                print('V2_JOB_RESULT',json.dumps(result),flush=True)
                if result['status']=='FAILED' and plan.get('stop_on_failure',True):break
        finally:
            owner.restore(pair);preserve()
    write(ROOT/'state'/f'{Path(plan_name).stem}_lane{lane_index}_complete.json',dict(time=time.time(),results=results,complete=len(results)==len(jobs)))

if __name__=='__main__':
    import sys
    lane(sys.argv[1],int(sys.argv[2]))
