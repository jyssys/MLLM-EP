"""Freeze each job,lease one GPU pair,and preserve every failure."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import zipfile
from .common import ROOT,REPO,SOURCE,PYTHON,BASE,write,digest,initialize,preserve,resources,dispatch_allowed

_SAVE_INITIALIZED=False


def save(cfg):
    global _SAVE_INITIALIZED
    if not _SAVE_INITIALIZED:
        initialize();_SAVE_INITIALIZED=True
    # Configuration generation writes only this namespace. Each actual dispatch
    # still checks the user checkout and index before and after GPU execution.
    assert cfg['batch_invariant'] is False
    assert cfg['method'] in ('Agg','Native','Layer','Token','Hybrid')
    assert cfg['token_budget']==8192 and cfg['max_num_seqs']==1024
    cfg=dict(cfg,entry_module='poc_flashvep.dense_ar_longbench400_pd.serving')
    path=ROOT/'configs'/(cfg['name']+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:write(path,cfg)
    return cfg['name']


def require_role_compile_gate(cfg):
    if cfg.get('preserve_compiled_decode_only') and not cfg.get('correctness_only'):
        if cfg.get('D_partial_IR_norm_priority'):
            from .scoped_decode_compile_followup import signature as role_signature
        elif cfg.get('D_mode0_IR_norm_priority'):
            from .decode_ir_priority_followup import signature as role_signature
        else:
            from .decode_only_compile_followup import role_signature
        role_proof=json.loads((ROOT/'state'/cfg['D_compile_role_gate']).read_text())
        assert role_proof['passed'] and role_proof['full400_passed']
        assert role_proof['role_signature']==role_signature(cfg)
        assert role_proof['source_capsule_SHA256']==cfg['source_capsule_SHA256']


def execute(name):
    initialize();preserve()
    cfg_path=ROOT/'configs'/(name+'.json');cfg=json.loads(cfg_path.read_text())
    pair=cfg['gpus'];assert pair in ([0,1],[2,3],[4,5],[6,7])
    budget=json.loads((ROOT/'state/budget.json').read_text())
    assert dispatch_allowed(budget)
    assert not (ROOT/'state/STOP_DISPATCH.json').exists()
    guardian=json.loads((ROOT/'state/guardian.json').read_text())
    assert guardian['status']=='ACTIVE'
    words=Path('/proc',str(guardian['pid']),'cmdline').read_bytes()
    assert b'poc_flashvep.dense_ar_longbench400_pd.guardian' in words
    gate=None
    require_role_compile_gate(cfg)
    if cfg.get('defer_D_request_registration') and not cfg.get('correctness_only'):
        from .registration_deferral_campaign import registration_signature
        proof=json.loads((ROOT/'state'/cfg['registration_deferral_gate']).read_text())
        assert proof['passed'] and proof['full400_passed']
        assert proof['signature']==registration_signature(cfg)
        assert proof['source_capsule_SHA256']==cfg['source_capsule_SHA256']
    if cfg.get('D_prefill_completion_guard') and not cfg.get('correctness_only'):
        completion=json.loads((ROOT/'state'/cfg['completion_guard_qualification_gate']).read_text())
        assert completion['passed'] and completion['full400_passed']
        assert completion['axis']==cfg.get('axis_kind','Native') and completion['cut']==cfg.get('axis_cut',0)
    if cfg.get('source_credit_qualification_gate') and not cfg.get('correctness_only'):
        credit=json.loads((ROOT/'state'/cfg['source_credit_qualification_gate']).read_text())
        assert credit['passed'] and credit['full400_passed']
        assert credit['credit']==cfg['continuous_source_credit_limit']
        assert credit['hidden_slots']==cfg['hidden_source_slots']==cfg['hidden_destination_slots']
    if cfg.get('final_composition_gate') and not cfg.get('correctness_only'):
        from .final_confirmation_jobs import signature
        final=json.loads((ROOT/'state'/cfg['final_composition_gate']).read_text())
        assert final['passed'] and final['full400_passed']
        assert final['signature']==signature(cfg)
        assert final['source_capsule_SHA256']==cfg['source_capsule_SHA256']
    if cfg.get('fractional_token_work') and not cfg.get('correctness_only'):
        fractional=json.loads((ROOT/'state'/cfg['fractional_token_gate']).read_text())
        assert fractional['passed'] and fractional['full400_passed']
        assert fractional['effective_cut']==cfg['fractional_token_work']
    if cfg.get('D_running_admission_cap') and not cfg.get('correctness_only'):
        cap_gate=json.loads((ROOT/'state'/cfg['D_cap_qualification_gate']).read_text())
        assert cap_gate['passed'] and cap_gate['full400_passed']
        assert cap_gate['cap']==cfg['D_running_admission_cap']
    if cfg.get('defer_D_remote_setup') and not cfg.get('correctness_only'):
        setup=json.loads((ROOT/'state/D_remote_setup_deferral_gate_v1.json').read_text())
        assert setup['passed'] and setup['full400_passed'], 'Remote-setup deferral needs independent state and lifetime validation'
        if cfg.get('remote_family_gate'):
            family=json.loads((ROOT/'state'/cfg['remote_family_gate']).read_text())
            assert family['passed'] and family['full400_passed']
            assert family['action']['axis']==cfg['axis_kind'] and family['action']['cut']==cfg['axis_cut']
            if cfg['axis_kind']=='Hybrid':assert family['action']['fraction']==cfg['hybrid_suffix_fraction']
    if cfg.get('whole_prefill_deflection') and not cfg.get('correctness_only'):
        filename='whole_prefill_deflection_continuous_gate_v2.json' if cfg.get('d_compute_throttle') else 'whole_prefill_deflection_gate_v1.json'
        deflect=json.loads((ROOT/'state'/filename).read_text())
        assert deflect['passed'] and deflect['full400_passed'], 'Whole-request deflection needs its own full-state validation'
    if cfg.get('d_compute_throttle') and not cfg.get('correctness_only'):
        throttle=json.loads((ROOT/'state/D_compute_throttle_gate_v1.json').read_text())
        assert throttle['passed'] and cfg.get('axis_kind','Native') in throttle['qualified_axes']
        assert throttle['full400_passed'], 'Compute deferral needs its separate full400 validation'
        if cfg.get('compute_family_gate'):
            family=json.loads((ROOT/'state'/cfg['compute_family_gate']).read_text())
            assert family['passed'] and family['full400_passed'], 'This action needs its own compute-policy stress proof'
            assert family['action']['axis']==cfg['axis_kind'] and family['action']['cut']==cfg['axis_cut']
            if cfg['axis_kind']=='Hybrid':assert family['action']['fraction']==cfg['hybrid_suffix_fraction']
        if cfg.get('token_stock_model') and cfg.get('continuous_source_credit_limit',0)==0:
            unbounded=json.loads((ROOT/'state/D_compute_throttle_StockToken_uncapped400_gate_v1.json').read_text())
            assert unbounded['passed'], 'Uncapped stock-token retention needs its own full400 stress proof'
    if cfg.get('d_quiescent_admission') and not cfg.get('correctness_only'):
        admission=json.loads((ROOT/'state/D_quiescent_admission_gate_v1.json').read_text())
        assert admission['passed'] and cfg.get('axis_kind','Native') in admission['qualified_axes']
        assert admission['full400_passed'], 'The admission intervention needs its separate full400 gate'
    if cfg.get('axis_kind') and not cfg.get('correctness_only'):
        gate=json.loads((ROOT/'state'/cfg['qualification_gate']).read_text())
        if cfg.get('fuse_short_hybrid'):
            assert gate.get('fused_short_hybrid'), 'The fused query order needs its own teacher and full400 proof'
        if cfg.get('preserve_compiled_full_model'):
            assert gate.get('preserved_compiled_full_model'), 'The preserved-model adapter needs a separate gate'
        assert gate['passed'] and cfg['axis_kind'] in gate['qualified_axes']
        signatures=gate.get('qualified_actions')
        if signatures is None:
            signatures=[]
            for result in gate.get('results',[]):
                assert result['passed']
                proven=json.loads((ROOT/'raw'/result['actual']/'configuration.json').read_text())
                signatures.append(dict(axis=proven['axis_kind'],cut=proven['axis_cut'],
                    suffix_fraction=proven.get('hybrid_suffix_fraction') if proven['axis_kind']=='Hybrid' else None))
        action=dict(axis=cfg['axis_kind'],cut=cfg['axis_cut'],
            suffix_fraction=cfg.get('hybrid_suffix_fraction') if cfg['axis_kind']=='Hybrid' else None)
        assert action in signatures,('The exact action family needs its own correctness proof',action,signatures)
        for requested in cfg.get('axis_actions_by_request',{}).values():
            actual=dict(axis=requested['kind'],cut=requested['cut'],
                suffix_fraction=requested.get('suffix_fraction') if requested['kind']=='Hybrid' else None)
            assert actual in signatures,('An individual action lacks its own proof',actual)
        if cfg.get('token_stock_model'):
            assert gate.get('stock_token_model'), 'Stock-token needs its separate gate'
            if cfg.get('execution_family')=='MATCHED_NO_COMPILE':
                assert cfg.get('controlled_pipeline') and cfg['axis_cut'] in gate.get('mode0_controlled_cuts',[])
            if not cfg.get('controlled_pipeline'):assert gate.get('full400_runtime_stress_passed'), 'Full400 stock-token runtime stress is required'
        if not cfg.get('controlled_pipeline'):assert gate.get('mixed_rows_passed'), 'Continuous partitioning requires its own mixed-row gate'
    folder=ROOT/'raw'/name
    if (folder/'dispatcher_result.json').exists():return json.loads((folder/'dispatcher_result.json').read_text())['complete']
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert dispatch_allowed(budget)
        assert not folder.exists(),'Preserve failed evidence. Use a new job version.'
        folder.mkdir();write(folder/'configuration.json',cfg)
        if gate is not None:write(folder/'qualification_gate_used.json',gate)
        dataset=ROOT/'data'/cfg['dataset'];assert digest(dataset)==cfg['dataset_SHA256']
        archive=folder/'source.zip'
        packages=('dense_ar_longbench400_pd','dense_ar_bi0_shortanswer_pd','dense_ar_slo_layer_pd')
        files=[REPO/'poc_flashvep/__init__.py']
        for package in packages:files.extend(p for p in (REPO/'poc_flashvep'/package).glob('*') if p.suffix in ('.py','.cc','.cu','.h'))
        if cfg.get('source_capsule'):
            capsule=ROOT/'references/source_capsules'/cfg['source_capsule']
            assert digest(capsule)==cfg['source_capsule_SHA256']
            shutil.copyfile(capsule,archive)
        else:
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in files:bundle.write(path,str(path.relative_to(REPO)))
        frozen=folder/'worker_source';frozen.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            hashes={item:hashlib.sha256(bundle.read(item)).hexdigest() for item in bundle.namelist()}
            bundle.extractall(frozen)
        write(folder/'source_hashes.json',hashes)
        write(folder/'input_provenance.json',dict(dataset_SHA256=digest(dataset),model_revision='b968826d9c46dd6066d109eabc6255188de91218',source_archive_SHA256=digest(archive),base=BASE))
        env={key:value for key,value in os.environ.items() if not key.startswith(('DENSE_','DUETSERVE_'))}
        env.update(PYTHONPATH=str(frozen),PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'],
            VLLM_BATCH_INVARIANT='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            VLLM_NO_USAGE_STATS='1',UCX_RCACHE_MAX_UNRELEASED='1024')
        if cfg.get('kv_layout'):env['VLLM_KV_CACHE_LAYOUT']=cfg['kv_layout']
        owner=resources();owner.stop(pair)
        active_GPUs=pair[:1] if cfg.get('reference_single') else pair
        if cfg.get('reference_single'):owner.restore(pair[1:])
        command=[str(PYTHON),'-u','-m',cfg['entry_module'],str(cfg_path)]
        child=None;started=time.time();result=False
        state_path=ROOT/'state'/f'pair_lease_{pair[0]}_{pair[1]}.json'
        try:
            with (ROOT/'logs'/(name+'.log')).open('w') as log:
                affinity=set(range((pair[0]//2)*40,(pair[0]//2+1)*40))
                child=subprocess.Popen(command,cwd=frozen,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,affinity))
                write(folder/'command.json',dict(command=command,pid=child.pid,GPUs=pair,
                    CPU_affinity=sorted(affinity),source_archive_SHA256=digest(archive),
                    config_SHA256=digest(cfg_path),start_epoch=started))
                write(state_path,dict(name=name,pid=child.pid,GPUs=pair,status='RUNNING',time=time.time()))
                term=None
                while child.poll() is None:
                    if time.time()-started>cfg.get('wall_timeout',1800) or (ROOT/'state/STOP_DISPATCH.json').exists():
                        if term is None:
                            os.killpg(child.pid,signal.SIGINT);term=time.time()
                        elif time.time()-term>75:os.killpg(child.pid,signal.SIGKILL)
                    time.sleep(.5)
                status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                exits=json.loads((folder/'worker_exits.json').read_text()) if (folder/'worker_exits.json').exists() else {}
                result=child.returncode==0 and status.get('status')=='COMPLETE' and exits.get('passed',False)
                write(folder/'dispatcher_result.json',dict(complete=result,returncode=child.returncode,
                    elapsed_seconds=time.time()-started,source_SHA256=digest(archive)))
                return result
        finally:
            if child is not None and child.poll() is None:
                assert os.getsid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            write(state_path,dict(name=name,GPUs=pair,status='RELEASED',time=time.time(),complete=result))
            preserve();assert not [row for row in owner.processes() if row['gpu'] in active_GPUs]
            owner.restore(active_GPUs)


if __name__=='__main__':
    import sys
    raise SystemExit(0 if execute(sys.argv[1]) else 1)
