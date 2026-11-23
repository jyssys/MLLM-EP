"""Qualify a warm-compiler common policy before bounded independent C3 timing."""
import fcntl
import json
import re
import os
import signal
from pathlib import Path
from functools import lru_cache
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,preserve,resources,dispatch_allowed

QUAL='common_warm_regime_qualification_v8.json'
PLAN='coordinator_hygiene_c3_confirmation_warm_regime_v8.json'
VERSION=8
LANES={0:['NativeStrong','NativeCommon2','StockToken'],1:['Hybrid','HybridIso8Approx','HybridGCControl'],3:['Layer','Layer8']}
MAPPING={'NativeStrong':'NativeStrong','NativeCommon2':'NativeCommon2','StockToken':'StockToken',
    'OriginalLayer':'Layer','OptimizedLayer':'Layer8','OriginalHybrid':'Hybrid','OptimizedHybrid':'HybridIso8Approx',
    'HybridGCControl':'HybridGCControl'}


def gate(label):return 'pc_warm_regime_'+label.lower()+f'_composition_v{9 if VERSION==10 else VERSION}.json'


@lru_cache(maxsize=1)
def signature_reader():
    from .common import bind
    return bind().signature


def priming_identity(cfg):
    signature=dict(signature_reader()(cfg))
    for key in ('official_compiler_cache_regime','defer_coordinator_cyclic_gc'):signature.pop(key,None)
    return json.dumps(dict(model_contract=signature,source_SHA256=cfg['source_capsule_SHA256'],pair=cfg['gpus']),sort_keys=True)


def serving_prime_template(parent,pair,name):
    """Prime the real continuous policy,not its different strict-test pipeline."""
    cfg=json.loads((ROOT/'configs'/f'PC_WarmQual_{parent}_Natural400.json').read_text())
    control=json.loads((ROOT/'configs'/f'PC_WarmQual_{parent}_Prime8.json').read_text())
    cfg.update(name=name,gpus=pair,dataset=control['dataset'],dataset_SHA256=control['dataset_SHA256'],count=8,
        serial=True,ignore_eos=True,output_cap=16,arrival_offsets=[0.]*8,correctness_only=True,
        state_capture=not bool(cfg.get('axis_kind')),axis_state_capture=bool(cfg.get('axis_kind')),
        gpu_state_digest=True,axis_capture_hidden_last_only=True,
        official_compiler_cache_regime='OFFICIAL_CACHE_PRIMING_ONLY',
        scope='Eight-request correctness-only cache priming under the unchanged real continuous pipeline and model contract. '
            'Serial arrivals do not replace controlled_pipeline=False. No performance or new independent serving start is claimed.')
    if cfg.get('axis_kind')=='Token':
        from ..dense_ar_output_aware_v2.work_contract import regenerate
        regenerate(cfg,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
    return cfg


def compiler_regions(log):
    markers=list(re.finditer(r'Using cache directory: (.*?) for vLLM',log));rows=[]
    for index,marker in enumerate(markers):
        section=log[marker.end():markers[index+1].start() if index+1<len(markers) else len(log)]
        root=marker[1]
        rows.append(dict(cache_root=root,official_AOT_required='/torch_compile_cache/' in root,
            graph_reused='Directly load the compiled graph(s)' in section,
            AOT_reused='Directly load AOT compilation from path ' in section,
            new_compilation='Compiling a graph for compile range ' in section))
    return rows


def expected_roots(name):
    roots={}
    for role in ('prefill','decode'):
        log=(ROOT/'raw'/name/(role+'.log')).read_text()
        found=[row['cache_root'] for row in compiler_regions(log)]
        if found:roots[role]=list(dict.fromkeys(found))
    return roots


def check_reuse(name,roots):
    observed={}
    for role,expected in roots.items():
        rows=compiler_regions((ROOT/'raw'/name/(role+'.log')).read_text())
        assert set(row['cache_root'] for row in rows)==set(expected),'A full or partial compiler cache identity changed'
        assert all(row['graph_reused'] and not row['new_compilation'] for row in rows),'A compiled stage was not warm-loaded'
        assert all(not row['official_AOT_required'] or row['AOT_reused'] for row in rows),'A full-model stage did not load official AOT'
        observed[role]=rows
    return observed


def prepare():
    from .stable_source import install
    runner=install();preserve();items=[]
    for lane,labels in LANES.items():
        pair=[2*lane,2*lane+1]
        for label in labels:
            jobs=[]
            for count in (32,400):
                cfg=json.loads((ROOT/'configs'/f'PC_PortQual_{label}_Natural{count}.json').read_text())
                name=f'PC_WarmQual_{label}_Natural{count}'
                cfg.update(name=name,gpus=pair,official_compiler_cache_regime='PREEXISTING_VERIFIED_AOT')
                runner.save(cfg);jobs.append(name)
            final=cfg;strict=[]
            if label not in ('NativeStrong','NativeCommon2'):
                source_label='Hybrid' if label=='HybridGCControl' else label
                for suffix in ('NormalGC','DeferredGC'):
                    parent_name=(f'PC_FixedPort_StockToken_{suffix}_Strict8' if label=='StockToken' else
                        f'PC_PortQual_{source_label}_{suffix}_Strict8')
                    cfg=json.loads((ROOT/'configs'/(parent_name+'.json')).read_text())
                    name=f'PC_WarmQual_{label}_{suffix}_Strict8'
                    cfg.update(name=name,gpus=pair,official_compiler_cache_regime='PREEXISTING_VERIFIED_AOT')
                    runner.save(cfg);strict.append(name)
                cfg=json.loads((ROOT/'configs'/(strict[0]+'.json')).read_text())
            else:cfg=json.loads((ROOT/'configs'/f'PC_PortQual_{label}_State8.json').read_text())
            prime=f'PC_WarmQual_{label}_Prime8'
            cfg.update(name=prime,gpus=pair,official_compiler_cache_regime='OFFICIAL_CACHE_PRIMING_ONLY')
            runner.save(cfg)
            items.append(dict(label=label,lane=lane,pair=pair,prime=prime,strict=strict,jobs=jobs,
                gate=gate(label),signature=runner.signature(final),source_capsule_SHA256=final['source_capsule_SHA256']))
    target=ROOT/'configs'/QUAL;assert not target.exists()
    write(target,dict(time=time.time(),items=items,
        prerequisite='official_warm_cache_stock_strict_complete_v7.json',
        scope='A new explicit warm-compiler regime. Preserve all failed cold/warm controls. '
            'Every arm gets a fresh8-request priming/state control,an exact warm pair for axis methods,and Natural32/400 quality/lifetime gates. '
            'Native uses full36 valid-state and exact imported KV controls. No universal BI0 bitwise claim is introduced. '
            'Use the same fixed NIXL address policy,stable source paths,evidence-write buffering and final coordinator GC policy. '
            'HybridGCControl alone keeps normal coordinator GC. Workers,math,stock compilation,dataset and SLO stay frozen. '
            'Eight arms cover the original required baselines,geometry candidates and one-factor GC control. '
            'Optional common-source16 controls are not expanded under the remaining budget. Their earlier outcomes remain preserved.'))
    children=[]
    for lane in LANES:
        log=ROOT/'logs'/f'common_warm_regime_qualification_lane{lane}_v8.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','qualify',str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=lane,pid=child.pid))
    log=ROOT/'logs/common_warm_regime_confirmation_v8.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','confirm'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/common_warm_regime_controllers_v8.json',dict(time=time.time(),children=children,confirmation_pid=child.pid))
    return children


def qualify(lane):
    from .stable_source import install
    runner=install()
    from .port_qualification import native_state
    from .coordinator_state_fix import compare
    from ..dense_ar_longbench400_pd.qualification import audit
    from .quality import compare as quality
    plan=json.loads((ROOT/'configs'/QUAL).read_text());items=[item for item in plan['items'] if item['lane']==lane]
    if VERSION==9:
        previous=json.loads((ROOT/'state/common_warm_regime_controllers_v8.json').read_text())
        pid=next(row['pid'] for row in previous['children'] if row['lane']==lane)
        while True:
            try:words=Path('/proc',str(pid),'cmdline').read_bytes()
            except OSError:break
            if b'poc_flashvep.dense_ar_p_critical_path.warm_serving_followup' not in words:break
            if (ROOT/'state'/f'common_warm_regime_qualification_lane{lane}_complete_v8.json').exists():break
            if not dispatch_allowed():break
            time.sleep(10)
    prerequisite=ROOT/'state'/plan['prerequisite'];pair=[2*lane,2*lane+1];owner=resources();results=[]
    while not prerequisite.exists():
        if not dispatch_allowed():break
        time.sleep(10)
    baseline_ok=prerequisite.exists() and json.loads(prerequisite.read_text())['passed']
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for item in items:
                proof=dict(time=time.time(),passed=False,signature=item['signature'],source_capsule_SHA256=item['source_capsule_SHA256'],
                    evidence='CORRECTNESS_ONLY',stages=[],prime=None,strict=None)
                try:
                    assert baseline_ok,'The prospective warm Stock triple failed or did not complete within the dispatch budget'
                    def execute(name):
                        assert dispatch_allowed(),'Dispatch budget ended before the next predeclared control'
                        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
                        if receipt.exists():
                            assert VERSION==9 and json.loads(receipt.read_text())['complete'],'Do not retry a failed GPU job'
                            command=json.loads((folder/'command.json').read_text())
                            assert command['config_SHA256']==digest(ROOT/'configs'/(name+'.json'))
                            assert json.loads(receipt.read_text())['source_SHA256']==json.loads((ROOT/'configs'/(name+'.json')).read_text())['source_capsule_SHA256']
                            proof.setdefault('reused_completed_GPU_jobs',[]).append(name)
                        else:
                            assert not folder.exists(),'Inspect an interrupted or live GPU handle instead of restarting it'
                            owner.stop(pair);assert runner.execute(name),name
                        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                        state=audit(name) if cfg.get('axis_kind') else native_state(name)
                        assert state['passed'],name
                        if cfg.get('axis_kind'):
                            assert state['decode_full36_observation_complete'],name
                            assert all(row['status']=='PASS' for row in state['imported_prefix_KV']),name
                        return state
                    state=execute(item['prime']);roots=expected_roots(item['prime'])
                    if item['label'] in ('NativeStrong','NativeCommon2','StockToken'):
                        assert set(roots)=={'prefill','decode'},'Missing official compiled-backbone cache identity'
                    proof['prime']=dict(name=item['prime'],state=state,official_cache_roots=roots)
                    for name in item['strict']:
                        execute(name);check_reuse(name,roots)
                    if item['strict']:
                        proof['strict']=compare(*item['strict']);assert proof['strict']['passed'],proof['strict']
                    for name in item['jobs']:
                        state=execute(name);reuse=check_reuse(name,roots);score=quality(name);assert score['passed'],score
                        proof['stages'].append(dict(name=name,state=state,quality=score,cache_reuse=reuse))
                    proof.update(passed=True,full400_passed=True,quality_passed=True)
                except Exception as error:
                    import traceback
                    proof.update(error=repr(error),traceback=traceback.format_exc(),evidence='FAILED')
                write(ROOT/'state'/item['gate'],proof)
                results.append(dict(label=item['label'],passed=proof['passed'],error=proof.get('error')))
                write(ROOT/'state'/f'common_warm_regime_qualification_lane{lane}_progress_v{VERSION}.json',dict(time=time.time(),results=results))
                print('WARM_REGIME_COMPOSITION',results[-1],flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'common_warm_regime_qualification_lane{lane}_complete_v{VERSION}.json',dict(time=time.time(),results=results,complete=True))


def confirm():
    from .stable_source import install
    runner=install();plan=json.loads((ROOT/'configs'/QUAL).read_text());proofs={}
    for item in plan['items']:
        path=ROOT/'state'/item['gate']
        while not path.exists():
            if not dispatch_allowed():
                write(ROOT/'state'/f'coordinator_hygiene_warm_regime_not_run_v{VERSION}.json',dict(time=time.time(),status='NOT_RUN',reason='Qualification did not finish within the dispatch budget.'));return
            time.sleep(10)
        proofs[item['label']]=json.loads(path.read_text())
    invalid={label:proof for label,proof in proofs.items() if not proof['passed']}
    if any(label in invalid for label in ('NativeStrong','NativeCommon2','StockToken')):
        write(ROOT/'state'/f'coordinator_hygiene_warm_regime_not_run_v{VERSION}.json',dict(time=time.time(),status='NOT_RUN',
            reason='A required strong baseline failed the new warm-regime qualification. Keep the baseline and the failed gate.',invalid=invalid));return
    # A complete N3 block needs enough budget. Do not start a known partial grid.
    budget=json.loads((ROOT/'state/budget.json').read_text())
    if time.time()+100*60>budget['deadline_epoch']:
        write(ROOT/'state'/f'coordinator_hygiene_warm_regime_not_run_v{VERSION}.json',dict(time=time.time(),status='NOT_RUN',
            reason='Less than100minutes remain for three parallel independent blocks and closeout. No partial N3 timing is launched.',invalid=invalid));return
    data_name='balanced400_v2_heldout.json';data=json.loads((ROOT/'data'/data_name).read_text());count=len(data)
    lengths=json.loads((ROOT/'configs/PC_C3_EqualReplay_s0_OriginalLayer.json').read_text())['replay_output_lengths']
    lanes=[[],[],[],[]];blocks=[];preludes=[];prime_refs={}
    stage_lanes={mode:[[],[],[],[]] for mode in ('EqualReplay','Natural')}
    available_primes={}
    if VERSION==10:
        for item in plan['items']:
            if not proofs[item['label']]['passed']:continue
            candidate=json.loads((ROOT/'configs'/(item['prime']+'.json')).read_text())
            available_primes[priming_identity(candidate)]=item['prime']
    for index,lane in enumerate((0,1,3)):
        pair=[2*lane,2*lane+1];seed=2026112451+index
        for label,parent in MAPPING.items():
            if parent in invalid:continue
            name=f'PC_WarmC3V{VERSION}_Prime_s{index}_{label}'
            template=serving_prime_template(parent,pair,name) if VERSION==10 else json.loads((ROOT/'configs'/f'PC_WarmQual_{parent}_Prime8.json').read_text())
            template.update(name=name,gpus=pair)
            identity=priming_identity(template)
            if VERSION==10 and identity in available_primes:
                prime_refs[index,label]=available_primes[identity]
            else:
                runner.save(template);lanes[lane].append(name);stage_lanes['EqualReplay'][lane].append(name);preludes.append(name)
                prime_refs[index,label]=name;available_primes[identity]=name
        for mode in ('EqualReplay','Natural'):
            labels=[label for label in MAPPING if mode!='Natural' or label!='HybridGCControl'];names={}
            for label in labels:
                parent=MAPPING[label];cfg=json.loads((ROOT/'configs'/f'PC_WarmQual_{parent}_Natural400.json').read_text())
                name=f'PC_WarmC3V{VERSION}_{mode}_s{index}_{label}'
                cfg.update(name=name,gpus=pair,dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),count=count,
                    correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                    qps=2.,seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist(),
                    ignore_eos=mode=='EqualReplay',output_cap=2048,confirmation=True,qualification_gate_v2=gate(parent))
                if VERSION==10:cfg['cache_priming_reference']=prime_refs.get((index,label))
                if mode=='EqualReplay':cfg.update(replay_output_lengths=lengths,diagnostic_controlled_output_replay=True)
                else:cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None)
                if cfg.get('axis_kind')=='Token':
                    from ..dense_ar_output_aware_v2.work_contract import regenerate
                    regenerate(cfg,data)
                assert runner.signature(cfg)==proofs[parent]['signature'],label
                runner.save(cfg);names[label]=name
            order=list(labels)
            if index==1:order.reverse()
            elif index==2:order=order[2:]+order[:2]
            lanes[lane].extend(names[label] for label in order if MAPPING[label] not in invalid)
            stage_lanes[mode][lane].extend(names[label] for label in order if MAPPING[label] not in invalid)
            blocks.append(dict(mode=mode,start=index,seed=seed,qps=2.,pair=pair,arms=names,order=order))
    target=ROOT/'jobs'/PLAN;assert not target.exists()
    main=dict(time=time.time(),blocks=blocks,arms=list(MAPPING),lanes=lanes,stop_on_failure=True,
        invalid_compositions=invalid,correctness_only_pair_cache_priming=preludes,
        extra_pairs=[['OriginalHybrid','HybridGCControl']],analysis_scope=
            'Fresh warm-regime common coordinator family. All arms retain their strong original model paths. '
            'Frozen addresses,stable worker paths and evidence-write buffering apply to every arm. '
            'HybridGCControl changes only coordinator cyclic GC relative to OriginalHybrid. '
            'Three independent starts use three verified H100/NV18 pairs. Each paired comparison stays on the same pair and arrival trace. '
            'Do not pool hardware-pair starts with earlier fixed-pair families or claim shared-host interference was excluded. '
            'Each physical pair receives a separate correctness-only official-cache priming control before timing. '
            'Actual graph/AOT reuse must still be audited in the timed worker logs. '
            'Natural EOS has seven arms. Equal Replay includes the eighth one-factor GC control. '
            'No safe-window predictor,active-decode insertion,changed output contract,or sustainable-capacity claim.')
    if VERSION==10:
        main.update(execution_stage_plans=['warm_regime_v10_equal_stage.json','warm_regime_v10_natural_stage.json'],
            stage_scope='Equal-output N3 has priority. Launch the entire Natural N3 stage only if an empirical runtime budget fits. '
                'The decision uses elapsed runtime and correctness,not whether an axis wins. '
                'Reuse a qualification prime only on the identical physical pair,worker source and model contract. '
                'Coordinator GC and the priming-status label do not change the model cache identity. '
                'Every timed log must still show warm reuse of all expected full and partial stages.')
        main['stage_lanes']=stage_lanes
    write(target,main)
    from .bootstrap import launch
    if VERSION==10:run_stages(main)
    else:print('WARM_REGIME_C3_LAUNCHED',launch(PLAN),flush=True)


def run_stages(main):
    from .bootstrap import launch
    equal,natural=main['execution_stage_plans']
    for name,mode in ((equal,'EqualReplay'),(natural,'Natural')):
        write(ROOT/'jobs'/name,dict(time=time.time(),lanes=main['stage_lanes'][mode],stop_on_failure=True,
            main_plan=PLAN,main_plan_SHA256=digest(ROOT/'jobs'/PLAN),stage=mode))
    children=launch(equal);state=ROOT/'state/warm_regime_v10_stage_decision.json'
    write(state,dict(time=time.time(),status='EQUAL_RUNNING',children=children))
    while True:
        terminal=[ROOT/'state'/(Path(equal).stem+f'_lane{row["lane"]}_complete.json') for row in children]
        if all(path.exists() for path in terminal):break
        for row,path in zip(children,terminal):
            if path.exists():continue
            try:words=Path('/proc',str(row['pid']),'cmdline').read_bytes()
            except OSError:words=b''
            if equal.encode() not in words:
                write(ROOT/'state/coordinator_hygiene_warm_regime_not_run_v10.json',dict(time=time.time(),status='NOT_RUN',modes=['Natural'],
                    reason='An Equal-stage process handle disappeared before its terminal record. Do not restart it.'))
                return
        time.sleep(10)
    failed=[row for path in terminal for row in json.loads(path.read_text())['results'] if row['status']!='COMPLETE']
    if failed:
        write(ROOT/'state/coordinator_hygiene_warm_regime_not_run_v10.json',dict(time=time.time(),status='NOT_RUN',modes=['Natural'],
            reason='The complete Equal N3 stage did not execute. Preserve failures and do not launch a partial followup.',failed=failed));return
    estimates=[]
    for block in main['blocks']:
        if block['mode']!='Natural':continue
        equal_block=next(row for row in main['blocks'] if row['mode']=='EqualReplay' and row['start']==block['start'])
        seconds=sum(json.loads((ROOT/'raw'/equal_block['arms'][label]/'dispatcher_result.json').read_text())['elapsed_seconds']
            for label in block['arms'] if label in equal_block['arms'] and (ROOT/'raw'/equal_block['arms'][label]/'dispatcher_result.json').exists())
        estimates.append(1.15*seconds)
    budget=json.loads((ROOT/'state/budget.json').read_text());needed=max(estimates)
    decision=dict(time=time.time(),status='NATURAL_LAUNCH' if time.time()+needed<=budget['last_large_dispatch_epoch'] else 'NATURAL_NOT_RUN_BY_BUDGET',
        estimated_lane_seconds=estimates,margin_fraction=.15,last_large_dispatch_epoch=budget['last_large_dispatch_epoch'],
        scope='Use only observed Equal-stage process duration with15percent margin. Do not select a mode from a favorable TTFT or goodput result.')
    write(state,decision)
    if decision['status']!='NATURAL_LAUNCH':
        write(ROOT/'state/coordinator_hygiene_warm_regime_not_run_v10.json',dict(time=time.time(),status='NOT_RUN',modes=['Natural'],
            reason='The entire Natural N3 stage does not fit the remaining frozen dispatch window.',decision=decision));return
    decision['children']=launch(natural);write(state,decision)


def timed_regime_check(name,label,start,cfg):
    match=re.match(r'PC_WarmC3(?:V\d+)?_',name);assert match
    prime=cfg.get('cache_priming_reference') or match[0]+f'Prime_s{start}_{label}'
    try:
        prime_cfg=json.loads((ROOT/'configs'/(prime+'.json')).read_text())
        assert priming_identity(prime_cfg)==priming_identity(cfg),'Priming reference belongs to a different pair,source,or model contract'
        roots=expected_roots(prime)
        if cfg.get('axis_kind') in (None,'Token'):
            assert set(roots)=={'prefill','decode'},'A compiled Native/Stock role has no recorded official cache root'
        observed=check_reuse(name,roots)
        return dict(passed=True,prime=prime,observed=observed,
            scope='Observed official backbone caches were warm-loaded. Partial-model segment compilation is a separate unchanged worker path.')
    except Exception as error:
        return dict(passed=False,prime=prime,error=repr(error),
            scope='Keep measured metrics diagnostic. The required warm-compiler execution regime was not verified.')


def prepare_reader_recovery():
    from .stable_source import install
    install();preserve()
    old=ROOT/'configs/common_warm_regime_qualification_v8.json'
    assert not (ROOT/'jobs/coordinator_hygiene_c3_confirmation_warm_regime_v8.json').exists()
    controller=json.loads((ROOT/'state/common_warm_regime_controllers_v8.json').read_text())['confirmation_pid']
    proc=Path('/proc',str(controller));assert proc.stat().st_uid==os.getuid()
    words=(proc/'cmdline').read_bytes()
    assert b'poc_flashvep.dense_ar_p_critical_path.warm_serving_followup' in words and b'confirm' in words
    assert all(not (task/'children').read_text().strip() for task in (proc/'task').iterdir()),'Only a childless CPU waiter can stop here'
    os.kill(controller,signal.SIGTERM)
    write(ROOT/'state/coordinator_hygiene_warm_regime_not_run_v8.json',dict(time=time.time(),status='NOT_RUN',
        reason='The CPU cache reader assumed one cache per role. Partial models have several actual compiled stages. '
            'Stop only the verified childless confirmation waiter. Existing qualification GPU jobs continue. '
            'Preserve failed reader proofs. A new reader version reuses completed GPU bytes and continues only unlaunched controls.',
        stopped_CPU_pid=controller,owned_GPU_processes_stopped=False,parent_plan_SHA256=digest(old)))
    value=json.loads(old.read_text())
    for item in value['items']:
        item['reader_recovery_parent_gate']=item['gate'];item['gate']='pc_warm_regime_'+item['label'].lower()+'_composition_v9.json'
    value.update(time=time.time(),parent_plan_SHA256=digest(old),reader_recovery_only=True,
        reader_scope='Track every full/partial compiler region. Require unchanged root sets and observed graph reuse for all stages. '
            'Require official AOT loading for official full-model cache regions. Partial stages use their recorded graph loader. '
            'Keep completed GPU controls. Never repeat a failed or interrupted GPU job. Preserve the8/32/400 quality and strict criteria.')
    path=ROOT/'configs/common_warm_regime_qualification_v9.json';assert not path.exists();write(path,value)
    children=[]
    for lane in LANES:
        log=ROOT/'logs'/f'common_warm_regime_qualification_lane{lane}_v9.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','qualify9',str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=lane,pid=child.pid))
    log=ROOT/'logs/common_warm_regime_confirmation_v9.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','confirm9'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/common_warm_regime_controllers_v9.json',dict(time=time.time(),children=children,confirmation_pid=child.pid))
    return children


def prepare_staged_dispatch():
    from .common import bind
    bind();preserve();controller=json.loads((ROOT/'state/common_warm_regime_controllers_v9.json').read_text())['confirmation_pid']
    assert not (ROOT/'jobs/coordinator_hygiene_c3_confirmation_warm_regime_v9.json').exists()
    proc=Path('/proc',str(controller));assert proc.stat().st_uid==os.getuid()
    words=(proc/'cmdline').read_bytes()
    assert b'warm_serving_followup' in words and b'confirm9' in words
    assert all(not (task/'children').read_text().strip() for task in (proc/'task').iterdir())
    os.kill(controller,signal.SIGTERM)
    old_path=ROOT/'state/budget.json';old=json.loads(old_path.read_text());original_SHA=digest(old_path)
    new=dict(old,last_large_dispatch_epoch=old['deadline_epoch']-75*60,closeout_reserve_hours=1.25)
    write(ROOT/'state/budget_dispatch_reallocation_v1.json',dict(time=time.time(),original=old,original_SHA256=original_SHA,new=new,
        stopped_childless_CPU_pid=controller,owned_GPU_work_stopped=False,maximum_wall_clock_changed=False,
        reason='Reserve75minutes for closeout within the unchanged10-hour end time. Prioritize complete Equal-output N3. '
            'Gate the entire Natural N3 stage on observed runtime,not on favorable performance. Reuse only verified same-pair compiler primes.'))
    write(old_path,new)
    write(ROOT/'state/coordinator_hygiene_warm_regime_not_run_v9.json',dict(time=time.time(),status='NOT_RUN',
        reason='Before any timing launch,replace the childless V9 all-at-once waiter with a budgeted two-stage dispatcher. '
            'All live V9 GPU qualification continues. Gates,model,workload,actions and scientific thresholds stay unchanged.'))
    log=ROOT/'logs/common_warm_regime_confirmation_v10.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','confirm10'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/common_warm_regime_controller_v10.json',dict(time=time.time(),pid=child.pid));return child.pid


def restart_staged_after_template_fix():
    from .common import bind
    bind();preserve();record=ROOT/'state/common_warm_regime_controller_v10.json'
    old=json.loads(record.read_text());assert not (ROOT/'jobs/coordinator_hygiene_c3_confirmation_warm_regime_v10.json').exists()
    proc=Path('/proc',str(old['pid']));assert proc.stat().st_uid==os.getuid()
    words=(proc/'cmdline').read_bytes();assert b'warm_serving_followup' in words and b'confirm10' in words
    assert all(not (task/'children').read_text().strip() for task in (proc/'task').iterdir())
    os.kill(old['pid'],signal.SIGTERM)
    target=ROOT/'state/warm_v10_priming_template_recovery.json';assert not target.exists()
    write(target,dict(time=time.time(),stopped_childless_CPU_pid=old['pid'],GPU_work_stopped=False,
        original_controller_record_SHA256=digest(record),
        reason='The CPU identity test found that the strict Stock primer used controlled_pipeline=True. '
            'The real C3 configuration uses False. Do not ignore this flag or weaken the model identity check. '
            'Build timing primers from the actual Natural400 continuous configuration instead. '
            'No V10 timing or priming job was created or executed before this correction.'))
    log=ROOT/'logs/common_warm_regime_confirmation_v10_template_fix1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_serving_followup','confirm10'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/common_warm_regime_controller_v10_template_fix1.json',dict(time=time.time(),pid=child.pid));return child.pid


if __name__=='__main__':
    import sys
    if sys.argv[1] in ('qualify9','confirm9','confirm10'):
        VERSION=9;QUAL='common_warm_regime_qualification_v9.json';PLAN='coordinator_hygiene_c3_confirmation_warm_regime_v9.json'
    if sys.argv[1]=='confirm10':VERSION=10;PLAN='coordinator_hygiene_c3_confirmation_warm_regime_v10.json'
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    elif sys.argv[1]=='recover-reader':print(prepare_reader_recovery(),flush=True)
    elif sys.argv[1]=='stage-dispatch':print(prepare_staged_dispatch(),flush=True)
    elif sys.argv[1]=='fix-priming-template':print(restart_staged_after_template_fix(),flush=True)
    elif sys.argv[1] in ('qualify','qualify9'):qualify(int(sys.argv[2]))
    elif sys.argv[1] in ('confirm','confirm9','confirm10'):confirm()
    else:raise SystemExit('Expected prepare,qualify,or confirm')
