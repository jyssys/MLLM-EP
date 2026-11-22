"""Tune explicit admission limits without weakening the compiled Token baseline."""
import argparse
import copy
import hashlib
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,dispatch_allowed
from .runner import save,signature
from .c3_campaign import get,qualified_registry,performance,launch,wait_file

FAMILIES=('Layer','StockToken','Hybrid','Deflection')

def family(item):return 'StockToken' if item['arm']=='StockToken' else item['arm']

def screen_score(label,blocks,common_label='NativeCommon2'):
    ratios=[];goodput=[];safe=True;quality=True;seen=0
    from .quality_gate import compare
    for qps in sorted({r['qps'] for r in blocks}):
        bylabel={r['label']:r['name'] for r in blocks if r['qps']==qps}
        if label not in bylabel:continue
        m=json.loads((ROOT/'analysis'/(bylabel[label]+'_metrics.json')).read_text())
        refs=[json.loads((ROOT/'analysis'/(bylabel[k]+'_metrics.json')).read_text()) for k in ('NativeStrong',common_label)]
        qpath=ROOT/'analysis'/(bylabel[label]+'_quality_gate.json')
        q=json.loads(qpath.read_text()) if qpath.exists() else compare(bylabel[label])
        quality &= q['passed'];safe &= m['decode_SLO_attainment']>=.95 and all(
            m[field]<=1.05*r[field] for r in refs for field in ('ITL_p99_ms','TPOT_p95_ms'))
        ratios.append(m['TTFT_p95_ms']/refs[0]['TTFT_p95_ms']);goodput.append(m['SLO_qualified_output_tokens_per_second']);seen+=1
    assert seen==len({r['qps'] for r in blocks})
    return dict(label=label,quality=bool(quality),discovery_point_decode_screen=bool(safe),
        mean_Native_normalized_TTFT=float(np.mean(ratios)),mean_goodput=float(np.mean(goodput)),
        scope='Single-start discovery screen. This is not the paired95%CI safety gate.')

def freeze_geometry():
    plan=json.loads((ROOT/'jobs/c3_clean_geometry_discovery_v2.json').read_text());registry=qualified_registry();selected={};scores=[]
    supplement=json.loads((ROOT/'jobs/c3_geometry_knee_refinement_v3.json').read_text())
    plan['blocks']+=supplement['blocks']
    for axis in FAMILIES:
        candidates=[]
        for label,item in registry.items():
            if family(item)!=axis:continue
            if not all((ROOT/'analysis'/(b['name']+'_metrics.json')).exists() for b in plan['blocks'] if b['label']==label):continue
            if not any(b['label']==label for b in plan['blocks']):continue
            score=screen_score(label,plan['blocks']);scores.append(score)
            if score['quality']:candidates.append(score)
        assert candidates,('No quality-qualified geometry in family',axis)
        winner=min(candidates,key=lambda r:(not r['discovery_point_decode_screen'],r['mean_Native_normalized_TTFT'],-r['mean_goodput'],r['label']))
        selected[axis]=dict(label=winner['label'],item=registry[winner['label']],screen=winner,
            safety_status='POINT_SCREEN_ONLY' if winner['discovery_point_decode_screen'] else 'SAFETY_UNCONFIRMED')
    target=ROOT/'configs/c3_geometry_selection_v2.json';assert not target.exists()
    result=dict(time=time.time(),selected=selected,scores=scores,heldout_seen=False,
        capacity_refinement_SHA256=digest(ROOT/'configs/c3_confirmation_knee_refinement_v3.json'),
        scope='Freeze geometry before changing admission caps. Capacity-boundary and both original overload discovery loads enter the score. '
            'Every family uses the same prospective screen. An unsafe retained configuration is not a serving success.')
    write(target,result);return result

def policies(selection):
    source='v2_preemption_observer_fix_v5.zip';sha=digest(ROOT/'references/source_capsules'/source)
    registry=qualified_registry();items=[];reuse={};requested=[]
    for axis in FAMILIES:
        base=selection['selected'][axis]['item'];cfg=get(base['jobs'][-1]);credit=cfg.get('continuous_source_credit_limit',0)
        for cap in (0,1,2,4):requested.append((axis,base,cap,credit,None,False))
        if axis in ('Layer','Hybrid'):
            requested.append((axis,base,2,16,None,False))
            for compiled in (False,True):requested.append((axis,base,2,credit,compiled,False))
        if axis=='Hybrid':requested.append((axis,base,2,credit,None,True))
        if axis=='StockToken':
            for other in (0,8,16):
                if other!=credit:requested.append((axis,base,2,other,None,False))
    native=registry['NativeCommon2']
    for cap in (0,1,2,4):requested.append(('NativeCommon',native,cap,0,None,False))
    for index,(axis,base,cap,credit,compiled,fused) in enumerate(requested):
        variant='_FusedShort' if fused else '' if compiled is None else '_CompiledPrefix' if compiled else '_EagerPrefix'
        label=f'{axis}{variant}_Cap{cap}_Credit{credit}';basecfg=get(base['jobs'][-1]);lane=index%2;pair=[2*lane,2*lane+1]
        if compiled is None and not fused and basecfg.get('D_running_admission_cap')==cap and basecfg.get('continuous_source_credit_limit',0)==credit:
            reuse[label]=dict(base,label=label,family=axis,cap=cap,credit=credit);continue
        teacher=base.get('teacher');strict=None
        if teacher:
            old=get(base['strict']);strict=f'V2_C3Policy_{label}_Strict8';old.update(name=strict,gpus=pair,
                source_capsule=source,source_capsule_SHA256=sha,D_running_admission_cap=cap,d_compute_throttle=True)
            # Keep actual serial geometry separate from the prospective C3 window.
            old['continuous_source_credit_limit']=0;save(old)
            if compiled is not None:
                # Replace the just-planned name with a new immutable config.
                # The initial strict name is retained as NOT_RUN configuration.
                strict+='_PFlag';old.update(name=strict,P_compile_prefix=compiled);save(old)
            if fused:
                # Numeric query geometry changes. Use an independent local
                # three-wave original-layer teacher,not the old chunk teacher.
                pool=json.loads((ROOT/'data/combined_original_candidate_pool_v2.json').read_text())
                main_contexts={r['context_SHA256'] for r in json.loads((ROOT/'data/longbench_balanced400_v2.json').read_text())}
                small=sorted([r for r in pool if r['input_bucket']==0 and r['context_SHA256'] not in main_contexts],
                    key=lambda r:hashlib.sha256(('v2-fused-control:'+r['prompt_id']).encode()).hexdigest())[:8]
                assert len(small)==8 and all(r['input_tokens']<=8192 for r in small)
                dataset='fused_short_original_qualification8_v2.json';data_path=ROOT/'data'/dataset
                if data_path.exists():assert json.loads(data_path.read_text())==small
                else:write(data_path,small)
                strict+='_FusedDAG';old.update(name=strict,dataset=dataset,dataset_SHA256=digest(data_path),count=8,fuse_short_hybrid=True);save(old)
                teacher_cfg=get(base['teacher']);teacher='V2_C3Policy_HybridFused_LocalThreeWaveTeacher8'
                for key in ('token_teacher_cut','teacher_suffix_fraction','teacher_eager_prefix_compiled_suffix','suffix_by_request'):
                    teacher_cfg.pop(key,None)
                teacher_cfg.update(name=teacher,gpus=pair,dataset=dataset,dataset_SHA256=digest(data_path),count=8,
                    local_hybrid_teacher_m=basecfg['axis_cut'],local_hybrid_teacher_fraction=basecfg['hybrid_suffix_fraction'],
                    source_capsule=source,source_capsule_SHA256=sha)
                save(teacher_cfg)
        jobs=[]
        for n,dataset in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            cfg=get(base['jobs'][0 if n==32 else -1]);name=f'V2_C3Policy_{label}_Natural{n}'
            cfg.update(name=name,gpus=pair,source_capsule=source,source_capsule_SHA256=sha,
                D_running_admission_cap=cap,continuous_source_credit_limit=credit)
            if compiled is not None:cfg['P_compile_prefix']=compiled
            if fused:cfg['fuse_short_hybrid']=True
            save(cfg);jobs.append(name)
        items.append(dict(label=label,arm=base['arm'],family=axis,cut=base.get('cut'),fraction=base.get('fraction'),cap=cap,credit=credit,
            P_compile_variant=compiled,fused_short=fused,lane=lane,pair=pair,teacher=teacher,strict=strict,jobs=jobs,gate=f'c3_policy_{label}_v2.json',signature=signature(cfg)))
    plan=dict(time=time.time(),items=items,reused=reuse,state_prefix='c3_policy_qualification_v2',source_capsule=source,source_capsule_SHA256=sha,
        geometry_selection_SHA256=digest(ROOT/'configs/c3_geometry_selection_v2.json'),
        scope='Same cap set0/1/2/4 for Layer,Strong Stock Token,Hybrid,Deflection and common-policy Native. '
            'Cap0 retains decode-first compute,remote READ and registration deferral. It removes only the running-request cap. '
            'The added credit16 controls do not change batch limits. Add same-source P-compiler on/off controls for Layer and Hybrid at cap2. '
            'Keep D adapter,compiled decode,original weights and every other runtime flag unchanged. '
            'Also test the existing dependency-compatible fused short-Hybrid path. It needs its independent original-layer three-wave teacher. '
            'Only prompts<=B8192 use the fused D frame. Long prompts retain the aligned path. '
            'These controls enter discovery before heldout selection. Every new signature gets state/quality32/400 validation. '
            'Exact stock-token suffixes remain bound to the current dataset. No unsafe active-decode insertion is introduced.')
    write(ROOT/'configs/c3_policy_qualification_v2.json',plan);print('C3_SHARED_POLICY_SPACE_FROZEN',len(items),'new',len(reuse),'reused',flush=True)
    return plan

def run_qualification():
    processes=[]
    for lane in (0,1):
        log=ROOT/'logs'/f'c3_policy_qualification_lane{lane}_v2.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.c3_family','run','--lane',str(lane),
                '--plan','c3_policy_qualification_v2.json'],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(child)
    while any(p.poll() is None for p in processes):time.sleep(5)
    assert all(p.returncode==0 for p in processes)

def policy_registry():
    plan=json.loads((ROOT/'configs/c3_policy_qualification_v2.json').read_text());registry=dict(plan['reused'])
    for item in plan['items']:
        gate=ROOT/'state'/item['gate']
        if gate.exists() and json.loads(gate.read_text())['passed']:registry[item['label']]=item
    return registry

def discovery():
    registry=policy_registry();original=qualified_registry();grid=json.loads((ROOT/'configs/c3_common_qps_grid_v2.json').read_text())
    lanes=[[],[],[],[]];blocks=[]
    loads=json.loads((ROOT/'configs/c3_confirmation_knee_refinement_v3.json').read_text())['effective_discovery_QPS']
    for index,qps in enumerate(loads):
        lane=index%2
        pair=[2*lane,2*lane+1];actions=dict(registry,NativeStrong=original['NativeStrong'],Agg=original['Agg'])
        # Native cap2 remains an explicit anchor even when a reused label changed.
        actions['NativeCommon2']=original['NativeCommon2']
        ordered=['NativeStrong','NativeCommon2','Agg']+sorted(k for k in actions if k not in ('NativeStrong','NativeCommon2','Agg'))
        for label in ordered:
            name=f'V2_C3PolicyTune_q{qps}_{label}_Discovery208'
            performance(actions[label],name,'balanced400_v2_discovery.json',pair,qps,2026112263)
            lanes[lane].append(name);blocks.append(dict(qps=qps,label=label,name=name,pair=pair,gate=actions[label]['gate']))
    target=ROOT/'jobs/c3_policy_discovery_v2.json';assert not target.exists()
    write(target,dict(time=time.time(),lanes=lanes,blocks=blocks,stop_on_failure=False,
        capacity_refinement_SHA256=digest(ROOT/'configs/c3_confirmation_knee_refinement_v3.json'),
        scope='Same original discovery208 and trace for every policy within each load. '
            'Cap controls are not actual-batch guarantees. No heldout data enter tuning.'))
    return target.name

def run():
    wait_file(ROOT/'state/c3_pilot_and_geometry_complete_v2.json')
    wait_file(ROOT/'state/c3_geometry_knee_refinement_complete_v3.json')
    selection=freeze_geometry();policies(selection);run_qualification();launch(discovery())
    write(ROOT/'state/c3_policy_tuning_complete_v2.json',dict(time=time.time(),complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','freeze_geometry','discovery'));args=p.parse_args();result=globals()[args.action]()
    if result is not None:print(result,flush=True)
