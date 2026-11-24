"""Prospectively qualify one-factor policies under an explicit warm regime."""
import argparse
import copy
import fcntl
import json
import subprocess
import time
import traceback
import numpy as np
from .common import ROOT,PRIOR,V2,REPO,PYTHON,write,digest,preserve,resources
from .runtime import bind,copy_data,source_capsule,dispatch_allowed,start_guardian

PLAN='root_cause_policy_qualification_v1.json'
ARMS={
    'NativeStrong':dict(parent='NativeStrong',lane=0),
    'NativeCommon2':dict(parent='NativeCommon2',lane=0),
    'NativeRegOpen':dict(parent='NativeCommon2',lane=0,control='NativeCommon2',changes=dict(defer_D_request_registration=False)),
    'StockToken':dict(parent='StockToken',lane=1),
    'StockCap2':dict(parent='StockToken',lane=1,control='StockToken',changes=dict(D_running_admission_cap=2)),
    'StockCap2RegOpen':dict(parent='StockToken',lane=1,control='StockCap2',changes=dict(D_running_admission_cap=2,defer_D_request_registration=False)),
    'Layer':dict(parent='Layer',lane=2),
    'LayerRegOpen':dict(parent='Layer',lane=2,control='Layer',changes=dict(defer_D_request_registration=False)),
    'LayerRetirement':dict(parent='Layer',lane=2,control='Layer',changes=dict(source_credit_release_on_retirement=True)),
    'Hybrid':dict(parent='Hybrid',lane=3),
    'HybridRegOpen':dict(parent='Hybrid',lane=3,control='Hybrid',changes=dict(defer_D_request_registration=False)),
    'HybridRetirement':dict(parent='Hybrid',lane=3,control='Hybrid',changes=dict(source_credit_release_on_retirement=True))}


def template(label):
    spec=ARMS[label]
    cfg=json.loads((PRIOR/'configs'/f'PC_WarmQual_{spec["parent"]}_Natural400.json').read_text())
    capsule=source_capsule(cfg['source_capsule'])
    cfg.update(source_capsule=capsule.name,source_capsule_SHA256=digest(capsule))
    cfg.update(spec.get('changes',{}))
    cfg.pop('qualification_gate_v2',None)
    return cfg


def capture(cfg):
    axis=bool(cfg.get('axis_kind'))
    cfg.update(correctness_only=True,state_capture=not axis,axis_state_capture=axis,gpu_state_digest=True,
        axis_capture_hidden_last_only=False,axis_import_hash_after_model=True)
    return cfg


def save_on_data(runner,cfg,name,data,count,pair,serial,output):
    path=copy_data(data)
    cfg=copy.deepcopy(cfg);cfg.update(name=name,gpus=pair,dataset=data,dataset_SHA256=digest(path),count=count,
        serial=serial,ignore_eos=bool(output),output_cap=output or 2048,
        arrival_offsets=[0.]*count if serial else np.r_[0.,np.cumsum(np.random.default_rng(2026112401).exponential(.5,count-1))].tolist(),
        official_compiler_cache_regime='PREEXISTING_VERIFIED_AOT',serving_timeout=4000,wall_timeout=4500)
    cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None)
    from ..dense_ar_output_aware_v2.work_contract import regenerate
    regenerate(cfg,json.loads(path.read_text())[:count]);runner.save(cfg)
    return name


def prepare():
    runner=bind();preserve();start_guardian()
    for name in ('longbench_balanced400_v2.json','balanced400_v2_heldout.json','balanced400_v2_discovery.json',
            'qualification32.json','compiler_control8_original_v1.json'):copy_data(name)
    source=PRIOR/'configs/c3_quality_contract_v2.json'
    target=ROOT/'configs/c3_quality_contract_v2.json'
    if not target.exists():write(target,json.loads(source.read_text()))
    assert digest(target)==digest(source)
    items=[]
    for label,spec in ARMS.items():
        lane=spec['lane'];pair=[2*lane,2*lane+1];cfg=capture(template(label));steps={}
        for stage,count in (('Single1',1),('Prime8',8),('Strict8A',8),('Strict8B',8)):
            name=f'NS_Qual_{label}_{stage}'
            steps[stage]=save_on_data(runner,cfg,name,'compiler_control8_original_v1.json',count,pair,True,16)
        for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            value=dict(cfg,axis_capture_hidden_last_only=True)
            name=f'NS_Qual_{label}_Natural{count}'
            steps[f'Natural{count}']=save_on_data(runner,value,name,data,count,pair,False,None)
        final=json.loads((ROOT/'configs'/(steps['Natural400']+'.json')).read_text())
        items.append(dict(label=label,lane=lane,pair=pair,control=spec.get('control'),steps=steps,
            gate=f'ns_{label.lower()}_composition_v1.json',signature=runner.signature(final),
            source_capsule_SHA256=final['source_capsule_SHA256']))
    # Assert the advertised causal contrasts in the exact runtime signature.
    contrasts=[];indexed={row['label']:row for row in items}
    for row in items:
        if not row['control']:continue
        left=row['signature'];right=indexed[row['control']]['signature']
        changed={key:(right.get(key),left.get(key)) for key in set(left)|set(right) if left.get(key)!=right.get(key)}
        assert len(changed)==1,(row['label'],changed)
        contrasts.append(dict(candidate=row['label'],control=row['control'],changed_signature_fields=changed))
    path=ROOT/'configs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),items=items,contrasts=contrasts,
        scope='Twelve policies,not a partition or QPS sweep. Preserve all original strong compiled paths. '
            'Fresh Single1,priming8,prospective warm strict8 pairs,Natural32 and Natural400 quality/lifetime checks. '
            'RegOpen changes host registration only;existing decode-compute priority and remote-READ deferral remain unchanged. '
            'No future safe-window predictor,active-decode insertion or output correction is introduced. '
            'Retirement on Main400 is new workload-level confirmation,not a new credit mechanism. '
            'Qualification timings are not primary performance starts. Independent timing and host-contention validation follow separately.'))
    children=[]
    for lane in range(4):
        log=ROOT/'logs'/f'policy_qualification_lane{lane}_v1.log'
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.qualification','run',str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=lane,pid=child.pid,log=str(log.relative_to(ROOT))))
    write(ROOT/'state/policy_qualification_controllers_v1.json',dict(time=time.time(),children=children))
    return children


def state_audit(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    if cfg.get('axis_kind'):
        from ..dense_ar_longbench400_pd.qualification import audit
        value=audit(name)
        assert value['passed'] and value['decode_full36_observation_complete']
        assert all(row['status']=='PASS' for row in value['imported_prefix_KV'])
        return value
    from ..dense_ar_longbench400_pd.remote_setup_qualification import native_state
    return native_state(name)


def strict_compare(left,right,axis):
    if axis:
        from ..dense_ar_p_critical_path.coordinator_state_fix import compare
        return compare(left,right)
    from ..dense_ar_longbench400_pd.remote_setup_qualification import native_state
    value=native_state(right,left)
    write(ROOT/'analysis'/(right+'_native_strict_pair.json'),dict(time=time.time(),**value))
    return value


def run(lane):
    bind()
    from ..dense_ar_p_critical_path.stable_source import install
    runner=install()
    from ..dense_ar_p_critical_path.warm_serving_followup import expected_roots,check_reuse
    from ..dense_ar_p_critical_path.quality import compare as quality
    plan=json.loads((ROOT/'configs'/PLAN).read_text());pair=[2*lane,2*lane+1];owner=resources();results=[]
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for item in [row for row in plan['items'] if row['lane']==lane]:
                proof=dict(time=time.time(),passed=False,label=item['label'],stages=[],signature=item['signature'],
                    source_capsule_SHA256=item['source_capsule_SHA256'],evidence='CORRECTNESS_ONLY')
                try:
                    def execute(name):
                        assert dispatch_allowed(),'Dispatch deadline reached before the next control'
                        assert not (ROOT/'raw'/name).exists(),'Never retry a live,failed or interrupted job in place'
                        owner.stop(pair);assert runner.execute(name),name
                        value=state_audit(name);assert value['passed'],name
                        proof['stages'].append(dict(name=name,state=value))
                        return value
                    for stage in ('Single1','Prime8'):execute(item['steps'][stage])
                    roots=expected_roots(item['steps']['Prime8'])
                    assert set(roots)=={'prefill','decode'}
                    for stage in ('Strict8A','Strict8B'):
                        name=item['steps'][stage];execute(name);check_reuse(name,roots)
                    cfg=json.loads((ROOT/'configs'/(item['steps']['Strict8B']+'.json')).read_text())
                    proof['warm_strict']=strict_compare(item['steps']['Strict8A'],item['steps']['Strict8B'],bool(cfg.get('axis_kind')))
                    assert proof['warm_strict']['passed']
                    if item['control']:
                        control=next(row for row in plan['items'] if row['label']==item['control'])
                        control_proof=json.loads((ROOT/'state'/control['gate']).read_text());assert control_proof['passed']
                        proof['one_factor_strict']=strict_compare(control['steps']['Strict8B'],item['steps']['Strict8B'],bool(cfg.get('axis_kind')))
                        assert proof['one_factor_strict']['passed']
                    for stage in ('Natural32','Natural400'):
                        name=item['steps'][stage];execute(name);reuse=check_reuse(name,roots);score=quality(name)
                        proof['stages'][-1].update(quality=score,cache_reuse=reuse)
                        assert score['passed'],score
                    proof.update(passed=True,full400_passed=True,quality_passed=True,official_cache_roots=roots)
                except Exception as error:
                    proof.update(error=repr(error),traceback=traceback.format_exc(),evidence='FAILED')
                write(ROOT/'state'/item['gate'],proof)
                results.append(dict(label=item['label'],passed=proof['passed'],error=proof.get('error')))
                write(ROOT/'state'/f'policy_qualification_lane{lane}_progress_v1.json',dict(time=time.time(),results=results))
                print('NS_POLICY_QUALIFICATION',json.dumps(results[-1]),flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'policy_qualification_lane{lane}_complete_v1.json',dict(time=time.time(),results=results,complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));parser.add_argument('lane',type=int,nargs='?');args=parser.parse_args()
    print(prepare() if args.action=='prepare' else run(args.lane),flush=True)
