"""Restore the required independent three-wave teacher dispatch context."""
import argparse
import copy
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest
from .runner import save
from .c3_campaign import wait_file,launch

def prepare():
    target=ROOT/'configs/c3_fused_teacher_context_supplement_v15.json';assert not target.exists()
    plan=json.loads((ROOT/'configs/c3_policy_qualification_v2.json').read_text());items=[];audit=[]
    for lane,label in enumerate(('Hybrid_FusedShort_Cap2_Credit8','Hybrid_FusedShort_DCompiledInterval_Cap2_Credit8')):
        item=copy.deepcopy(next(r for r in plan['items'] if r['label']==label));pair=[2*lane,2*lane+1]
        before=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text())
        assert 'local_hybrid_teacher_m' in before and before['teacher_eager_all_prompt']
        assert not before.get('teacher_eager_prefix_compiled_suffix')
        teacher=item['teacher']+'_ContextFixV15';after=dict(before,name=teacher,gpus=pair,teacher_eager_prefix_compiled_suffix=True)
        save(after)
        strict=json.loads((ROOT/'configs'/(item['strict']+'.json')).read_text());strict.update(name=item['strict']+'_ContextFixV15',gpus=pair)
        save(strict);jobs=[]
        for old in item['jobs']:
            cfg=json.loads((ROOT/'configs'/(old+'.json')).read_text());cfg.update(name=old+'_ContextFixV15',gpus=pair)
            save(cfg);jobs.append(cfg['name'])
        item.update(label=label+'_ContextFixV15',teacher=teacher,strict=strict['name'],jobs=jobs,lane=lane,pair=pair,
            gate=f'c3_policy_{label}_context_fix_v15.json')
        # The candidate runtime is unchanged. Only its independent teacher's
        # explicit eager-dispatch activation is repaired in this new version.
        items.append(item);audit.append(dict(old_teacher=before['name'],new_teacher=teacher,
            old_required_dispatch_flag=False,new_required_dispatch_flag=True,
            original_failed_gate_retained=True,candidate_model_runtime_changed=False))
    write(target,dict(time=time.time(),items=items,state_prefix='c3_fused_teacher_context_v15',audit=audit,
        source_capsule_SHA256=items[0]['source_capsule_SHA256'],
        scope='The source5 teacher-stage installer returns immediately without teacher_eager_prefix_compiled_suffix. '
            'The all-prompt flag alone does not activate that installer. Restore its explicit dispatch activation. '
            'The local original-layer three-wave teacher stays independent of NIXL and the partition adapter. '
            'Do not alter weights,inputs,outputs,tensors,sampling,mask or tolerances. '
            'Fresh strict8 and mixed natural32/400 gates remain mandatory. Preserve the old failed comparison.'))
    print('FUSED_INDEPENDENT_TEACHER_CONTEXT_FIX_FROZEN',flush=True)

def run_qualification():
    children=[]
    for lane in (0,1):
        log=ROOT/'logs'/f'c3_fused_teacher_context_v15_lane{lane}.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_output_aware_v2.c3_family','run','--lane',str(lane),
                '--plan','c3_fused_teacher_context_supplement_v15.json'],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
    while any(c.poll() is None for c in children):time.sleep(5)
    assert all(c.returncode==0 for c in children)
    for item in json.loads((ROOT/'configs/c3_fused_teacher_context_supplement_v15.json').read_text())['items']:
        path=ROOT/'raw'/item['teacher']/'independent_teacher_execution_path.json'
        assert path.exists() and json.loads(path.read_text())['rows'],'The teacher dispatch path must actually execute'

def resume():
    for lane in (0,1):wait_file(ROOT/'state'/f'c3_policy_qualification_v2_lane{lane}_complete_v4.json')
    run_qualification()
    launch('stock_pre_forward_fence_control_v14.json',allow_partial_budget=True)
    from .stock_prefence_control import analyze
    analyze()
    from .c3_policy import discovery
    launch(discovery())
    write(ROOT/'state/c3_policy_tuning_complete_v2.json',dict(time=time.time(),complete=True,
        policy_load_budget_SHA256=digest(ROOT/'configs/c3_policy_load_budget_v7.json'),
        independent_teacher_fix='c3_fused_teacher_context_supplement_v15.json',prefence_diagnostic='stock_pre_forward_fence_control_v14.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','resume'));a=p.parse_args();globals()[a.action]()
