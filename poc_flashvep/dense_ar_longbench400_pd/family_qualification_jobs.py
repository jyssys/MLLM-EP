"""Share independent full-model boundary teachers across genuine2D actions."""
import copy
import hashlib
import json
from .common import ROOT,write,digest
from .experiment import create
from .next_jobs import plan
from .geometry import aligned_suffix


def suffix(length,action):
    if action['kind']=='Layer':return 0
    if action['kind']=='Token':return aligned_suffix(length,action['cut'])
    f=action['suffix_fraction']
    return 0 if not f else min((length-1)//16*16,max(16,int(round(length*f/16))*16))


def save_data(path,value):
    if path.exists():assert json.loads(path.read_text())==value
    else:write(path,value)


def main():
    selected=json.loads((ROOT/'data/longbench_balanced400_v1.json').read_text())
    excluded={r['context_SHA256'] for r in selected}
    pool=json.loads((ROOT/'data/candidate_pool_original.json').read_text())
    base=[]
    for bucket in range(4):
        eligible=[r for r in pool if r['input_bucket']==bucket and r['context_SHA256'] not in excluded]
        ordered=sorted(eligible,key=lambda r:hashlib.sha256(('family-qualification-v1-'+r['prompt_id']).encode()).hexdigest())
        picked=[]
        for row in ordered:
            if row['context_SHA256'] not in {r['context_SHA256'] for r in picked}:picked.append(row)
            if len(picked)==2:break
        assert len(picked)==2;base.extend(picked)
    actions=[dict(kind='Hybrid',cut=m,suffix_fraction=f) for m in (0,2,4,8,12)
        for f in (0.,1/16,1/8,1/4,1/3) if m or f]
    actions.extend([dict(kind='Layer',cut=1),dict(kind='Token',cut=1)])
    actual=[];teachers=[];teacher_index={};mapping=[];action_map={};actual_suffixes={};teacher_suffixes={}
    for action_index,action in enumerate(actions):
        for original in base:
            s=suffix(original['input_tokens'],action);key=original['prompt_id'],s
            if key not in teacher_index:
                row=copy.deepcopy(original);row.update(source_prompt_id=original['prompt_id'],
                    prompt_id=original['prompt_id']+f'_BOUNDARY{s}',diagnostic_repeated_prompt=True)
                teacher_index[key]=len(teachers);teacher_suffixes['q'+str(len(teachers))]=s;teachers.append(row)
            row=copy.deepcopy(original);row.update(source_prompt_id=original['prompt_id'],
                prompt_id=original['prompt_id']+f'_ACTION{action_index}',diagnostic_repeated_prompt=True)
            index=len(actual);actual.append(row);action_map['q'+str(index)]=action;actual_suffixes['q'+str(index)]=s
            mapping.append(dict(actual_index=index,teacher_index=teacher_index[key],source_prompt_id=original['prompt_id'],
                input_tokens=original['input_tokens'],suffix=s,action=action))
    ap=ROOT/'data/family_qualification_actions208.json';tp=ROOT/'data/family_boundary_teachers.json'
    assert len(actual)==208;save_data(ap,actual);save_data(tp,teachers)
    manifest=dict(actions=actions,base_prompt_ids=[r['prompt_id'] for r in base],mapping=mapping,
        actual_dataset_SHA256=digest(ap),teacher_dataset_SHA256=digest(tp),
        source_context_overlap_with_Main400=0,evidence='FROZEN_CORRECTNESS_DESIGN',
        scope='Eight unmodified calibration prompts,disjoint from Main400 contexts. Each action repeats those prompts. '
            'These208 executions are not208 independent source requests or server starts. '
            'Teacher reuse is legal because all aligned interior actions share the same full36 query-boundary geometry.')
    save_data(ROOT/'configs/family_full_state_qualification_frozen_v1.json',manifest)
    jobs=[create('Q7_Full36_BoundaryTeachers_v1','Native',tp.name,pair=(4,5),output=16,fixed=True,
        correctness=True,reference_single=True,serial=True,axis_state_capture=False,state_capture=True,
        token_teacher_cut=4,suffix_by_request=teacher_suffixes,numerical_probe=True,
        numerical_probe_sync=True,kv_layout='HND',wall_timeout=3600,serving_timeout=3300),
        create('Q7_Hybrid_Family208_FullState_v1','Hybrid',ap.name,cut=12,pair=(4,5),output=16,fixed=True,
            correctness=True,controlled=True,serial=True,axis_state_capture=True,
            axis_capture_hidden_last_only=True,axis_hash_workers=8,
            hybrid_suffix_fraction=.125,partition_aligned_chunks=True,
            axis_actions_by_request=action_map,suffix_by_request=actual_suffixes,
            wall_timeout=5400,serving_timeout=5100)]
    plan('family_full_state_qualification_v1',jobs)
    print('Frozen',len(teachers),'original-model boundary cases in one teacher server start and208 action executions.',flush=True)


if __name__=='__main__':main()
