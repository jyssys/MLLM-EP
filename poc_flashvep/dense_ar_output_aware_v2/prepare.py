"""Freeze prospective jobs using the preserved runtime contracts."""
import argparse
import json
import time
import numpy as np
from .common import ROOT,PREVIOUS,write,digest,initialize
from .runner import save,signature

CAPSULE='v2_p_compiled_prefix_experiment_v1.zip'

def clone(old,name,dataset,pair,count=None,qps=2.,seed=2026112201,**extra):
    cfg=json.loads((PREVIOUS/'configs'/(old+'.json')).read_text())
    cfg={k:v for k,v in cfg.items() if 'gate' not in k and not k.startswith(('final_selection_','final_static_choice_'))}
    # Request-local maps belong to a dataset and order,not to an engine prototype.
    for key in ('suffix_by_request','aggregate_work_audit','axis_actions_by_request','replay_output_lengths'):
        cfg.pop(key,None)
    rows=json.loads((ROOT/'data'/dataset).read_text());n=count or len(rows)
    cfg.update(name=name,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=n,gpus=list(pair),qps=qps,seed=seed,
        arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,n-1))].tolist(),
        source_capsule=CAPSULE,source_capsule_SHA256=digest(ROOT/'references/source_capsules'/CAPSULE),
        scope='Versioned v2 original prompts and frozen arrival contract. Prior v1 results remain historical.',**extra)
    return cfg

def c3(arm,name,dataset,pair=(2,3),cut=8,qps=2.,seed=2026112201,count=None,correctness=False,p_compiled=False,cap=2,fraction=.125):
    protos=dict(NativeStrong='EagerSourceQ_NativeStrong_Natural400_v2',NativeCommon='EagerSourceQ_NativeCommon_Natural400_v2',
        Agg='FinalC3_Heldout200_q1.8_s0_Agg_v1',Deflection='FinalC3_Heldout200_q1.8_s0_BestWholeDeflection_v1',
        Layer='EagerDecodeQ_Layer_Natural400_v2',StockToken='EagerSourceQ_StockToken_Natural400_v2',Hybrid='EagerDecodeQ_Hybrid_Natural400_v2')
    cfg=clone(protos[arm],name,dataset,pair,count,qps,seed,correctness_only=correctness,
        controlled_pipeline=False,axis_state_capture=bool(correctness and arm in ('Layer','StockToken','Hybrid')),
        state_capture=bool(correctness and arm not in ('Layer','StockToken','Hybrid')),
        gpu_state_digest=correctness,axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,
        numerical_probe=False,numerical_probe_sync=False,kernel_activity=False,serving_timeout=4000,wall_timeout=4500)
    if arm not in ('NativeStrong','Agg'):
        cfg.update(d_compute_throttle=True,defer_D_remote_setup=True,defer_D_request_registration=True,D_running_admission_cap=cap,
            continuous_source_credit_limit=0 if arm=='NativeCommon' else 8)
    if arm in ('Layer','Hybrid'):
        cfg.update(axis_cut=cut,hidden_source_slots=16,hidden_destination_slots=16,preserve_compiled_decode_only=True,
            D_execution_family_override='INSTALLED_DEFAULT_COMPILE',D_compiled_custom_ops_all=True,D_partial_IR_norm_priority=True,
            D_eager_prefill_IR_norm=True,P_compile_prefix=bool(p_compiled),preserve_compiled_full_model=bool(p_compiled))
    if arm=='StockToken':
        from .work_contract import regenerate
        cfg['axis_cut']=cut;regenerate(cfg,json.loads((ROOT/'data'/dataset).read_text())[:cfg['count']])
    if arm=='Hybrid':cfg.update(hybrid_suffix_fraction=fraction,partition_aligned_chunks=True)
    return cfg

def native():
    initialize();plan=ROOT/'configs/native_baseline_qualification_v2.json';assert not plan.exists()
    jobs=[]
    for arm in ('NativeStrong','Agg','NativeCommon'):
        name=f'V2_Qual_{arm}_Natural400';cfg=c3(arm,name,'longbench_balanced400_v2.json',correctness=True)
        save(cfg);jobs.append(dict(arm=arm,name=name,gate=f'composition_{arm}_v2.json',signature=signature(cfg)))
    write(plan,dict(time=time.time(),jobs=jobs,pair=[2,3],source_capsule=CAPSULE,
        source_capsule_SHA256=digest(ROOT/'references/source_capsules'/CAPSULE)))
    print('NATIVE_AGG_FULL400_QUALIFICATION_FROZEN',flush=True)

def official():
    initialize();contract=ROOT/'configs/c3_quality_contract_v2.json';assert not contract.exists()
    rows=[];seen=set()
    for name in ('longbench_balanced400_v2.json','qualification32.json','compiler_control8_original_v1.json'):
        for row in json.loads((ROOT/'data'/name).read_text()):
            if row['prompt_id'] not in seen:rows.append(row);seen.add(row['prompt_id'])
    data='official_compiled_c1_reference_roster_v2.json';write(ROOT/'data'/data,rows)
    name='V2_OfficialCompiled_C1_NaturalReference'
    cfg=c3('NativeStrong',name,data,pair=(0,1),correctness=True)
    cfg.update(reference_single=True,serial=True,continuous_source_credit_limit=0,state_capture=False,
        gpu_state_digest=False,axis_state_capture=False,output_cap=2048,ignore_eos=False)
    save(cfg)
    write(contract,dict(time=time.time(),primary_reference=name,reference_roster_SHA256=digest(ROOT/'data'/data),
        paired_task_macro_margin_percentage_points=2.,confidence=.95,
        primary='Independent original Full36 official compiled C1 numerical family for C3 quality.',
        secondary='Preserve Mode0 C1 comparison. Dataset length selection remains that frozen Mode0 reference.',
        scope='Frozen before viewing new C3 task-quality results. No change to prompts,dataset IDs,SLO or margin. '
            'Different BI0 geometry may change greedy tokens. State ownership and coverage remain separate mandatory tests.'))
    write(ROOT/'jobs/official_reference_v2.json',dict(lanes=[[name],[],[],[]],stop_on_failure=True))
    print('OFFICIAL_COMPILED_C1_REFERENCE_FROZEN',len(rows),flush=True)

def c0():
    initialize();path=ROOT/'configs/c0_full_state_qualification_v2.json';assert not path.exists()
    jobs=[]
    def add(old,name,**extra):
        cfg=clone(old,name,'qualification8.json',(6,7),count=8,qps=20.,correctness_only=True,**extra)
        save(cfg);return name
    teacher=add('Q4_Full36_Teacher_N8','V2_C0_Full36_Teacher8')
    native_cfg=clone('ConfirmC0_I0_O8_s2026112251_Native_r2','V2_C0_Native_State8','qualification8.json',(6,7),
        count=8,qps=20.,correctness_only=True,output_cap=16,ignore_eos=True,state_capture=True,gpu_state_digest=True,serial=True)
    save(native_cfg);jobs.append(dict(axis='Native',cut=0,name=native_cfg['name'],gate='c0_Native_gate_v2.json',signature=signature(native_cfg)))
    token_teachers={}
    for cut in (2,4,8,12):
        token_teachers[cut]=add(f'Q4_TokenTeacher{cut}_N8',f'V2_C0_TokenTeacher{cut}_N8')
        for axis in ('Layer','Token'):
            name=add(f'Q4_{axis}{cut}_N8',f'V2_C0_{axis}{cut}_State8')
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            jobs.append(dict(axis=axis,cut=cut,name=name,teacher=teacher if axis=='Layer' else token_teachers[cut],
                gate=f'c0_{axis}{cut}_gate_v2.json',signature=signature(cfg)))
    write(path,dict(time=time.time(),pair=[6,7],teacher=teacher,token_teachers=token_teachers,jobs=jobs,
        source_capsule=CAPSULE,source_capsule_SHA256=digest(ROOT/'references/source_capsules'/CAPSULE),
        scope='Fresh v2 full valid KV,hidden,greedy and lifetime tests before C0 performance. Actual1/1/1; not a C3 proof.'))
    print('C0_ALL_CUTS_FULL_STATE_PLAN_FROZEN',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('native','c0','official'));args=parser.parse_args();globals()[args.action]()
