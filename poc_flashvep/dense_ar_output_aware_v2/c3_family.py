"""Qualify the prospective C3 tuning space before any candidate timing."""
import argparse
import fcntl
import json
import time
from .common import ROOT,REPO,write,digest,initialize,preserve,resources
from .runner import save,execute,signature
from . import prepare

SOURCE='v2_native_decode_scoped_prompt_ops_v4.zip'
PLAN='c3_family_qualification_v4.json'

def candidate(arm,name,dataset,pair=(0,1),cut=8,fraction=.125,credit=None,cap=2,correctness=True):
    prepare.CAPSULE=SOURCE
    cfg=prepare.c3(arm,name,dataset,pair=pair,cut=cut,qps=3.2,seed=2026112251,
        correctness=correctness,p_compiled=arm in ('Layer','Hybrid'),cap=cap,fraction=fraction)
    if arm in ('Layer','Hybrid'):
        cfg.update(P_compiled_custom_ops_all=True,D_compiled_custom_ops_all=False,D_partial_IR_norm_priority=False,
            D_scoped_prompt_custom_ops=True)
    if arm=='StockToken':cfg['continuous_source_credit_limit']=0 if credit is None else credit
    elif credit is not None:cfg['continuous_source_credit_limit']=credit
    return cfg

def prepare_plan():
    initialize();path=ROOT/'configs'/PLAN;assert not path.exists()
    for axis in ('Layer','Hybrid'):
        proof=json.loads((ROOT/'state'/f'compiler_control_decode_scope_v4_{axis}_complete.json').read_text());assert proof['passed']
    actions=[]
    for cut in (2,4,8,12):
        actions.append(dict(label=f'Layer{cut}',arm='Layer',cut=cut))
        actions.append(dict(label=f'StockToken{cut}',arm='StockToken',cut=cut))
    for cut,fraction in ((2,.125),(4,.125),(8,.125),(4,.25)):
        actions.append(dict(label=f'Hybrid{cut}_f{str(fraction).replace(".","")}',arm='Hybrid',cut=cut,fraction=fraction))
    actions += [dict(label='NativeStrong',arm='NativeStrong'),dict(label='NativeCommon2',arm='NativeCommon'),
        dict(label='Agg',arm='Agg'),dict(label='Deflection',arm='Deflection'),dict(label='StockToken8Credit8',arm='StockToken',cut=8,credit=8)]
    # Baselines lead their lanes so capacity pilots can start early.
    actions.sort(key=lambda a:(a['arm'] not in ('NativeStrong','NativeCommon','Agg'),a['label']))
    lanes=[[],[]];items=[]
    for index,action in enumerate(actions):
        lane=index%2;pair=(2*lane,2*lane+1);label=action['label'];arm=action['arm'];cut=action.get('cut',8);fraction=action.get('fraction',.125)
        strict=None;teacher=None
        if arm in ('Layer','Hybrid'):
            base=f'V2_PCompile_{arm}_Compiled_N8_DecodeScopeV4'
            strict_cfg=json.loads((ROOT/'configs'/(base+'.json')).read_text())
            strict=f'V2_C3Q_{label}_Strict8'
            strict_cfg.update(name=strict,gpus=list(pair),axis_cut=cut,source_capsule=SOURCE,
                source_capsule_SHA256=digest(ROOT/'references/source_capsules'/SOURCE))
            teacher=f'V2_DecodeScope_{arm}_OriginalTeacher8'
            if arm=='Hybrid':
                strict_cfg['hybrid_suffix_fraction']=fraction
                if fraction!=.125:
                    old=json.loads((ROOT/'configs'/(teacher+'.json')).read_text());teacher=f'V2_C3Q_{label}_OriginalTeacher8'
                    old.update(name=teacher,gpus=list(pair),teacher_suffix_fraction=fraction);save(old)
            save(strict_cfg)
        elif arm=='StockToken':
            base=json.loads((ROOT/'configs/V2_DecodeScope_Layer_OriginalTeacher8.json').read_text())
            teacher=f'V2_C3Q_StockToken{cut}_OriginalTeacher8'
            for key in list(base):
                if key.startswith('teacher_'):base.pop(key)
            base.update(name=teacher,gpus=list(pair),token_teacher_cut=cut)
            if not (ROOT/'configs'/(teacher+'.json')).exists():save(base)
            strict=f'V2_C3Q_{label}_Strict8'
            cfg=candidate(arm,strict,'compiler_control8_original_v1.json',pair,cut,fraction,action.get('credit'))
            cfg.update(serial=True,controlled_pipeline=True,output_cap=16,ignore_eos=True,count=8,
                gpu_state_digest=False,axis_state_capture=True)
            save(cfg)
        jobs=[]
        for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            name=f'V2_C3Q_{label}_Natural{count}'
            cfg=candidate(arm,name,data,pair,cut,fraction,action.get('credit'));save(cfg);jobs.append(name)
        final=json.loads((ROOT/'configs'/(jobs[-1]+'.json')).read_text());gate=f'c3_composition_{label}_v4.json'
        item=dict(action,lane=lane,pair=list(pair),teacher=teacher,strict=strict,jobs=jobs,gate=gate,signature=signature(final))
        items.append(item);lanes[lane].append(label)
    write(path,dict(time=time.time(),items=items,lanes=lanes,source_capsule=SOURCE,
        source_capsule_SHA256=digest(ROOT/'references/source_capsules'/SOURCE),
        scope='Prospective common-cap2 tuning space. Stock Token uses the original compiled model and includes uncapped source credit0. '
            'Every candidate gets actual state/lifetime and natural-quality tests. Strict teachers are independent Full36 models. '
            'No active-decode prefill insertion is added. C0 results never substitute for C3. '
            'After geometry discovery,tune the same predeclared admission set0/1/2/4 and source-credit set8/16 (plus0 for Stock Token) before heldout choices freeze.'))
    print('C3_STRONG_CANDIDATE_SPACE_FROZEN',len(items),flush=True)

def scheduling_fix():
    original=ROOT/'configs'/PLAN;target=ROOT/'configs/c3_family_qualification_v4_schedule_fix1.json';assert not target.exists()
    plan=json.loads(original.read_text())
    for item in plan['items']:
        if item['arm']!='StockToken':continue
        old=json.loads((ROOT/'configs'/(item['teacher']+'.json')).read_text())
        teacher=f'V2_C3Q_{item["label"]}_LocalOriginalTeacher8_Fix1'
        old.update(name=teacher,gpus=item['pair']);save(old);item['teacher']=teacher
        cfg=json.loads((ROOT/'configs'/(item['strict']+'.json')).read_text());name=item['strict']+'_CoordinatorFix1'
        cfg.update(name=name,continuous_source_credit_limit=0);save(cfg);item['strict']=name
    plan.update(time=time.time(),parent_plan_SHA256=digest(original),
        scheduling_fix='Give each stock-token teacher its action-local pair. Disable only the continuous coordinator window in serial controlled strict tests. '
            'The actual C3 credit0/8 candidates and all datasets,runtime,admission choices and quality/SLO contracts remain unchanged.')
    write(target,plan);print('C3_CPU_SCHEDULE_AND_STRICT_PROTOCOL_PREFLIGHT_FIXED',flush=True)

def run(lane,plan_name=PLAN):
    initialize()
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.qualification_gate import compare as strict_compare
    from ..dense_ar_longbench400_pd.decode_cap_discovery import full_native_state
    from ..dense_ar_longbench400_pd.metrics import summarize
    from .quality_gate import compare as quality
    from .original_arm_audit import audit_agg
    plan=json.loads((ROOT/'configs'/plan_name).read_text());items=[r for r in plan['items'] if r['lane']==lane]
    pair=[2*lane,2*lane+1];owner=resources();results=[];state_prefix=plan.get('state_prefix','c3_family')
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner.stop(pair)
        try:
            for item in items:
                proof=None
                try:
                    strict=None
                    if item['teacher']:
                        owner.stop(pair);assert execute(item['teacher']),item['teacher']
                        owner.stop(pair);assert execute(item['strict']),item['strict'];summarize(item['strict'])
                        coverage=audit(item['strict']);strict=strict_compare(item['teacher'],item['strict'])
                        assert coverage['passed'] and strict['passed'],(item['label'],strict)
                    stages=[]
                    for name in item['jobs']:
                        owner.stop(pair);assert execute(name),name;summarize(name)
                        if item['arm'] in ('Layer','StockToken','Hybrid'):
                            state=audit(name);assert state['passed'] and state['decode_full36_observation_complete']
                            assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
                        elif item['arm']=='Agg':state=audit_agg(name)
                        else:state=full_native_state(name,item['arm']=='Deflection')
                        score=quality(name);assert state['passed'] and score['passed'],score
                        stages.append(dict(name=name,state=state,quality=score))
                    proof=dict(time=time.time(),passed=True,full400_passed=True,quality_passed=True,strict=strict,stages=stages,
                        signature=item['signature'],source_capsule_SHA256=item.get('source_capsule_SHA256',plan['source_capsule_SHA256']),evidence='CORRECTNESS_ONLY')
                except Exception as error:
                    proof=dict(time=time.time(),passed=False,error=repr(error),label=item['label'],evidence='FAILED')
                write(ROOT/'state'/item['gate'],proof);results.append(dict(label=item['label'],passed=proof['passed'],error=proof.get('error')))
                write(ROOT/'state'/f'{state_prefix}_lane{lane}_progress_v4.json',dict(time=time.time(),results=results))
                print('C3_FAMILY_GATE',results[-1],flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'{state_prefix}_lane{lane}_complete_v4.json',dict(time=time.time(),results=results,complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','scheduling_fix','run'));p.add_argument('--lane',type=int,choices=(0,1));p.add_argument('--plan',default=PLAN);args=p.parse_args()
    if args.action=='prepare':prepare_plan()
    elif args.action=='scheduling_fix':scheduling_fix()
    else:run(args.lane,args.plan)
