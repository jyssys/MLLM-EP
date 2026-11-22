"""Test the P compiler against fresh unchanged controls and legal teachers."""
import argparse
import fcntl
import json
import time
from .common import ROOT,PREVIOUS,write,digest,initialize,preserve,resources,dispatch_allowed
from .runner import save,execute

PLAN='compiler_prefix_control_v1.json'

def prepare():
    initialize();assert not (ROOT/'configs'/PLAN).exists()
    data='compiler_control8_original_v1.json'
    write(ROOT/'data'/data,json.loads((PREVIOUS/'data/qualification8.json').read_text()))
    baseplan=json.loads((PREVIOUS/'configs/eager_prefill_native_decode_correctness_frozen_v2.json').read_text())
    capsule='v2_p_compiled_prefix_experiment_v1.zip';sha=digest(ROOT/'references/source_capsules'/capsule)
    actions={}
    for axis,lane in (('Layer',0),('Hybrid',2)):
        source=baseplan['qualification'][axis]
        def clone(old,name,**extra):
            cfg=json.loads((PREVIOUS/'configs'/(old+'.json')).read_text())
            cfg={k:v for k,v in cfg.items() if 'gate' not in k}
            cfg.update(name=name,dataset=data,dataset_SHA256=digest(ROOT/'data'/data),gpus=[lane*2,lane*2+1],
                source_capsule=capsule,source_capsule_SHA256=sha,correctness_only=True,
                scope='Original v1 calibration prompts reused only as an explicitly exposed compiler diagnostic. Not v2 Main serving.',**extra)
            return save(cfg)
        teacher=clone(source['teacher'],f'V2_PCompile_{axis}_Teacher8')
        jobs=[]
        for experimental in (False,True):
            for count in (1,8):
                old=source['jobs'][0 if count==1 else 1]
                name=f'V2_PCompile_{axis}_{"Compiled" if experimental else "Control"}_N{count}'
                extra=dict(P_compile_prefix=True,preserve_compiled_full_model=True) if experimental else {}
                clone(old,name,**extra);jobs.append(dict(name=name,count=count,experimental=experimental))
        actions[axis]=dict(lane=lane,teacher=teacher,jobs=jobs)
    write(ROOT/'configs'/PLAN,dict(time=time.time(),source_capsule=capsule,source_capsule_SHA256=sha,actions=actions,
        design='Same original requests,cut,output16,BI0,geometry and D policy. P compiler switch only. Fresh original Full36 stage teachers and unchanged P controls precede the intervention. Stop that branch on the first strict failure.'))
    print('P_COMPILE_CONTROL_FROZEN',flush=True)

def run(axis,plan_name=PLAN):
    initialize()
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.quiescent_admission_qualification import strict_first
    from ..dense_ar_longbench400_pd.qualification_gate import compare
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.metrics import summarize
    plan=json.loads((ROOT/'configs'/plan_name).read_text());item=plan['actions'][axis];lane=item['lane'];pair=[2*lane,2*lane+1]
    prefix=plan.get('state_prefix','compiler_control')
    while not (ROOT/'state/guardian.json').exists():assert dispatch_allowed();time.sleep(5)
    owner=resources();results=[];failure=None
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner.stop(pair)
        try:
            assert execute(item['teacher']),item['teacher'];summarize(item['teacher'])
            for job in item['jobs']:
                try:
                    owner.stop(pair);assert execute(job['name']),job['name']
                    summarize(job['name']);coverage=audit(job['name'])
                    assert coverage['passed'] and coverage['decode_full36_observation_complete']
                    value=strict_first(item['teacher'],job['name']) if job['count']==1 else compare(item['teacher'],job['name'])
                    passed=all(value.values()) if job['count']==1 else value['passed']
                    if job['experimental']:
                        events=json.loads((ROOT/'raw'/job['name']/'axis_events_prefill.json').read_text())['rows']
                        compiled=[r for r in events if r['event']=='P_COMPILED_PREFIX_SEGMENT']
                        assert compiled and all(r['compiled'] for r in compiled)
                    results.append(dict(**job,passed=passed,coverage=coverage,numerical=value))
                    assert passed,value
                except Exception as error:
                    failure=dict(**job,error=repr(error),status='FAILED');break
                write(ROOT/'state'/f'{prefix}_{axis}_progress.json',dict(time=time.time(),results=results))
        except Exception as error:
            if failure is None:failure=dict(error=repr(error),status='FAILED',stage='TEACHER_OR_SETUP')
        finally:
            owner.restore(pair);preserve()
    passed=failure is None and len(results)==len(item['jobs'])
    write(ROOT/'state'/f'{prefix}_{axis}_complete.json',dict(time=time.time(),passed=passed,results=results,failure=failure,
        source_capsule_SHA256=plan['source_capsule_SHA256'],full400_passed=False,evidence='CORRECTNESS_ONLY' if passed else 'FAILED',
        scope='N1/N8 strict compiler diagnostic only. Mixed/variable/natural/full400 and decode-tail gates remain required.'))
    print('P_COMPILER_CONTROL_TERMINAL',axis,passed,failure,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--axis',choices=('Layer','Hybrid'));parser.add_argument('--plan',default=PLAN);args=parser.parse_args()
    prepare() if args.action=='prepare' else run(args.axis,args.plan)
