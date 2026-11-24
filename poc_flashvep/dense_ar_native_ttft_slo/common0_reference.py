"""Qualify the exact uncapped common-D Native reference before timing."""
import argparse
import fcntl
import json
import subprocess
import time
import traceback
from .common import ROOT,REPO,PYTHON,write,digest,preserve,resources
from .runtime import bind,dispatch_allowed

PLAN='native_common0_reference_addendum_v1.json'
GATE='ns_nativecommon0_composition_v1.json'
COMPLETE='native_common0_reference_addendum_complete_v1.json'


def prepare():
    runner=bind()
    from .qualification import capture,save_on_data
    plan=json.loads((ROOT/'configs/root_cause_policy_qualification_v1.json').read_text())
    parent=next(row for row in plan['items'] if row['label']=='NativeCommon2')
    cfg=json.loads((ROOT/'configs'/(parent['steps']['Natural400']+'.json')).read_text())
    cfg['D_running_admission_cap']=0;steps={}
    for stage,count in (('Single1',1),('Prime8',8),('Strict8A',8),('Strict8B',8)):
        name=f'NS_Qual_NativeCommon0_{stage}'
        steps[stage]=save_on_data(runner,capture(dict(cfg)),name,'compiler_control8_original_v1.json',count,[0,1],True,16)
    for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
        name=f'NS_Qual_NativeCommon0_Natural{count}'
        value=capture(dict(cfg));value['axis_capture_hidden_last_only']=True
        steps[f'Natural{count}']=save_on_data(runner,value,name,data,count,[0,1],False,None)
    final=json.loads((ROOT/'configs'/(steps['Natural400']+'.json')).read_text())
    signature=runner.signature(final);diff={key:(parent['signature'].get(key),signature.get(key))
        for key in set(signature)|set(parent['signature']) if signature.get(key)!=parent['signature'].get(key)}
    assert set(diff)=={'D_running_admission_cap'},diff
    item=dict(label='NativeCommon0',lane=0,pair=[0,1],steps=steps,gate=GATE,signature=signature,
        source_capsule_SHA256=final['source_capsule_SHA256'])
    target=ROOT/'configs'/PLAN;assert not target.exists()
    write(target,dict(time=time.time(),item=item,parent=parent,changed_signature_fields=diff,
        scope='Prospective exact common-D reference for best Stock Cap0,not a weaker replacement for NativeStrong. '
            'Wait for all initial qualification GPU jobs. Complete this addendum before serial profiling and any new performance measurement. '
            'Keep Main400 data,SLO,compiled model,source bytes and READ/registration deferral fixed. Only D cap differs from Common2. '
            'The initially unlaunched timing plans remain preserved. No GPU experiment is stopped or restarted.'))
    log=ROOT/'logs/native_common0_reference_addendum_v1.log'
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.common0_reference','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/native_common0_reference_controller_v1.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    bind()
    from ..dense_ar_p_critical_path.stable_source import install
    from ..dense_ar_p_critical_path.warm_serving_followup import expected_roots,check_reuse
    from ..dense_ar_p_critical_path.quality import compare as quality
    from .qualification import state_audit,strict_compare
    runner=install();plan=json.loads((ROOT/'configs'/PLAN).read_text());item=plan['item'];owner=resources()
    while not all((ROOT/'state'/f'policy_qualification_lane{lane}_complete_v1.json').exists() for lane in range(4)):
        if not dispatch_allowed():break
        time.sleep(10)
    proof=dict(time=time.time(),passed=False,label='NativeCommon0',signature=item['signature'],
        source_capsule_SHA256=item['source_capsule_SHA256'],stages=[],evidence='CORRECTNESS_ONLY')
    with (ROOT/'state/pair_0_1.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            assert json.loads((ROOT/'state'/plan['parent']['gate']).read_text())['passed']
            def execute(name):
                assert dispatch_allowed() and not (ROOT/'raw'/name).exists()
                owner.stop([0,1]);assert runner.execute(name),name
                state=state_audit(name);assert state['passed']
                proof['stages'].append(dict(name=name,state=state));return state
            for stage in ('Single1','Prime8'):execute(item['steps'][stage])
            roots=expected_roots(item['steps']['Prime8']);assert set(roots)=={'prefill','decode'}
            for stage in ('Strict8A','Strict8B'):
                name=item['steps'][stage];execute(name);check_reuse(name,roots)
            proof['warm_strict']=strict_compare(item['steps']['Strict8A'],item['steps']['Strict8B'],False)
            assert proof['warm_strict']['passed']
            proof['one_factor_strict']=strict_compare(plan['parent']['steps']['Strict8B'],item['steps']['Strict8B'],False)
            assert proof['one_factor_strict']['passed']
            for stage in ('Natural32','Natural400'):
                name=item['steps'][stage];execute(name);reuse=check_reuse(name,roots);score=quality(name)
                proof['stages'][-1].update(cache_reuse=reuse,quality=score);assert score['passed']
            proof.update(passed=True,full400_passed=True,quality_passed=True,official_cache_roots=roots)
        except Exception as error:proof.update(error=repr(error),traceback=traceback.format_exc(),evidence='FAILED')
        finally:owner.restore([0,1]);preserve()
    write(ROOT/'state'/GATE,proof)
    write(ROOT/'state'/COMPLETE,dict(time=time.time(),complete=True,passed=proof['passed'],error=proof.get('error')))
    print('NS_COMMON0_REFERENCE',proof['passed'],proof.get('error'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    print(prepare() if args.action=='prepare' else run(),flush=True)
