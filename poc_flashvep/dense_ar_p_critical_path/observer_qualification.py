"""Qualify live-namespace P events without changing the credit experiment."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,copy_data,bind,preserve,resources
from .capsule import build
from .qualification import differential


def prepare():
    runner=bind();items=[]
    for label,lane in (('Layer',0),('Hybrid',1)):
        parent=json.loads((ROOT/'configs'/f'PC_Metadata_{label}.json').read_text())['source_capsule'].removeprefix('pc_')
        source=build(parent,'r2');jobs=[];pair=[2*lane,2*lane+1]
        original=f'PC_Q_{label}_Original_Strict8_ArrivalFix2'
        strict_cfg=json.loads((ROOT/'configs'/(original+'.json')).read_text())
        strict=f'PC_LiveNamespace_{label}_Strict8_R2'
        strict_cfg.update(name=strict,source_capsule=source.name,source_capsule_SHA256=digest(source),
            P_event_timing=True,P_critical_trace=True,source_credit_release_on_retirement=False)
        runner.save(strict_cfg)
        for count,dataset in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            cfg=json.loads((ROOT/'configs'/f'PC_Q_{label}_Control_Natural{count}.json').read_text())
            name=f'PC_LiveNamespace_{label}_Natural{count}_R2'
            cfg.update(name=name,source_capsule=source.name,source_capsule_SHA256=digest(source),P_event_timing=True)
            runner.save(cfg);jobs.append(name)
        items.append(dict(label=label,lane=lane,pair=pair,original=original,strict=strict,jobs=jobs,
            gate=f'pc_{label.lower()}_live_observer_composition_r2.json',signature=runner.signature(cfg),
            source_capsule=source.name,source_capsule_SHA256=digest(source)))
    path=ROOT/'configs/live_namespace_observer_qualification_r2.json';assert not path.exists()
    write(path,dict(time=time.time(),items=items,
        scope='R1 event profiles were stopped before GPU dispatch. Main retirement A/B has events disabled and is unaffected. '
            'R2 uses the live original module namespace. Require actual NONE dispatch for every partial P forward. '
            'Fresh N8 exact state,N32/N400 coverage,lifetime and task quality precede any new R2 CPC timing.'))
    children=[]
    for item in items:
        log=ROOT/'logs'/f'live_observer_qualification_lane{item["lane"]}.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.observer_qualification',
                'run',str(item['lane'])],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=item['lane'],pid=child.pid))
    write(ROOT/'state/live_observer_qualification_controller.json',dict(time=time.time(),children=children))
    return children


def run(lane):
    runner=bind();owner=resources();pair=[2*lane,2*lane+1]
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .quality import compare
    item=next(v for v in json.loads((ROOT/'configs/live_namespace_observer_qualification_r2.json').read_text())['items'] if v['lane']==lane)
    proof=None
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner.stop(pair)
        try:
            assert runner.execute(item['strict']),item['strict'];exact=differential(item['original'],item['strict'])
            strict_state=audit(item['strict']);assert strict_state['passed']
            stages=[]
            for name in item['jobs']:
                owner.stop(pair);assert runner.execute(name),name;summarize(name)
                state=audit(name);score=compare(name)
                assert state['passed'] and state['decode_full36_observation_complete'] and score['passed'],(state,score)
                event=json.loads((ROOT/'raw'/name/'P_model_forward_events_prefill.json').read_text())
                assert event['rows'] and all(v['mode']=='NONE' for v in event['rows']),('Partial P graph guard changed',name)
                stages.append(dict(name=name,state=state,quality=score,all_partial_P_dispatch_NONE=True))
            proof=dict(time=time.time(),passed=True,full400_passed=True,quality_passed=True,strict=exact,stages=stages,
                signature=item['signature'],source_capsule_SHA256=item['source_capsule_SHA256'],evidence='CORRECTNESS_ONLY')
        except Exception as error:proof=dict(time=time.time(),passed=False,error=repr(error),evidence='FAILED')
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/item['gate'],proof);print('PC_LIVE_OBSERVER_GATE',item['label'],proof['passed'],proof.get('error'),flush=True)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[2])),flush=True)
