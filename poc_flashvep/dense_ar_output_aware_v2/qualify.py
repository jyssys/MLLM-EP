"""Publish gates only after actual state and task-quality checks pass."""
import argparse
import fcntl
import json
import time
from .common import ROOT,write,initialize,preserve,resources,dispatch_allowed
from .runner import execute

def run(kind):
    initialize()
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.metrics import summarize
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.qualification_gate import compare
    from ..dense_ar_longbench400_pd.decode_cap_discovery import full_native_state
    from ..dense_ar_longbench400_pd.quality_gate import compare as quality
    plan=json.loads((ROOT/'configs'/('native_baseline_qualification_v2.json' if kind=='native' else 'c0_full_state_qualification_v2.json')).read_text())
    pair=plan['pair'];owner=resources();results=[];failure=None
    terminal=ROOT/'state'/f'{kind}_qualification_complete_v2.json'
    if terminal.exists():
        previous=json.loads(terminal.read_text())
        assert not previous.get('passed'),'Do not rerun a completed qualification branch'
        archive=ROOT/'state/controller_failure_history'/f'{kind}_{int(time.time()*1000)}.json'
        write(archive,dict(previous=previous,gates={item['gate']:json.loads((ROOT/'state'/item['gate']).read_text())
            for item in plan['jobs'] if (ROOT/'state'/item['gate']).exists()},
            scope='Preserve the controller API-call failure. Reuse complete immutable GPU receipts. Do not relabel it as a model-state failure.'))
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX);owner.stop(pair)
        try:
            teachers_done=set()
            for item in plan['jobs']:
                try:
                    if kind=='c0' and item.get('teacher') not in teachers_done and item.get('teacher'):
                        name=item['teacher'];owner.stop(pair);assert execute(name),name;summarize(name);teachers_done.add(name)
                    name=item['name'];owner.stop(pair);assert execute(name),name;summarize(name)
                    if item.get('axis') in ('Layer','Token'):
                        state=audit(name);numerical=compare(item['teacher'],name)
                        assert state['passed'] and state['decode_full36_observation_complete'] and numerical['passed']
                        assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
                        score=None
                    else:
                        if item.get('arm')=='Agg':
                            from .original_arm_audit import audit_agg
                            state=audit_agg(name)
                        else:state=full_native_state(name,False)
                        numerical=None;score=quality(name) if kind=='native' else None
                        assert state['passed'] and (score is None or score['passed'])
                    proof=dict(time=time.time(),passed=True,full400_passed=kind=='native',quality_passed=bool(score and score['passed']),
                        name=name,signature=item['signature'],source_capsule_SHA256=plan['source_capsule_SHA256'],
                        state=state,numerical=numerical,quality=score,evidence='CORRECTNESS_ONLY')
                    write(ROOT/'state'/item['gate'],proof);results.append(dict(name=name,passed=True,gate=item['gate']))
                except Exception as error:
                    failure=dict(name=item['name'],error=repr(error),status='FAILED')
                    write(ROOT/'state'/item['gate'],dict(passed=False,failure=failure,evidence='FAILED'));break
                write(ROOT/'state'/f'{kind}_qualification_progress_v2.json',dict(time=time.time(),results=results))
                print('V2_QUALIFICATION_PASS',item['name'],flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'{kind}_qualification_complete_v2.json',dict(time=time.time(),passed=failure is None and len(results)==len(plan['jobs']),results=results,failure=failure))
    print('V2_QUALIFICATION_TERMINAL',kind,failure,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('native','c0'));run(p.parse_args().kind)
