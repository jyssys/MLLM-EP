"""Run bounded falsifiable controls. Each next stage requires actual evidence."""
from .common import *
from .prepare import base,put
from .run import execute


def gate(name,reference=None):
    activate()
    from ..looped_pd_atomic_prefill import evidence
    evidence.ROOT=ROOT
    checks=dict(transfer=evidence.transfer(name),work=evidence.full_work(name))
    if reference:checks['comparison']=evidence.compare(reference,name)
    checks['passed']=all(value['passed'] for value in checks.values())
    write(ROOT/'raw'/name/'recovery_gate.json',checks)
    return checks['passed']


def first_controls():
    results=[]
    for model in ('ouro','huginn'):
        operator=ROOT/'raw'/('operator_'+model+'_g1024')/'operator_summary.json'
        if not operator.exists() or not json.loads(operator.read_text())['all_exact']:
            results.append(dict(model=model,status='BLOCKED_OPERATOR_CORRECTNESS'));continue
        names=[]
        for label,rows in (('baseline',0),('graph1024',1024)):
            c=base(model,f'parity_{model}_{label}_gate8',phase='correctness',
                verify_transfer=True,state_probe=True,projection_graph_rows=rows)
            if (ROOT/'raw'/c['name']/'user_pause.json').exists():
                c['name'] += '_resume1'
            name=put(c,8)
            write(ROOT/'state/campaign.json',dict(stage='QUALIFIED_REQUEST_GATE',model=model,name=name,time=time.time()))
            okay=execute(name) and gate(name,names[0] if names else None)
            results.append(dict(model=model,label=label,gate=okay,name=name))
            write(ROOT/'state/first_controls.json',results)
            if not okay:break
            names.append(name)
        if len(names)!=2:continue
        for label,rows in (('baseline',0),('graph1024',1024)):
            c=base(model,f'pilot_{model}_{label}_n24',phase='qps_pilot',projection_graph_rows=rows)
            name=put(c,24,2.5 if model=='ouro' else 1.0)
            write(ROOT/'state/campaign.json',dict(stage='MATCHED_CLEAN_PILOT',model=model,name=name,time=time.time()))
            okay=execute(name)
            if okay:
                activate()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                work=evidence.full_work(name)
                write(ROOT/'raw'/name/'full_work_check.json',work)
                okay=work['passed']
                if label!='baseline':
                    comparison=evidence.compare(f'pilot_{model}_baseline_n24',name)
                    write(ROOT/'raw'/name/'matched_output_parity.json',comparison)
                    okay=okay and comparison['passed']
            results.append(dict(model=model,label=label,pilot=okay,name=name))
            write(ROOT/'state/first_controls.json',results)
            if not okay:break
    write(ROOT/'state/campaign.json',dict(stage='FIRST_CONTROLS_COMPLETE_REVIEW_REQUIRED',time=time.time(),results=results))


if __name__=='__main__':
    initialize()
    try:first_controls()
    except BaseException as error:
        write(ROOT/'state/campaign.json',dict(stage='CONTROLLER_FAILURE_REVIEW_REQUIRED',time=time.time(),error=repr(error)));raise
