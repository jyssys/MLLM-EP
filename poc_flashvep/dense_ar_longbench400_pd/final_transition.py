"""Freeze only after declared discovery,then checkpoint final paired starts."""
import json
import os
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,preserve,dispatch_allowed


def pending():
    missing=[]
    for name in ('remote_family_tuning_lane0_complete_v1','remote_family_tuning_lane1_complete_v1',
            'compound_credit_cap_discovery_complete_v1','final_candidate_prequalification_complete_v1',
            'final_native_prequalification_complete_v1','final_credit_native_prequalification_complete_v1',
            'completion_guard_discovery_lane0_complete_v1'):
        if not (ROOT/'state'/(name+'.json')).exists():missing.append(name)
    for axis in ('Native','Layer','Token','Hybrid'):
        name=f'source_credit_sensitivity_{axis}_complete_v1'
        if not (ROOT/'state'/(name+'.json')).exists():missing.append(name)
    manifest=ROOT/'jobs/completion_guard_discovery_q2_8_GPU23_v2.json'
    plans=[manifest]
    for lane in range(4):
        path=ROOT/'configs'/f'decode_concurrency_discovery_lane{lane}_v1.json'
        if not path.exists():
            missing.append(path.name)
            continue
        for action,cap in json.loads(path.read_text())['cases']:
            for qps in (1.8,2.8):
                name=f'DcapTune_Discovery200_q{qps}_{action["label"]}_Cap{cap}_v1'
                receipt=ROOT/'raw'/name/'dispatcher_result.json'
                if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
                    missing.append(name)
    for plan in plans:
        for name in json.loads(plan.read_text())['jobs']:
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
                missing.append(name)
    return missing


def call(module,*args):
    subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.'+module,*args],
        cwd=REPO,check=True)


def main():
    while True:
        assert dispatch_allowed()
        missing=pending()
        write(ROOT/'state/final_transition_barrier_v1.json',dict(time=time.time(),pending=missing,
            evidence='OPERATIONAL_CHECKPOINT',scope='No final latency starts before declared discovery and source proofs finish.'))
        if not missing:break
        time.sleep(30)
    preserve()
    choice=ROOT/'configs/final_continuous_static_choices_frozen_v1.json'
    if not choice.exists():call('final_continuous_selection','--freeze')
    plan_path=ROOT/'configs/final_continuous_execution_plan_v1.json'
    if not plan_path.exists():call('final_confirmation_jobs')
    children=[]
    for phase in ('qualify','performance'):
        for lane in range(4):
            done=ROOT/'state'/f'final_{"qualification" if phase=="qualify" else "performance"}_lane{lane}_complete_v1.json'
            if done.exists():continue
            label=f'final_{phase}_lane{lane}_v1'
            receipt=ROOT/'state'/('background_'+label+'.json')
            if receipt.exists():
                pid=json.loads(receipt.read_text())['pid']
                try:os.kill(pid,0)
                except ProcessLookupError:
                    raise RuntimeError('Final controller stopped without completion. Inspect '+str(receipt))
                continue
            call('launch',label,'poc_flashvep.dense_ar_longbench400_pd.final_confirmation_execute',phase,str(lane))
            children.append(label)
    write(ROOT/'state/final_confirmation_dispatch_started_v1.json',dict(time=time.time(),children=children,
        plan=str(plan_path.relative_to(ROOT)),scope='Controllers require their composition proof before each performance start.'))
    while True:
        call('final_confirmation_analysis')
        complete=all((ROOT/'state'/f'final_performance_lane{lane}_complete_v1.json').exists() for lane in range(4))
        if complete:break
        time.sleep(30)
    write(ROOT/'state/final_transition_complete_v1.json',dict(time=time.time(),complete=True,
        scope='Final timings drained. Root review,all remaining evidence,figures,CPU tests,resource closeout and publication remain.'))


if __name__=='__main__':
    main()
