"""Snapshot actual jobs. Planned code is never counted as measured work."""
import json
import argparse
import subprocess
import time
from .common import ROOT,write
from .resources import processes,burners

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--complete',action='store_true');args=parser.parse_args()
    groups={}
    for name in ('p0_contention_plan.json','p1_numerical_plan.json','p1_observer_control_plan.json','p1_fairness_plan.json',
            'p2_service_plan.json','p2_paired_service_plan.json','p2_controlled_service_plan.json','p1_dynamic_frontier_plan.json','p3_budget_pilot_plan.json',
            'p5_quiet_confirmation_plan.json','p6_quality_plan.json'):
        if not (ROOT/name).exists():
            continue
        plan=json.loads((ROOT/name).read_text())
        jobs=plan.get('jobs',[j for g in plan.get('groups',[]) for j in g['jobs']])
        counts=dict(planned=len(jobs),complete=0,failed=0,started=0)
        for job in jobs:
            folder=ROOT/'raw'/job['id']
            counts['complete']+=(folder/'completed.json').exists()
            counts['failed']+=(folder/'failure.json').exists()
            counts['started']+=(folder/'identity.json').exists()
        groups[name]=counts
    owned=burners();active=processes()
    phase=0 if groups['p0_contention_plan.json']['complete']<64 else 1
    if groups.get('p1_fairness_plan.json',{}).get('complete')==8:
        phase=2
    if (ROOT/'cost_model_artifacts/freeze_v1.json').exists():
        phase=3
    if groups.get('p3_budget_pilot_plan.json',{}).get('started',0):
        phase=5
    quality=groups.get('p6_quality_plan.json',{})
    quality_status='NOT_RUN'
    if quality.get('started',0):
        phase=6;quality_status='RUNNING'
    if quality.get('complete',0) and quality['complete']==quality['planned']:
        quality_status='COMPLETE_UNGRADED'
    if (ROOT/'quality_summary.csv').exists():
        quality_status='MEASURED_PUBLIC_PILOT_NOT_EQUIVALENCE_PROOF'
    if args.complete:
        assert groups['p3_budget_pilot_plan.json']['complete']==32
        assert groups['p5_quiet_confirmation_plan.json']['complete']==12
        assert quality.get('complete')==8 and (ROOT/'quality_summary.csv').exists()
    status='COMPLETE_PHASED_NEGATIVE_DECISION' if args.complete else 'RUNNING'
    state=dict(updated_wall_s=time.time(),status=status,phase=phase,queues=groups,owned_burner_GPUs=sorted(owned),active_GPU_processes=active,
        last_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        online_joint='MEASURED_TRAIN_AND_QUIET' if groups.get('p5_quiet_confirmation_plan.json',{}).get('complete')==12 else 'RUNNING' if phase==5 else 'NOT_RUN',
        capacity='NOT_RUN',quality=quality_status,
        next='No automatic further experiments. Read analysis.md and preserve raw evidence.' if args.complete else 'Read current coordinator states before resuming incomplete queues.',
        verdict='NO_GO_INCREMENTAL' if args.complete else 'PENDING',
        paused_scope='Native-vLLM Loop PP remains paused')
    write(ROOT/'research_state.json',state)
    lines=['# Compute-budget campaign progress','',f'Phase: {phase}. Status: {status}.',
        'These counts come from immutable attempt completion markers.','',
        '| Queue | Planned | Started | Complete | Failed |','| --- | --- | --- | --- | --- |']
    for name,c in groups.items():
        lines.append(f'| {name} | {c["planned"]} | {c["started"]} | {c["complete"]} | {c["failed"]} |')
    lines+=['',f'Source HEAD when this snapshot was prepared: `{state["last_commit"]}`.',
        'This is not a self-referential final published commit identifier.',f'Recognized owned burner GPUs: {sorted(owned)}.',
        'During quiet contention controls, otherwise idle GPUs are reserved measurement resources.',
        'Other idle periods use owned burners. Unrelated GPU jobs remain untouched.','',
        f'Online Joint: {state["online_joint"]}. Qualified capacity: NOT_RUN. Quality: {quality_status}.',
        'Sixteen CPU tests passed before GPU qualification.',
        'These tests cover native software fixtures and policy definitions, not pretrained quality.',
        'The paused native-vLLM PP program remains excluded.']
    (ROOT/'progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups))

if __name__=='__main__':
    main()
