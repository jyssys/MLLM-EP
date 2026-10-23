"""Snapshot actual jobs. Planned code is never counted as measured work."""
import json
import subprocess
import time
from .common import ROOT,write
from .resources import processes,burners

def main():
    groups={}
    for name in ('p0_contention_plan.json','p1_numerical_plan.json','p1_fairness_plan.json','p2_service_plan.json','p2_paired_service_plan.json'):
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
    state=dict(updated_wall_s=time.time(),status='RUNNING',phase=phase,queues=groups,owned_burner_GPUs=sorted(owned),active_GPU_processes=active,
        last_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        online_joint='NOT_RUN',capacity='NOT_RUN',quality='New policy quality NOT_RUN',
        next='Review pretrained observer/frontier qualification. Then fairness and fresh service costs.',
        paused_scope='Native-vLLM Loop PP remains paused')
    write(ROOT/'research_state.json',state)
    lines=['# Compute-budget campaign progress','',f'Phase: {phase}. Status: RUNNING.',
        'These counts come from immutable attempt completion markers.','',
        '| Queue | Planned | Started | Complete | Failed |','| --- | --- | --- | --- | --- |']
    for name,c in groups.items():
        lines.append(f'| {name} | {c["planned"]} | {c["started"]} | {c["complete"]} | {c["failed"]} |')
    lines+=['',f'Last commit: `{state["last_commit"]}`.',f'Recognized owned burner GPUs: {sorted(owned)}.',
        'During quiet contention controls, otherwise idle GPUs are reserved measurement resources.',
        'Other idle periods use owned burners. Unrelated GPU jobs remain untouched.','',
        'New online Joint, qualified capacity, and new adaptive quality are NOT_RUN.',
        'Sixteen CPU tests passed before GPU qualification.',
        'These tests cover native software fixtures and policy definitions, not pretrained quality.',
        'The paused native-vLLM PP program remains excluded.']
    (ROOT/'progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(groups))

if __name__=='__main__':
    main()
