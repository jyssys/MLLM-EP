"""Generate an honest campaign snapshot from completed evidence only."""
import csv
import json
import time
from .common import ROOT,write

def main():
    jobs={}
    for path in sorted((ROOT/'jobs').glob('*.json')):
        job=json.loads(path.read_text())
        target=ROOT/'raw'/job['id']
        if (target/'completed.json').exists():
            status='COMPLETED'
        elif (target/'failure.json').exists():
            status='FAILED'
        else:
            status='RUNNING_OR_INTERRUPTED'
        jobs[job['id']]=dict(id=job['id'],kind=job['kind'],gpu=job['gpu'],split=job.get('split'),status=status,job=job)
    for plan_name in ['p0_contention_plan.json','p1_diagnostic_plan.json','p1_quality_plan.json','p2_surface_plan.json']:
        path=ROOT/plan_name
        if not path.exists():
            continue
        plan=json.loads(path.read_text())
        planned=plan.get('jobs',[j for g in plan.get('groups',[]) for j in g['jobs']])
        for job in planned:
            jobs.setdefault(job['id'],dict(id=job['id'],kind=job['kind'],gpu=job['gpu'],split=job.get('split'),status='PENDING',job=job))
    fields=['id','kind','gpu','split','status']
    with (ROOT/'experiment_matrix.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in jobs.values())
    counts={}
    for row in jobs.values():
        counts[row['status']]=counts.get(row['status'],0)+1
    p0_complete=sum(r['status']=='COMPLETED' and r['kind']=='calibration' for r in jobs.values())
    p1_active=any(r['id'].startswith('p1_') and r['status']!='PENDING' for r in jobs.values())
    p2_active=any(r['id'].startswith('p2_') and r['status']!='PENDING' for r in jobs.values())
    fitted=(ROOT/'predictor_blind_results.json').exists()
    current='Phase3/4 predictor and offline headroom review' if fitted else 'Phase2 exploratory cost characterization' if p2_active else 'Phase1 numerical and root-cause diagnostics' if p1_active else 'Phase0 contention controls'
    ledger=[json.loads(line) for line in (ROOT/'phase_exit_decisions.jsonl').read_text().splitlines()] if (ROOT/'phase_exit_decisions.jsonl').exists() else []
    write(ROOT/'research_state.json',dict(updated_wall_s=time.time(),status='RUNNING',job_counts=counts,
        current_primary=current,completed_P0_jobs=p0_complete,phase_exit_ledger=ledger,
        next_gate='Resolve root-cause evidence and review interference limits before adaptive implementation',
        method_claim='NOT_ESTABLISHED',quality='114_CASE_FIXED_POLICY_PILOT_COMPLETE' if (ROOT/'quality_summary.csv').exists() else 'NOT_RUN',online_adaptive='NOT_RUN',capacity='NOT_RUN',
        safety='Quiet-control devices are reserved measurement resources. Other idle devices receive owned burners.',
        source_code='poc_flashvep/ouro_adaptive_chunk_loop'))
    print(counts)

if __name__=='__main__':
    main()
