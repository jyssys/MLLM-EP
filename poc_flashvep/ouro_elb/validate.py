"""Audit immutable evidence and actual quiet-lane overlap after workers stop."""
import csv
import json
from .common import ROOT,UUIDS,UPSTREAM,write,digest
from .dispatch import completed

def main():
    results=[];failures=[];runtime_hashes={};jobs={}
    for path in ROOT.glob('*plan.json'):
        plan=json.loads(path.read_text())
        for job in plan.get('jobs',[])+[j for g in plan.get('groups',[]) for j in g['jobs']]:
            if job['id'] in jobs:assert jobs[job['id']]==job,('CHANGED_JOB',job['id'])
            jobs[job['id']]=job
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        marker=folder/'completed.json';failure=folder/'failure.json'
        if failure.exists():failures.append(dict(run_id=folder.name,**json.loads(failure.read_text())))
        if not marker.exists():continue
        assert completed(dict(id=folder.name))
        who=json.loads((folder/'identity.json').read_text());gpu=who['gpu']
        assert gpu in UUIDS and who['UUID']==UUIDS[gpu] and who['CVD']==str(gpu)
        assert who['runtime_pin']==UPSTREAM
        assert who['job']['gpu']==gpu
        runtime_hashes.setdefault(who['runtime_diff_sha256'],[]).append(folder.name)
        results.append(dict(run_id=folder.name,gpu=gpu,artifact_hashes_valid=True,UUID_valid=True,
            runtime_pin=who['runtime_pin'],runtime_diff_sha256=who['runtime_diff_sha256']))
    matrix=[]
    for rid,job in sorted(jobs.items()):
        folder=ROOT/'raw'/rid
        status='PASS' if (folder/'completed.json').exists() else 'FAILED' if (folder/'failure.json').exists() else 'RUNNING_OR_INTERRUPTED' if (folder/'identity.json').exists() else 'NOT_RUN'
        matrix.append(dict(run_id=rid,gpu=job['gpu'],kind=job['kind'],status=status,
            shapes=json.dumps(job.get('shapes',job.get('cells',[]))),D=json.dumps(job.get('groups',job.get('D'))),
            asynchronous=job.get('async'),graphs=job.get('graphs'),mode=job.get('mode','refill'),
            lane=job.get('lane'),fairness=job.get('fairness','stock')))
    with (ROOT/'experiment_matrix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
    quiet=[];all_windows=[]
    for folder in (ROOT/'raw').iterdir():
        if not folder.is_dir() or not (folder/'completed.json').exists():continue
        for path in folder.glob('timing*.json'):
            r=json.loads(path.read_text());all_windows.append(dict(run=folder.name,gpu=r['gpu'],start=r['wall_start'],end=r['wall_end'],lane=r['lane']))
    for r in all_windows:
        if 'QUIET' not in r['lane']:continue
        overlapping=[x for x in all_windows if x['gpu']!=r['gpu'] and min(x['end'],r['end'])>max(x['start'],r['start'])]
        assert not overlapping,('QUIET_COEXECUTION',r,overlapping)
        quiet.append(r)
    write(ROOT/'evidence_validation.json',dict(completed_attempts=len(results),completed=results,failures=failures,
        runtime_patch_versions=runtime_hashes,quiet_measured_windows=len(quiet),quiet_cross_GPU_timing_overlap=0,
        telemetry_scope='Mutable dispatcher channel. Final raw_evidence_manifest.json hashes it after process exit.',
        limitations='No measurement-window overlap does not imply zero startup or system background load. Resource/telemetry logs remain separate.'))
    print(json.dumps(dict(completed=len(results),failures=len(failures),quiet_windows=len(quiet),jobs=len(matrix))))

if __name__=='__main__':main()
