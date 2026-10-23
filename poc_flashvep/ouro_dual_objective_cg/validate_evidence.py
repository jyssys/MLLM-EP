"""Check completed evidence, immutable plan locks, and actual device identity."""
import json
from collections import Counter
from .common import ROOT,GPUS,UUIDS,digest,write

def main():
    freeze=json.loads((ROOT/'final_freeze.json').read_text())
    assert digest(ROOT/'objective_shortlist.json')==freeze['objective_sha256']
    assert digest(ROOT/'final_serving_scenarios.json')==freeze['scenario_sha256']
    assert digest(ROOT/'slo_lock.json')==freeze['slo_sha256']
    current=ROOT.parent.parent/'ouro_dual_objective_cg'
    assert digest(current/'adaptive_selector.py')==freeze['selector_runtime_sha256']
    validated=[];pending=[];source_hashes={};failed=[]
    plans=['p0_contention_plan.json','p1_numerical_plan.json','p1_adaptive_smoke_retry_v2_plan.json',
        'p2_objective_validation_plan.json','p2_fixed_tuning_plan.json','p2_alternative_diagnostic_plan.json',
        'p2_event_audit_plan.json','p3_characterization_plan.json','p3_quiet_confirmation_plan.json',
        'p4_quality_smoke_plan.json','p4_quality_plan.json']
    for name in plans:
        plan=json.loads((ROOT/name).read_text());jobs=plan.get('jobs',[j for g in plan.get('groups',[]) for j in g['jobs']])
        for job in jobs:
            assert job['gpu'] in GPUS
            folder=ROOT/'raw'/job['id']
            if not (folder/'completed.json').exists():pending.append(job['id']);continue
            marker=json.loads((folder/'completed.json').read_text())
            for item in marker['artifacts']:
                assert digest(folder/item['path'])==item['sha256'],('Completed artifact hash mismatch',job['id'],item['path'])
            identity=json.loads((folder/'identity.json').read_text())
            assert identity['physical_GPU']==job['gpu']
            assert identity['UUID']==UUIDS[job['gpu']]
            assert identity['CVD']==str(job['gpu'])
            assert identity['scope']=='INDEPENDENT_SINGLE_GPU'
            validated.append(dict(run_id=job['id'],physical_GPU=job['gpu'],UUID=identity['UUID'],completed_hashes_valid=True))
            if name.startswith(('p3_','p4_')):
                sources=json.loads((folder/'dual_sources.json').read_text())
                for file in ('adaptive_selector.py','objective_features.py','objective_families.py'):
                    assert sources[file]['sha256']==digest(current/file),('Frozen selector changed',job['id'],file)
                    source_hashes[file]=sources[file]['sha256']
    for folder in sorted((ROOT/'raw').glob('*')):
        if (folder/'failure.json').exists():failed.append(dict(run_id=folder.name,failure=str((folder/'failure.json').relative_to(ROOT))))
    result=dict(validated_completed_attempts=len(validated),pending_registered_attempts=pending,
        frozen_plan_hashes_valid=True,final_selector_hashes=source_hashes,
        device_counts=dict(Counter(r['physical_GPU'] for r in validated)),failed_preserved=failed,
        all_complete=not pending,scope='Artifact and identity audit. Does not prove numerical or quality equivalence.')
    write(ROOT/'evidence_validation.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
