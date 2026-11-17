"""Inventory completed, failed and unstarted experiments without deleting evidence."""
import collections
import json
import subprocess
import time
from .common import ROOT,REPO,SOURCE,digest,preserve,write


def main():
    preserve();rows=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or cfg.get('name')!=path.stem:
            continue
        folder=ROOT/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json'
        result=json.loads(receipt.read_text()) if receipt.exists() else None
        attempted=folder.exists()
        status='MEASURED' if result and result.get('complete') else 'FAILED' if result or attempted else 'NOT_RUN'
        if status=='MEASURED' and cfg.get('correctness_only'):
            status='CORRECTNESS_ONLY'
        rows.append(dict(name=cfg['name'],status=status,phase=cfg.get('phase'),
            method=cfg.get('evaluation_arm'),count=cfg.get('count'),cap=cfg.get('cap'),
            QPS=cfg.get('arrival_rate'),seed=cfg.get('arrival_seed'),dataset=cfg.get('dataset'),
            configuration_SHA256=digest(path),receipt=result,
            incomplete_attempt_without_dispatcher_receipt=attempted and result is None,
            command=str((folder/'command.json').relative_to(ROOT)) if (folder/'command.json').exists() else None,
            immutable_source=str((folder/'source.zip').relative_to(ROOT)) if (folder/'source.zip').exists() else None))
    prior=json.loads((ROOT/'state/frozen_predecessor_hashes.json').read_text())
    changes=[name for name,expected in prior.items() if not (REPO/name).exists() or digest(REPO/name)!=expected]
    assert not changes,'Predecessor source changed: '+str(changes)
    dependencies=json.loads((ROOT/'state/additional_worker_dependencies_vs_base.json').read_text())['files']
    assert all(r['status']=='SAME' and digest(REPO/r['path'])==r['current_SHA256'] for r in dependencies)
    tests=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m','unittest',
        *['poc_flashvep.dense_ar_layer_token_pd.'+name for name in
          ('test_contracts','test_replay','test_credit_fast_lane','test_flow_policy','test_geometry_control','test_statistics')]],
        cwd=REPO,capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    write(ROOT/'analysis/final_CPU_verification.json',dict(time=time.time(),returncode=tests.returncode,
        stdout=tests.stdout,stderr=tests.stderr,predecessor_files=len(prior),predecessor_changes=changes,
        additional_worker_dependencies=len(dependencies),additional_dependency_changes=[],
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.glob('*.py')},
        evidence='CORRECTNESS_ONLY',scope='CPU contract tests do not replace GPU model-state correctness.'))
    write(ROOT/'analysis/experiment_inventory.json',dict(time=time.time(),schema_version=2,runs=rows,
        counts=dict(collections.Counter(r['status'] for r in rows)),
        recorded_dispatcher_wall_seconds=sum(r['receipt'].get('elapsed_seconds',0) for r in rows if r['receipt']),
        scope='Preserve every configuration and dispatcher receipt. A partial raw directory without a final receipt is a failed or incomplete attempt, not a zero-performance result. NOT_RUN means no raw attempt directory. Dispatcher wall time includes startup and host/state capture; it is not exclusive GPU compute. A completed dispatcher does not alone qualify a performance arm. State and output gates remain separate.'))
    print('EXPERIMENT_INVENTORY',collections.Counter(r['status'] for r in rows),flush=True)


if __name__=='__main__':
    main()
