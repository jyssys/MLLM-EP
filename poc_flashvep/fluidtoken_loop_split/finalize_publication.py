"""Finish publication and resource verification after the evidence batches."""
import time
from .common import *


def main():
    initialize()
    target = ROOT / 'state/final_closeout.json'
    receipt_path = ROOT / 'state/publication_receipt.json'
    while time.time() < DEADLINE - 300:
        if receipt_path.exists():
            initial = json.loads(receipt_path.read_text())
            if initial.get('verified') and initial['commit'] == initial['remote_SHA']:
                break
        time.sleep(5)
    else:
        write(target, dict(status='BLOCKED_PUBLICATION', time=time.time(),
                          reason='No verified evidence receipt arrived before the closeout boundary.'))
        return
    write(ROOT / 'state/evidence_publication_verified.json', dict(initial, time=time.time()))
    owner = prior.resources()
    owner.restore(list(range(8)))
    burners = owner.burners()
    active = owner.processes()
    assert set(burners) == set(range(8)), 'All eight approved burner workers must remain present'
    assert {r['gpu'] for r in active} == set(range(8))
    prior.preserve()
    write(ROOT / 'state/resource_restore_verified.json', dict(time=time.time(),
        approved_burners=burners, active=active, verified=True,
        policy='Keep approved burners active after the final response. Do not start more timing experiments.'))
    write(ROOT / 'state/CAMPAIGN_CLOSED.json', dict(time=time.time(),
        reason='GPU work ended. Publish the bounded recovery result and blocked method comparison.',
        no_further_GPU_work=True))
    write(target, dict(status='READY_FOR_FINAL_RECEIPT_COMMIT', time=time.time(),
        elapsed_hours=(time.time() - START) / 3600, evidence_commit=initial['commit'],
        original_index_preserved=True, external_runtime_preserved=True,
        method_status='BLOCKED_CORRECTNESS', recovery_status='OPERATIONAL_RECOVERY_READY'))
    from .closeout_report import main as report
    report()
    from .publish import main as publish
    publish('publish', push=True)
    final = json.loads(receipt_path.read_text())
    assert final['verified'] and final['commit'] == final['remote_SHA']
    prior.preserve()
    write(target, dict(status='CLOSED_AND_PUSH_VERIFIED', time=time.time(),
        elapsed_hours=(time.time() - START) / 3600, evidence_commit=initial['commit'],
        final_commit=final['commit'], remote_SHA=final['remote_SHA'], branch=BRANCH,
        approved_burners=owner.burners(), original_index_preserved=True,
        external_runtime_preserved=True, method_status='BLOCKED_CORRECTNESS',
        recovery_status='OPERATIONAL_RECOVERY_READY'))
    print(json.dumps(json.loads(target.read_text())), flush=True)


if __name__ == '__main__':
    main()
