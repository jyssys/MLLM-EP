"""Keep a resumable receipt-based progress manifest without latency inference."""
import json
import time

from .common import ROOT, write


def snapshot():
    initial = json.loads((ROOT / 'state/preservation.json').read_text())['time']
    rows = []
    for path in sorted((ROOT / 'raw').glob('*/dispatcher_result.json')):
        rows.append(dict(name=path.parent.name, **json.loads(path.read_text())))
    lease = ROOT / 'state/GPU_lease_current.json'
    active = json.loads(lease.read_text()) if lease.exists() else None
    names = [r['name'] for r in rows if r['complete']]
    revised = (ROOT / 'state/user_primary_scope_revision_v2.json').exists()
    primary_names = [n for n in names if n.startswith(('confirm_Native_', 'confirm_DP2_B8192_S1024_'))]
    value = dict(start_epoch=initial, updated_epoch=time.time(), elapsed_wall_seconds=time.time() - initial,
        status='CLOSED' if (ROOT / 'state/CAMPAIGN_CLOSED.json').exists() else 'ACTIVE',
        active_lease=active, completed_model_starts=len(names),
        failed_receipts=[r for r in rows if not r['complete']],
        scope_version=2 if revised else 1,
        fresh_confirmation_complete=len(primary_names) if revised else sum(n.startswith('confirm_') for n in names),
        fresh_confirmation_planned=54 if revised else 81,
        preserved_completed_small_budget_confirmations=sum(n.startswith('confirm_DP2_B512_') for n in names),
        long_window_complete=sum(n.startswith('long_') for n in names), long_window_planned=8 if revised else 10,
        secondary_complete=sum(n.startswith('secondary_') for n in names), secondary_planned=12 if revised else 18,
        TP2_diagnostic_complete=sum(n.startswith('tp_diagnostic_') for n in names), TP2_diagnostic_planned=18,
        scope='Counts are normal dispatcher receipts, not proof that every numerical or metrology gate passed. Wall time includes initialization, warmup, queueing, experiments, and analysis. It is not GPU kernel time.')
    write(ROOT / 'state/progress.json', value)
    return value


def main():
    while snapshot()['status'] != 'CLOSED':
        time.sleep(30)


if __name__ == '__main__':
    main()
