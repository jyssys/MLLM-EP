"""Prioritize predeclared transport controls after their N32 gate."""
from .common import *
from .static_campaign import configuration, plans, RATES
from .run import execute


def main():
    initialize()
    target = ROOT / 'state/transport_screen.json'
    ledger = dict(status='IN_PROGRESS', results=[],
        scope='Reorder existing medium-load cells. The full static controller later reuses these exact immutable namespaces.')
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .metrics import summarize
    from .lifetime_gate import check as lifetime
    from .work_accounting import summarize as accounting
    for model in ('huginn', 'ouro'):
        for plan, label, changes in plans(model):
            if plan not in ('staged_native', 'snapshot_full', 'token_full'):
                continue
            qualification = json.loads((ROOT / 'state/static_campaign.json').read_text())
            gate = qualification['models'][model]['qualification'][plan]
            assert gate['passed'], 'Transport control requires its own completed N8/N32 gate'
            cfg = configuration(model, label, changes, 128,
                f'f_{model}_medium_{plan}_start1_128', rate=RATES[model]['medium'])
            cfg.update(load='medium', independent_start=1)
            save(cfg)
            row = dict(name=cfg['name'], model=model, plan=plan, load='medium',
                       complete=execute(cfg['name']), passed=False)
            if row['complete']:
                with quiet_analysis(cfg['name'] + '_transport_control'):
                    row['output'] = evidence.compare(f'f_{model}_medium_native_start1_128', cfg['name'])
                    row['work'] = evidence.full_work(cfg['name'])
                    row['lifetime'] = lifetime(cfg['name'])
                    row['metrics'] = summarize(cfg['name'])
                    row['accounting'] = accounting(cfg['name'])
                    row['passed'] = all(row[key]['passed'] for key in ('output', 'work', 'lifetime'))
            ledger['results'].append(row)
            write(target, ledger)
            if not row['passed']:
                break
    ledger['status'] = 'TRANSPORT_SCREEN_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
