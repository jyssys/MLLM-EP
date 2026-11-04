"""Run already-qualified primary cells without waiting for every extra cut.

The main static controller later reuses these exact immutable namespaces.
The shared quiet-lane lock still permits only one GPU execution at a time.
"""
from .common import *
from .static_campaign import configuration, RATES
from .run import execute


def main():
    initialize()
    gate = json.loads((ROOT / 'state/split_qualification.json').read_text())
    assert gate['status'] == 'QUALIFICATION_PHASE_FINISHED'
    assert all(gate['models'][model][key]['passed'] for model in ('ouro', 'huginn')
               for key in ('native32', 'token32', 'loop32'))
    target = ROOT / 'state/primary_screen.json'
    ledger = dict(status='IN_PROGRESS', planned_load_order=['medium', 'heavy', 'low'], results=[],
        scope='Reorder existing primary N128 cells only. No changed rate, prompt, seed, precision, cut, SLO, or acceptance criterion.')
    write(target, ledger)
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .metrics import summarize
    from .work_accounting import summarize as accounting
    from .lifetime_gate import check as lifetime
    blocked_models, blocked_plans = set(), set()
    for load in ledger['planned_load_order']:
        for model in ('ouro', 'huginn'):
            reference = None
            cells = [('native', 'atomic_native'),
                     ('loop3' if model == 'ouro' else 'loop31', 'loop_partial'),
                     ('token3072' if model == 'ouro' else 'token1984', 'token_partial')]
            for plan, label in cells:
                if model in blocked_models or (model, plan) in blocked_plans:
                    ledger['results'].append(dict(model=model, load=load, plan=plan,
                        status='BLOCKED_EARLIER_PRIMARY_FAILURE', passed=False))
                    write(target, ledger)
                    continue
                cfg = configuration(model, label, {}, 128, f'f_{model}_{load}_{plan}_start1_128',
                                    rate=RATES[model][load])
                cfg.update(load=load, independent_start=1)
                save(cfg)
                result = dict(model=model, load=load, plan=plan, name=cfg['name'],
                              complete=execute(cfg['name']), passed=False)
                if result['complete']:
                    with quiet_analysis(cfg['name'] + '_primary_metrics'):
                        result['output'] = evidence.compare(reference, cfg['name']) if reference else dict(passed=True)
                        result['work'] = evidence.full_work(cfg['name'])
                        result['lifetime'] = lifetime(cfg['name'])
                        result['metrics'] = summarize(cfg['name'])
                        result['accounting'] = accounting(cfg['name'])
                        result['passed'] = (result['output']['passed'] and result['work']['passed']
                                            and result['lifetime']['passed'])
                ledger['results'].append(result)
                write(target, ledger)
                if not result['passed']:
                    if plan == 'native':
                        blocked_models.add(model)
                    else:
                        blocked_plans.add((model, plan))
                    continue
                reference = reference or cfg['name']
    ledger['status'] = 'PRIMARY_SCREEN_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
