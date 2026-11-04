"""Strengthen the Token baseline below the minimum whole-loop granularity."""
from .common import *
from .static_campaign import configuration, RATES
from .initial_qualification import qualify
from .run import execute


def plan(model):
    prefix = 3840 if model == 'ouro' else 2016
    return (f'token{prefix}', 'token_partial', dict(prefix_tokens=prefix))


def main():
    initialize()
    target = ROOT / 'state/fine_suffix.json'
    ledger = dict(status='IN_PROGRESS', models={},
        scope='Additional strong-baseline control. The core matched-row map retains its original candidate budget. Ouro already enumerates all nonzero internal whole-loop cuts. No exhaustive Token-optimality claim.')
    for model in ('ouro', 'huginn'):
        key, label, changes = plan(model)
        row = ledger['models'][model] = dict(plan=key, qualification=[], serving=[])
        for count in (8, 32):
            cfg = configuration(model, label, changes, count, f'n_{model}_{key}_state{count}', True,
                                seed=2026110401)
            save(cfg)
            reference = f'a_{model}_atomic_native_origin_v1_state8' if count == 8 else f'c_{model}_native_timed_state32'
            result = qualify(cfg['name'], reference)
            row['qualification'].append(result)
            write(target, ledger)
            if not result['passed']:
                break
        row['qualified'] = len(row['qualification']) == 2 and all(r['passed'] for r in row['qualification'])
        write(target, ledger)
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .metrics import summarize
    from .work_accounting import summarize as accounting
    from .lifetime_gate import check as lifetime
    for load in ('medium', 'heavy', 'low'):
        for model in ('ouro', 'huginn'):
            row = ledger['models'][model]
            if not row['qualified']:
                continue
            reference = f'f_{model}_{load}_native_start1_128'
            primary = ROOT / 'state/primary_screen.json'
            native = None
            while time.time() < LONG_START_CUTOFF:
                state = json.loads(primary.read_text())
                native = next((r for r in state['results'] if r.get('name') == reference), None)
                if native is not None:
                    break
                time.sleep(2)
            if native is None or not native['passed']:
                row['serving'].append(dict(load=load, status='BLOCKED_NATIVE_REFERENCE_OR_TIME_BUDGET'))
                write(target, ledger)
                continue
            key, label, changes = plan(model)
            cfg = configuration(model, label, changes, 128,
                f'n_{model}_{load}_{key}_start1_128', rate=RATES[model][load])
            cfg.update(load=load, independent_start=1)
            save(cfg)
            result = dict(name=cfg['name'], plan=key, model=model, load=load,
                          complete=execute(cfg['name']), passed=False)
            if result['complete']:
                with quiet_analysis(cfg['name'] + '_fine_suffix'):
                    result['output'] = evidence.compare(reference, cfg['name'])
                    result['work'] = evidence.full_work(cfg['name'])
                    result['lifetime'] = lifetime(cfg['name'])
                    result['metrics'] = summarize(cfg['name'])
                    result['accounting'] = accounting(cfg['name'])
                    result['passed'] = all(result[k]['passed'] for k in ('output', 'work', 'lifetime'))
            row['serving'].append(result)
            write(target, ledger)
            if not result['passed']:
                row['qualified'] = False
    ledger['status'] = 'FINE_SUFFIX_SCREEN_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
