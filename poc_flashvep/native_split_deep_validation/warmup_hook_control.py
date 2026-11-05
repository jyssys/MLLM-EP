"""Qualify terminal-state caching and measure its finite observer control."""
import copy
from .common import *
from .prepare import configuration
from .qualification import qualify
from .serving import evaluate


def main(gate):
    initialize()
    path = ROOT / 'state/warmup_hook_control.json'
    ledger = dict(status='STATE_QUALIFICATION_IN_PROGRESS', qualification=[], controls=[],
        scope='Generic startup-observer correction. Historical c_ discovery retains repeated marker IO. New calibration and held-out runs retire the terminal hook. No model operation or native readiness predicate changes.')
    write(path, ledger)
    for count in (8, 32):
        cfg = configuration('huginn', 'token1984', count,
            f'i_huginn_token1984_retired_warmup_hook_state{count}', state=True, simultaneous=True)
        cfg.update(deep_complete_D_geometry_warmup=True, deep_retire_geometry_hook_after_warmup=True,
                   startup_timeout=600, timeout=1100,
                   phase='WARMUP_HOOK_RETIREMENT_EXACT_QUALIFICATION')
        save(cfg)
        result = qualify(cfg['name'], f'b_huginn_native_reserved8_state{count}')
        ledger['qualification'].append(result)
        write(path, ledger)
        if not result['passed']:
            ledger['status'] = 'BLOCKED_CORRECTNESS'
            write(path, ledger)
            raise RuntimeError('Terminal warm-up observer correction failed exact state qualification')
    ledger['status'] = 'PAIRED_OBSERVER_CONTROL_IN_PROGRESS'
    write(path, ledger)
    for model, method in (('huginn', 'native'), ('huginn', 'loop31'),
                          ('huginn', 'token1984'), ('ouro', 'token3840')):
        if time.time() >= LONG_START_CUTOFF - 7200:
            ledger['status'] = 'NOT_RUN_TIME_BUDGET'
            write(path, ledger)
            return ledger
        original = f'c_{model}_{method}_discovery_s1_off128'
        cfg = copy.deepcopy(json.loads((ROOT / 'configs' / (original + '.json')).read_text()))
        cfg.update(name=f'i_{model}_{method}_retired_warmup_hook_off128',
            deep_retire_geometry_hook_after_warmup=True,
            phase='WARMUP_OBSERVER_CONTROL_NOT_CUT_SELECTION')
        result = evaluate(cfg, f'c_{model}_native_discovery_s1_off128')
        before = json.loads((ROOT / 'raw' / original / 'deep_serving_gate.json').read_text())
        row = dict(result=result, original=original)
        if result.get('passed') and before.get('passed'):
            old, new = before['metrics'], result['metrics']
            row['ratios'] = {key: new[key] / old[key] for key in
                ('TTFT_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms', 'response_span_s')}
            row['global_wall_5_percent_screen'] = abs(row['ratios']['response_span_s'] - 1) <= .05
        ledger['controls'].append(row)
        write(path, ledger)
    ledger['status'] = 'OBSERVER_CONTROLS_FINISHED'
    write(path, ledger)
    return ledger
