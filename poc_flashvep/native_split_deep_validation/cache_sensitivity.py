"""Matched legacy/reserved cache controls. Never use legacy as the main baseline."""
from .common import *
from .prepare import configuration
from .qualification import qualify
from .serving import evaluate


def cache_matrix(gate):
    initialize()
    if not gate['models']['ouro']['qualification'].get('2d_c3_k3072', {}).get('passed'):
        return
    cfg = configuration('ouro', '2d_c3_k3072', 8, 'h_ouro_2d_legacy_cache_state8',
                        state=True, simultaneous=True, policy='legacy', graph_budget=4)
    save(cfg)
    qualification = qualify(cfg['name'], 'b_ouro_native_reserved8_state8')
    ledger = dict(qualification=qualification, results=[])
    if not qualification['passed']:
        write(ROOT / 'state/cache_sensitivity.json', ledger)
        return
    for start in (1, 2, 3):
        choices = [('legacy', 4), ('primary_reserved', 4), ('primary_reserved', 8)]
        if start == 2:
            choices.reverse()
        for policy, budget in choices:
            cfg = configuration('ouro', '2d_c3_k3072', 128, f'h_ouro_2d_{policy}_budget{budget}_s{start}_on128',
                                start=start, policy=policy, graph_budget=budget, arrival_seed=2026110510 + start)
            cfg['deep_retire_geometry_hook_after_warmup'] = True
            result = evaluate(cfg, f'c_ouro_native_discovery_s{start}_off128')
            ledger['results'].append(result)
            write(ROOT / 'state/cache_sensitivity.json', ledger)
    return ledger


def main(gate):
    from .warmup_hook_control import main as observer_controls
    result = observer_controls(gate)
    write(ROOT / 'state/cache_sensitivity_deferred.json', dict(time=time.time(),
        status='DEFERRED_UNTIL_CORE_HELDOUT_REPETITIONS',
        reason='Qualify the terminal warm-up observer correction first. Preserve three paired starts in both models before optional legacy-cache controls.',
        observer_control_status=result['status'],
        scope='The complete three-start Ouro main-cache discovery map remains retained. Legacy4/reserved4/reserved8 sensitivity requires its own later receipts.'))
    return result
