"""Exact forced Loop/full-PD branch coverage, not an online policy result."""
from .common import *
from .prepare import configuration
from .qualification import qualify


def main(gate):
    results = {}
    for model, method in (('huginn', 'loop31'), ('ouro', 'loop3')):
        if not gate['models'][model]['qualification'].get(method, {}).get('passed'):
            continue
        results[model] = []
        for count in (8, 32):
            cfg = configuration(model, method, count, f'g_{model}_forced_loop_full_fallback_state{count}',
                                state=True, simultaneous=True)
            cfg['native_request_cuts'] = [cfg['depth'] if index % 2 else cfg['depth'] - 1 for index in range(count)]
            cfg['phase'] = 'FORCED_MIXED_NATIVE_FALLBACK_QUALIFICATION_NOT_ONLINE_POLICY'
            save(cfg)
            result = qualify(cfg['name'], f'b_{model}_native_reserved8_state{count}')
            results[model].append(result)
            write(ROOT / 'state/mixed_native_fallback.json', dict(results=results,
                scope='Preassigned Loop/full-PD branches. Not a live D-admission rejection policy or mixed-2D qualification.'))
            if not result['passed']:
                break
    return results
