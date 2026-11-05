"""Qualify exact graph-stable paths before any clean serving expansion."""
import traceback
from .common import *
from .prepare import configuration, ACTIONS
from .qualification import qualify
from .recovery import guard


def main():
    initialize()
    guard()
    recovery = ROOT / 'state/isolated_scheduler_recovery.json'
    while time.time() < GPU_DEADLINE - 3600:
        if recovery.exists():
            value = json.loads(recovery.read_text())
            if 'token1984' in value:
                break
            if 'native' in value and not value['native']['passed']:
                break
        time.sleep(2)
    else:
        raise RuntimeError('Huginn recovery did not finish within the qualification budget')
    if not value.get('token1984', {}).get('passed'):
        write(ROOT / 'state/qualification.json', dict(status='BLOCKED_HUGINN_SCHEDULER_RECOVERY', recovery=value))
        return
    ledger = dict(status='NATIVE_STATIC_QUALIFICATION_IN_PROGRESS', models={
        model: dict(qualification={}) for model in ('ouro', 'huginn')})
    path = ROOT / 'state/qualification.json'
    orders = {
        'huginn': ['native', 'zero', 'token1984', 'loop31', '2d_c31_k1024', 'token1792',
                   'loop28', '2d_c28_k1536', 'token2016', 'loop24', '2d_c31_k1792'],
        'ouro': ['native', 'zero', '2d_c3_k3072', 'loop3', 'token3840', 'token3072',
                 'loop2', '2d_c2_k2048', 'token4032', 'loop1', '2d_c3_k3840'],
    }
    for index in range(max(map(len, orders.values()))):
        for model in ('huginn', 'ouro'):
            method = orders[model][index]
            if method != 'native' and not ledger['models'][model]['qualification'].get('native', {}).get('passed'):
                continue
            results = []
            for count in (8, 32):
                if time.time() >= LONG_START_CUTOFF - 1200:
                    results.append(dict(passed=False, status='NOT_RUN_TIME_BUDGET'))
                    break
                name = f'b_{model}_{method}_reserved8_state{count}'
                cfg = configuration(model, method, count, name, state=True, simultaneous=True)
                save(cfg)
                reference = f'b_{model}_native_reserved8_state{count}' if method != 'native' else None
                if method == 'native':
                    reference = f'b_{model}_native_reserved4_state{count}'
                results.append(qualify(name, reference))
                ledger['models'][model]['qualification'][method] = dict(
                    passed=len(results) == 2 and all(row['passed'] for row in results),
                    checks=results, config=name)
                write(path, ledger)
                print('QUALIFICATION', model, method, count, results[-1]['passed'], flush=True)
                if not results[-1]['passed']:
                    break
    ledger['status'] = 'QUALIFICATION_FINISHED'
    write(path, ledger)
    from ..looped_pd_headroom_recovery.common import restore_if_idle
    restore_if_idle()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        initialize()
        write(ROOT / 'state/qualification_exception.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
