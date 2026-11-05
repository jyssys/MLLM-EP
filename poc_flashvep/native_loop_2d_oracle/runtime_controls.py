"""Rerun full-depth transport/runtime controls under the new common trace."""
import copy
from .common import *
from .serving import serving_configuration, evaluate


def main():
    initialize()
    path = ROOT / 'state/effective_qualification.json'
    while time.time() < GPU_DEADLINE - 900:
        if path.exists() and json.loads(path.read_text())['status'] == 'QUALIFICATION_FINISHED':
            break
        time.sleep(5)
    else:
        write(ROOT / 'state/runtime_controls.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    gate = json.loads(path.read_text())
    ledger = dict(status='IN_PROGRESS', results=[])
    target = ROOT / 'state/runtime_controls.json'
    for model in ('ouro', 'huginn'):
        if not gate['models'].get(model, {}).get('qualification', {}).get('native', {}).get('passed'):
            ledger['results'].append(dict(model=model, status='BLOCKED_NATIVE_N32'))
            continue
        for method in ('staged_native', 'snapshot_full'):
            original = PREVIOUS / 'configs' / f'f_{model}_medium_{method}_start1_128.json'
            cfg = serving_configuration(json.loads(original.read_text()),
                f'e_{model}_{method}_medium_control128')
            cfg.update(method=method, observe_transport=True,
                parent_configuration=dict(path=str(original), SHA256=digest(original), publication=BASE))
            # The current Native medium control can be queued behind this job.
            # Record output comparison after it finishes in the analysis pass.
            result = evaluate(cfg)
            result['reference_comparison'] = 'PENDING_MATCHED_CURRENT_NATIVE_CONTROL'
            ledger['results'].append(result)
            write(target, ledger)
    ledger['status'] = 'RUNTIME_CONTROLS_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
