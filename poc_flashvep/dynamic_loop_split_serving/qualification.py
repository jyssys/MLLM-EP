"""Run independent state gates. A failed axis does not skip the other axis."""
import time
from .common import *
from .initial_qualification import template, qualify


def main():
    initialize()
    initial = ROOT / 'state/initial_qualification.json'
    while not initial.exists() or json.loads(initial.read_text())['status'] != 'INITIAL_PHASE_FINISHED':
        if time.time() >= LONG_START_CUTOFF:
            raise TimeoutError('The initial gate exceeded the new-job cutoff')
        time.sleep(2)
    previous = json.loads(initial.read_text())
    ledger = {}
    destination = ROOT / 'state/split_qualification.json'

    def record(model, key, value):
        ledger.setdefault(model, {})[key] = value
        write(destination, dict(status='IN_PROGRESS', models=ledger))

    # First check both models at N8. Change only the explicit startup boundary.
    for model in ('ouro', 'huginn'):
        native = previous['models'][model].get('atomic_native', {})
        record(model, 'native8', native)
        if not native.get('passed'):
            record(model, 'status', 'BLOCKED_NATIVE_REFERENCE')
            continue
        reference = native['name']
        for axis, label in (('token', 'token_partial'), ('loop', 'loop_partial')):
            cfg = template(model, label)
            cfg.update(name=f'b_{model}_{axis}_startup_v2_state8', startup_relative_token_boundary=True)
            save(cfg)
            record(model, axis + '8', qualify(cfg['name'], reference))

    # The timing hook receives its own N32 exactness check in all three arms.
    for model in ('ouro', 'huginn'):
        if not ledger[model].get('native8', {}).get('passed'):
            continue
        cfg = template(model, 'atomic_native', 32)
        cfg.update(name=f'c_{model}_native_timed_state32', observe_service_timing=True,
                   startup_relative_token_boundary=True,
                   phase='N32_EXACT_STATE_AND_TIMING_HOOK_NOT_CLEAN_LATENCY')
        save(cfg)
        native = qualify(cfg['name'])
        record(model, 'native32', native)
        if not native['passed']:
            continue
        for axis, label in (('token', 'token_partial'), ('loop', 'loop_partial')):
            if not ledger[model].get(axis + '8', {}).get('passed'):
                continue
            cfg = template(model, label, 32)
            cfg.update(name=f'c_{model}_{axis}_timed_state32', observe_service_timing=True,
                       startup_relative_token_boundary=True,
                       phase='N32_EXACT_STATE_AND_TIMING_HOOK_NOT_CLEAN_LATENCY')
            save(cfg)
            record(model, axis + '32', qualify(cfg['name'], native['name']))

    # Keep boundary and chunk controls distinct. Do not treat failure as parity.
    for model in ('ouro', 'huginn'):
        if not ledger[model].get('token8', {}).get('passed'):
            continue
        reference = ledger[model]['native8']['name']
        for label in ('unaligned', 'half_chunk'):
            cfg = template(model, 'token_partial')
            cfg.update(name=f'd_{model}_token_{label}_state8', startup_relative_token_boundary=True)
            if label == 'unaligned':
                cfg['prefix_tokens'] -= 1 if model == 'ouro' else 3
            else:
                cfg['chunk'] //= 2
                cfg['chunk_label'] = 'HALF_PRIMARY_EXECUTION_CHUNK'
            save(cfg)
            record(model, label + '8', qualify(cfg['name'], reference))
    write(destination, dict(status='QUALIFICATION_PHASE_FINISHED', models=ledger,
        scope='State and output gates only. Missing or failed gates do not authorize a serving expansion.'))


if __name__ == '__main__':
    main()
