"""Qualify a shared submission optimization before any performance use."""
from .common import *
from .static_campaign import configuration
from .initial_qualification import qualify


def main():
    initialize()
    if (ROOT / 'state/burst_pilot.json').exists():
        prior_state = json.loads((ROOT / 'state/burst_pilot.json').read_text())
        if prior_state.get('status') == 'BLOCKED_NATIVE_WAVEFRONT_PROGRESS_GUARD':
            return
    target = ROOT / 'state/burst_pilot.json'
    ledger = dict(status='IN_PROGRESS', models={},
        scope='New optional execution epoch. No baseline or SLO change. N8 exactness only. No performance expansion before separate N32 qualification.')
    for model in ('huginn', 'ouro'):
        row = ledger['models'][model] = {}
        reference = None
        for label in ('atomic_native', 'loop_partial', 'token_partial'):
            if label != 'atomic_native':
                proof = ROOT / 'state/native_burst_cpu_replay.json'
                while not proof.exists() and time.time() < LONG_START_CUTOFF:
                    time.sleep(2)
                if not proof.exists() or not json.loads(proof.read_text()).get('passed'):
                    row[label] = dict(passed=False, status='BLOCKED_NATIVE_FRONTIER_CPU_REPLAY')
                    write(target, ledger)
                    continue
            cfg = configuration(model, label, {}, 8, f'm_{model}_{label}_burst_state8', True,
                                seed=2026110401)
            cfg.update(contiguous_prompt_loop_burst=True,
                phase='NATIVE_MULTI_LOOP_SUBMISSION_N8_EXACTNESS_NOT_LATENCY')
            save(cfg)
            result = qualify(cfg['name'], reference or f'a_{model}_atomic_native_origin_v1_state8')
            row[label] = result
            write(target, ledger)
            if label == 'atomic_native' and not result['passed']:
                break
            if not result['passed']:
                log = ROOT / 'logs' / (cfg['name'] + '.log')
                if log.exists() and 'wavefront prefill requires one recurrent depth per scheduling step' in log.read_text():
                    ledger['status'] = 'BLOCKED_NATIVE_WAVEFRONT_PROGRESS_GUARD'
                    ledger['shared_error'] = 'wavefront prefill requires one recurrent depth per scheduling step'
                    write(target, ledger)
                    return
            if result['passed']:
                reference = reference or cfg['name']
    ledger['status'] = 'BURST_PILOT_FINISHED_REVIEW_REQUIRED'
    write(target, ledger)


if __name__ == '__main__':
    main()
