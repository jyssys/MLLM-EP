"""Equal-budget discovery and untouched paired confirmation after exact gates."""
import random
import traceback
from .common import *
from .prepare import ACTIONS, configuration, permitted_methods
from .serving import evaluate


def wait_qualification():
    path = ROOT / 'state/qualification.json'
    while time.time() < LONG_START_CUTOFF - 3600:
        if path.exists():
            value = json.loads(path.read_text())
            if value['status'] == 'QUALIFICATION_FINISHED':
                return value
            if value['status'].startswith('BLOCKED'):
                return value
        time.sleep(5)
    return dict(status='NOT_RUN_TIME_BUDGET')


def main():
    initialize()
    gate = wait_qualification()
    target = ROOT / 'state/serving_campaign.json'
    ledger = dict(status='DISCOVERY_IN_PROGRESS', models={},
        contract='Discovery and held-out prompts/seeds are disjoint. Marker-free confirmation is primary clean latency.')
    write(target, ledger)
    if gate['status'] != 'QUALIFICATION_FINISHED':
        write(target, dict(status='BLOCKED_QUALIFICATION', gate_status=gate['status']))
        return
    from .mixed_fallback import main as fallback_checks
    ledger['forced_fallback'] = fallback_checks(gate)
    write(target, ledger)
    for model in ('huginn', 'ouro'):
        qualified = gate['models'][model]['qualification']
        methods = [method for method in permitted_methods(model) if qualified.get(method, {}).get('passed')]
        entry = ledger['models'][model] = dict(discovery=[], selected={}, confirmations=[], loads=[])
        if 'native' not in methods:
            entry['status'] = 'BLOCKED_NATIVE_N32'
            write(target, ledger)
            continue
        # Ouro's full three-start cache-control map is predeclared, not winner-only repetition.
        for start in range(1, 4 if model == 'ouro' else 2):
            rest = [method for method in methods if method != 'native']
            random.Random(2026110520 + start).shuffle(rest)
            # All discovery arms use marker-free latency. Instrumented cost
            # calibration and paired selected-arm markers remain separate.
            for markers in (False,):
                reference = None
                for method in ['native'] + rest:
                    if time.time() >= LONG_START_CUTOFF:
                        entry['status'] = 'NOT_RUN_TIME_BUDGET'
                        write(target, ledger)
                        return
                    label = 'on' if markers else 'off'
                    cfg = configuration(model, method, 128, f'c_{model}_{method}_discovery_s{start}_{label}128',
                        start=start, markers=markers, arrival_seed=2026110510 + start)
                    result = evaluate(cfg, reference)
                    entry['discovery'].append(result)
                    write(target, ledger)
                    if method == 'native':
                        reference = cfg['name']
                        if not result['passed']:
                            break
        for axis, prefix in (('token', 'token'), ('loop', 'loop'), ('2d', '2d_')):
            candidates = [row for row in entry['discovery'] if row['independent_start'] == 1 and not row['markers']
                          and row['method'].startswith(prefix) and row.get('passed')]
            safe = [row for row in candidates if row['metrics']['Decode_SLO_screen']
                    and row['performance_eligible_graphs']]
            if safe:
                best = max(safe, key=lambda row: (row['metrics']['joint_SLO_goodput'], -row['metrics']['TTFT_p95_ms']))
                entry['selected'][axis] = dict(method=best['method'], Decode_safe=True)
            elif candidates:
                best = min(candidates, key=lambda row: row['metrics']['ITL_p99_ms'])
                entry['selected'][axis] = dict(method=best['method'], Decode_safe=False,
                    scope='Least-tail diagnostic only. No tested Decode-safe partial action exists.')
        entry['status'] = 'DISCOVERY_FINISHED_SELECTION_FROZEN'
        write(target, ledger)
    write(ROOT / 'state/frozen_static_selection.json', dict(time=time.time(),
        selected={model: value.get('selected', {}) for model, value in ledger['models'].items()},
        cut_selection_uses_heldout=False))
    from .cache_sensitivity import main as cache_controls
    cache_controls(gate)

    # Collect independent cost data and freeze the model before held-out results exist.
    ledger['status'] = 'CALIBRATION_BEFORE_HELDOUT'
    write(target, ledger)
    from .calibration import main as calibrate
    calibrate(gate, ledger)
    from .oracle import fit_and_freeze
    try:
        fit_and_freeze()
    except Exception:
        write(ROOT / 'state/oracle_fit_exception.json', dict(time=time.time(),
            traceback=traceback.format_exc(), scope='CPU oracle fitting only. Qualified GPU comparison remains the higher priority.'))
    ledger['status'] = 'HELDOUT_CONFIRMATION_IN_PROGRESS'
    write(target, ledger)
    selected_methods = {}
    for model in ('huginn', 'ouro'):
        qualified = gate['models'][model]['qualification']
        if not qualified.get('native', {}).get('passed'):
            selected_methods[model] = []
            continue
        chosen = list(dict.fromkeys(['native', 'zero'] +
                        [row['method'] for row in ledger['models'][model]['selected'].values()] +
                        (['loop31'] if model == 'huginn' and qualified.get('loop31', {}).get('passed') else [])))
        selected_methods[model] = [method for method in chosen if qualified.get(method, {}).get('passed')]

    def confirmation(model, start, marker_order):
        entry = ledger['models'][model]
        selected = selected_methods[model]
        if not selected:
            return True
        shift = (start - 1) % len(selected)
        order = selected[shift:] + selected[:shift]
        for markers in marker_order:
            reference = f'd_{model}_native_heldout_s1_off128' if start > 1 or markers else None
            for method in order:
                if time.time() >= LONG_START_CUTOFF:
                    entry['status'] = 'NOT_RUN_TIME_BUDGET'
                    write(target, ledger)
                    return False
                label = 'on' if markers else 'off'
                cfg = configuration(model, method, 128, f'd_{model}_{method}_heldout_s{start}_{label}128',
                    split='heldout', start=start, markers=markers, arrival_seed=2026110600 + start)
                from .oracle import predict_configuration
                predict_configuration(cfg)
                result = evaluate(cfg, reference)
                entry['confirmations'].append(result)
                write(target, ledger)
                if method == 'native' and reference is None:
                    reference = cfg['name']
                if not result['passed']:
                    entry['status'] = 'BLOCKED_HELDOUT_CORRECTNESS'
                    write(target, ledger)
                    return False
        return True

    # Complete three starts in both models before optional extra starts.
    for start in (1, 2, 3):
        for model in ('huginn', 'ouro'):
            marker_order = (True, False) if start == 2 else (False, True)
            if not confirmation(model, start, marker_order):
                return
    from .oracle import predict_configuration
    load_orders = {model: [method for method in choices if method != 'zero'] +
                   (['zero'] if 'zero' in choices else []) for model, choices in selected_methods.items()}
    references = {}
    for index in range(max(map(len, load_orders.values()))):
        for load in ('low', 'heavy'):
            for model in ('huginn', 'ouro'):
                entry = ledger['models'][model]
                if index >= len(load_orders[model]):
                    continue
                method = load_orders[model][index]
                if time.time() >= LONG_START_CUTOFF:
                    entry['status'] = 'NOT_RUN_TIME_BUDGET'
                    write(target, ledger)
                    return
                cfg = configuration(model, method, 128, f'e_{model}_{method}_{load}_heldout_off128',
                                    split='heldout', markers=False, load=load, arrival_seed=2026110601)
                predict_configuration(cfg)
                result = evaluate(cfg, references.get((model, load)))
                entry['loads'].append(result)
                write(target, ledger)
                if method == 'native':
                    references[model, load] = cfg['name']
    for model in ('huginn', 'ouro'):
        entry = ledger['models'][model]
        entry['status'] = 'SERVING_FINISHED'
        write(target, ledger)
    for start in (4, 5):
        if not confirmation('huginn', start, (False,)):
            return
    ledger['status'] = 'SERVING_FINISHED'
    write(target, ledger)
    from ..looped_pd_headroom_recovery.common import restore_if_idle
    restore_if_idle()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        initialize()
        write(ROOT / 'state/serving_exception.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
