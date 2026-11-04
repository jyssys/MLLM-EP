"""Qualify cut choices, then measure the same finite serving traces."""
import copy
import random
import time
from .common import *
from .initial_qualification import template, qualify
from .run import execute

RATES = {'ouro': {'low': 1.23460875, 'medium': 3.4569045, 'heavy': 6.356},
         'huginn': {'low': .37903355455, 'medium': 1.06129395274, 'heavy': 1.971}}


def plans(model):
    cfg = template(model, 'loop_partial')
    result = [('native', 'atomic_native', {}), ('staged_native', 'staged_native', {}),
              ('snapshot_full', 'loop_partial', dict(cut=cfg['depth'])),
              ('token_full', 'token_partial', dict(prefix_tokens=cfg['length']))]
    for cut in ([3, 2, 1] if model == 'ouro' else [31, 28, 24, 16]):
        result.append((f'loop{cut}', 'loop_partial', dict(cut=cut)))
    for cut in ([3, 2, 1] if model == 'ouro' else [31, 28, 24, 16]):
        prefix = cfg['length'] * cut // cfg['depth']
        result.append((f'token{prefix}', 'token_partial', dict(prefix_tokens=prefix)))
    return result


def configuration(model, label, changes, count, name, state=False, rate=None, seed=2026110402):
    cfg = template(model, label, count)
    cfg.update(changes)
    cfg.update(name=name, startup_relative_token_boundary=True, observe_service_timing=True,
        state_probe=state, verify_transfer=state,
        workload=f"L{cfg['length']}_O{cfg['cap']}",
        phase='EXACT_CUT_QUALIFICATION_NOT_LATENCY' if state else 'CLEAN_FINITE_SERVING_WITH_EVENT_START_MARKERS',
        timeout=1800 if state else 900)
    if rate is not None:
        cfg['arrival_rate'] = rate
    cfg['arrival_seed'] = seed
    rng = random.Random(seed)
    cfg['arrival_offsets'] = [0.]
    for _ in range(count - 1):
        cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    cfg['request_caps'] = [cfg['cap']] * count
    pool = json.loads((ROOT / 'data' / cfg['dataset']).read_text())
    cfg['prompt_ids'] = [row['prompt_id'] for row in pool[:count]]
    return cfg


def main():
    initialize()
    source = ROOT / 'state/split_qualification.json'
    while time.time() < LONG_START_CUTOFF:
        if source.exists() and json.loads(source.read_text())['status'] == 'QUALIFICATION_PHASE_FINISHED':
            break
        time.sleep(2)
    else:
        write(ROOT / 'state/static_campaign.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    gate = json.loads(source.read_text())['models']
    ledger = dict(status='IN_PROGRESS', models={})
    target = ROOT / 'state/static_campaign.json'
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_atomic_prefill import evidence
    from .metrics import summarize
    from .work_accounting import summarize as work_and_payload
    from .lifetime_gate import check as lifetime

    def persist():
        write(target, ledger)

    # Cut tuning has the same number of partial candidates per axis and model.
    for model in ('ouro', 'huginn'):
        results = ledger['models'][model] = dict(qualification={}, serving=[], calibration=[])
        if not all(gate.get(model, {}).get(key, {}).get('passed') for key in ('native32', 'token32', 'loop32')):
            results['status'] = 'BLOCKED_N32_PRIMARY_SPLIT_GATE'
            persist()
            continue
        reference8 = gate[model]['native8']['name']
        reference32 = gate[model]['native32']['name']
        for name, label, values in plans(model):
            existing = (gate[model]['native32'] if name == 'native' else
                        gate[model]['loop32'] if name == ('loop3' if model == 'ouro' else 'loop31') else
                        gate[model]['token32'] if name == ('token3072' if model == 'ouro' else 'token1984') else None)
            if existing:
                results['qualification'][name] = existing
                continue
            checked = []
            for count, reference in ((8, reference8), (32, reference32)):
                cfg = configuration(model, label, values, count, f'e_{model}_{name}_state{count}', True,
                                    seed=2026110401)
                save(cfg)
                row = qualify(cfg['name'], reference)
                checked.append(row)
                if not row['passed']:
                    break
            results['qualification'][name] = dict(passed=len(checked) == 2 and all(r['passed'] for r in checked),
                                                  checks=checked)
            persist()
        results['status'] = 'CUT_QUALIFICATION_FINISHED'
        persist()

    # N128 has identical prompts, seed, cap, and arrival trace in every arm.
    # The order rotates across load conditions to reduce a fixed order effect.
    for load_index, load in enumerate(('low', 'medium', 'heavy')):
        for model in ('ouro', 'huginn'):
            results = ledger['models'][model]
            if results.get('status') in ('BLOCKED_N32_PRIMARY_SPLIT_GATE', 'BLOCKED_NATIVE_SERVING_FAILURE'):
                continue
            choices = plans(model)
            rotation = load_index % len(choices)
            choices = choices[rotation:] + choices[:rotation]
            reference = None
            # Obtain the native reference first, regardless of the rotated order.
            choices.sort(key=lambda item: item[0] != 'native')
            for name, label, values in choices:
                if load != 'medium' and name in ('staged_native', 'snapshot_full', 'token_full'):
                    results['serving'].append(dict(load=load, plan=name,
                        status='NOT_RUN_PREDECLARED_BUDGET_PRIORITY',
                        reason='The same-cut transport controls run at medium load for both models. Native PD and primary split comparisons retain all three loads.'))
                    persist()
                    continue
                primary = {'native', 'staged_native', 'snapshot_full', 'token_full',
                           'loop3' if model == 'ouro' else 'loop31',
                           'token3072' if model == 'ouro' else 'token1984'}
                if load != 'heavy' and name not in primary:
                    results['serving'].append(dict(load=load, plan=name,
                        status='NOT_RUN_PREDECLARED_BUDGET_PRIORITY',
                        reason='Additional cuts have N8/N32 qualification and heavy-load tuning. Normal-load serving prioritizes both primary axes and transport controls.'))
                    persist()
                    continue
                prior_receipt = ROOT / 'raw' / f'f_{model}_{load}_{name}_start1_128' / 'dispatcher_result.json'
                already_complete = prior_receipt.exists() and json.loads(prior_receipt.read_text()).get('complete')
                if time.time() >= LONG_START_CUTOFF and not already_complete:
                    results['serving'].append(dict(load=load, plan=name, status='NOT_RUN_TIME_BUDGET'))
                    persist()
                    continue
                if not results['qualification'].get(name, {}).get('passed'):
                    results['serving'].append(dict(load=load, plan=name, status='BLOCKED_CUT_CORRECTNESS'))
                    persist()
                    continue
                cfg = configuration(model, label, values, 128, f'f_{model}_{load}_{name}_start1_128',
                                    rate=RATES[model][load])
                cfg.update(load=load, independent_start=1)
                save(cfg)
                complete = execute(cfg['name'])
                row = dict(name=cfg['name'], plan=name, load=load, complete=complete, passed=False)
                if complete:
                    with quiet_analysis(cfg['name'] + '_metrics'):
                        row['work'] = evidence.full_work(cfg['name'])
                        row['output'] = evidence.compare(reference, cfg['name']) if reference else dict(passed=True)
                        row['lifetime'] = lifetime(cfg['name'])
                        row['metrics'] = summarize(cfg['name'])
                        row['work_and_payload'] = work_and_payload(cfg['name'])
                        row['passed'] = row['work']['passed'] and row['output']['passed'] and row['lifetime']['passed']
                results['serving'].append(row)
                persist()
                if name == 'native':
                    if not row['passed']:
                        results['status'] = 'BLOCKED_NATIVE_SERVING_FAILURE'
                        break
                    reference = cfg['name']
                elif not row['passed']:
                    results['qualification'][name]['passed'] = False
                    results['qualification'][name]['serving_failure'] = cfg['name']
            persist()
    ledger['status'] = 'STATIC_CAMPAIGN_FINISHED'
    ledger['rate_scope'] = 'Low/medium use 0.25/0.70 of the preserved strongest finite P reference. Heavy uses the fixed operational condition. These are not sustainable-capacity estimates.'
    ledger['timing_scope'] = 'Event-marked clean runs need matched marker-overhead controls before final clean-latency acceptance.'
    persist()


if __name__ == '__main__':
    main()
