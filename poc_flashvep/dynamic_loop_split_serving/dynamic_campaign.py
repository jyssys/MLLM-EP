"""Measure costs and qualify branches before bounded online comparisons."""
import copy
import random
import time
from .common import *
from .static_campaign import configuration, plans, RATES
from .initial_qualification import qualify
from .run import execute


def write_once(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value
    else:
        write(path, value)


def cached_calibration(folder, cfg, reference):
    path = folder / 'isolated_split_cost.json'
    receipt = folder / 'dispatcher_result.json'
    contract = folder / 'arc_contract.json'
    if not all(p.exists() for p in (path, receipt, contract)):
        return None
    row = json.loads(path.read_text())
    if not (row.get('passed') and json.loads(receipt.read_text()).get('complete')):
        return None
    assert json.loads(contract.read_text()) == cfg, 'Immutable calibration contract changed'
    if reference is not None and row.get('output', {}).get('reference') != reference:
        return None
    return dict(row, reuse_scope='Previously recorded isolated P-p95, D-maximum, and handoff-p95 predictor. Same immutable GPU run and reference. No new measurement or stronger latency bound.')


def calibration(model, qualified, record):
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from ..looped_pd_headroom_recovery.gates import percentile
    from .metrics import summarize
    source = configuration(model, 'loop_partial', {}, 8, 'unused')
    source_calibration = PREVIOUS / 'data' / f"calibration_{model}_L{source['length']}.json"
    pool = json.loads(source_calibration.read_text())[:8]
    assert len(pool) == 8
    measured_pool = json.loads((ROOT / 'data' / source['dataset']).read_text())[:128]
    assert not ({r['prompt_id'] for r in pool} & {r['prompt_id'] for r in measured_pool})
    dataset = f'cost_{model}_disjoint_calibration8.json'
    write_once(ROOT / 'data' / dataset, pool)
    rows = []
    reference = None
    from .fine_suffix import plan as fine_plan
    for plan, label, change in plans(model) + [fine_plan(model)]:
        if plan in ('native', 'staged_native') or not qualified.get(plan, {}).get('passed'):
            continue
        cfg = configuration(model, label, change, 8, f'g_{model}_{plan}_isolated_cost8')
        cfg.update(dataset=dataset, prompt_ids=[row['prompt_id'] for row in pool], seed=202611045001,
            arrival_rate=None, arrival_seed=None, arrival_offsets=[0.] * 8,
            calibration_serial=True, observe_device_execution=True,
            phase='ISOLATED_MEASURED_COST_NOT_CLEAN_SERVING',
            data_scope=dict(source=str(source_calibration), source_SHA256=digest(source_calibration),
                overlap_with_primary_128=0,
                scope='Eight preserved calibration records. These are disjoint from the primary 128-record cohort.'))
        save(cfg)
        cached = cached_calibration(ROOT / 'raw' / cfg['name'], cfg, reference)
        if cached is not None:
            rows.append(cached)
            record('calibration', rows)
            reference = reference or cfg['name']
            continue
        complete = execute(cfg['name'])
        row = dict(name=cfg['name'], plan=plan, axis=cfg['axis'], complete=complete, passed=False)
        if complete:
            with quiet_analysis(cfg['name'] + '_cost'):
                row['work'] = evidence.full_work(cfg['name'])
                row['output'] = evidence.compare(reference, cfg['name']) if reference else dict(passed=True)
                metric = summarize(cfg['name'])
                per_request = json.loads((ROOT / 'raw' / cfg['name'] / 'service_latency_requests.json').read_text())
                expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
                accepted = {r['seed']: r for p in (ROOT / 'raw' / cfg['name']).glob('accepted_partitions_*.jsonl')
                            for line in p.read_text().splitlines() if (r := json.loads(line))['seed'] in expected}
                outputs = {r['seed']: r for line in (ROOT / 'raw' / cfg['name'] / 'requests.jsonl').read_text().splitlines()
                           if not (r := json.loads(line)).get('warmup')}
                assert set(accepted) == expected
                partial = any(r['partial'] for r in accepted.values())
                p_samples = [r['P_prompt_device_enclosure_ms'] / 1000 for r in per_request]
                transfer = [r['observed_handoff_ms'] / 1000 for r in per_request]
                d_samples = [outputs[seed]['first_token'] - accepted[seed]['time'] for seed in expected]
                assert min(p_samples) > 0 and min(transfer) >= 0 and min(d_samples) >= 0
                row['cost'] = dict(key=cfg['cut'] if cfg['axis'] == 'loop' else cfg['prefix_tokens'],
                    p_service_s=percentile(p_samples, .95),
                    d_continuation_s=max(d_samples) if partial else 0.,
                    transfer_exposed_s=percentile(transfer, .95), d_extra_blocks=0)
                row['scope'] = ('Isolated measured device/host enclosures. D acceptance-to-first-output includes continuation, '
                    'Coda, scheduling, and delivery. These are finite predictors, not kernel unions or concurrent bounds.')
                row['passed'] = bool(row['work']['passed'] and row['output']['passed'])
                write(ROOT / 'raw' / cfg['name'] / 'isolated_split_cost.json', row)
        rows.append(row)
        record('calibration', rows)
        if not row['passed']:
            return rows
        reference = reference or cfg['name']
    return rows


def table_for(model, axis, cfg, costs, serving):
    from .dynamic_recovery import signature
    candidates = [row for row in serving if row.get('passed') and row.get('plan') == 'native'
                  and row.get('metrics', {}).get('Decode_SLO_screen')]
    assert candidates, 'A native baseline must satisfy the frozen Decode screens'
    baseline = max(candidates, key=lambda r: r['metrics']['nominal_QPS'])
    folder = ROOT / 'raw' / baseline['name']
    observed = [json.loads(line) for line in (folder / 'queue_observations.jsonl').read_text().splitlines()]
    peak = max(row['pending_phases'].get('decode', 0) for row in observed)
    assert peak > 0
    selected = [row for row in costs if row.get('passed') and row['axis'] == axis]
    full_key = cfg['depth'] if axis == 'loop' else cfg['length']
    assert len(selected) >= 2 and any(row['cost']['key'] == full_key for row in selected)
    artifacts = [folder / 'requests.jsonl', folder / 'queue_observations.jsonl',
                 ROOT / 'configs/frozen_SLO.json']
    for row in selected:
        path = ROOT / 'raw' / row['name']
        cal_cfg = json.loads((ROOT / 'configs' / (row['name'] + '.json')).read_text())
        artifacts.extend([path / 'isolated_split_cost.json', path / 'closed_stats.json',
                          ROOT / 'configs' / (row['name'] + '.json'), ROOT / 'data' / cal_cfg['dataset'],
                          *path.glob('service_device_times_*.json')])
    slo = baseline['metrics']['frozen_SLO']
    table = dict(qualified=True, axis=axis, execution_signature={key: cfg.get(key) for key in signature()},
        TTFT_limit_s=slo['TTFT_ms'] / 1000, ITL_limit_s=slo['ITL_ms'] / 1000,
        TPOT_margin_s=.9 * slo['TPOT_ms'] / 1000,
        safe_decode_batch=min(cfg['D_slots'], peak),
        baseline_token_interval_s=baseline['metrics']['ITL_p95_ms'] / 1000,
        calibration_evidence={str(path): digest(path) for path in artifacts},
        costs=[row['cost'] for row in selected],
        scope='Measured finite cost predictor. Online observations only. No future arrivals or actual future output lengths.',
        workspace_rule='The destination reserves the full native request KV before selecting the cut. Common graph/workspace allocation is unchanged.',
        policy='Discrete admission/frontier FluidToken-style reproduction. Not the complete continuous queue-budget algorithm.')
    path = ROOT / 'configs' / f'measured_cost_{model}_{axis}.json'
    write_once(path, table)
    return path


def serving_configuration(task, load):
    model, axis = task['model'], task['axis']
    cfg = configuration(model, task['label'], {}, 128,
        f'i_{model}_{load}_{axis}_dynamic_start1_128', rate=RATES[model][load])
    cfg.update(task['contract'], dynamic=True, arm=task['arm'], load=load, independent_start=1)
    return cfg


def main():
    initialize()
    source = ROOT / 'state/static_campaign.json'
    primary_source = ROOT / 'state/primary_screen.json'
    fine_source = ROOT / 'state/fine_suffix.json'
    while time.time() < LONG_START_CUTOFF:
        static = json.loads(source.read_text()) if source.exists() else {}
        primary = json.loads(primary_source.read_text()) if primary_source.exists() else {}
        fine = json.loads(fine_source.read_text()) if fine_source.exists() else {}
        fine_ready = fine.get('status') == 'FINE_SUFFIX_SCREEN_FINISHED'
        qualified = all(static.get('models', {}).get(model, {}).get('status') in
                        ('CUT_QUALIFICATION_FINISHED', 'BLOCKED_NATIVE_SERVING_FAILURE')
                        for model in ('ouro', 'huginn'))
        if fine_ready and (static.get('status') == 'STATIC_CAMPAIGN_FINISHED' or (
                qualified and primary.get('status') == 'PRIMARY_SCREEN_FINISHED')):
            break
        time.sleep(5)
    else:
        write(ROOT / 'state/dynamic_campaign.json', dict(status='NOT_RUN_TIME_BUDGET'))
        return
    # Primary cells use the exact same immutable namespaces as the main map.
    # Their completion can release calibration without waiting for every extra
    # static serving cut. The common quiet lane still serializes GPU execution.
    for model in ('ouro', 'huginn'):
        previous = static['models'][model]
        extra = fine.get('models', {}).get(model, {})
        if extra.get('qualified'):
            previous['qualification'][extra['plan']] = dict(passed=True, checks=extra['qualification'])
        combined = {row.get('name', f"missing_{index}"): row
                    for index, row in enumerate(previous.get('serving', []))}
        combined.update({row['name']: row for row in primary.get('results', [])
                         if row.get('model') == model and row.get('name')})
        combined.update({row['name']: row for row in extra.get('serving', []) if row.get('name')})
        previous['serving'] = list(combined.values())
    ledger = dict(status='IN_PROGRESS', stage='CALIBRATION_AND_N32_BOTH_MODELS', models={})
    target = ROOT / 'state/dynamic_campaign.json'
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    from .metrics import summarize
    from .work_accounting import summarize as accounting
    from .lifetime_gate import check as lifetime

    ready = []
    for model in ('huginn', 'ouro'):
        row = ledger['models'][model] = dict(qualification={}, serving=[])

        def record(key, value):
            row[key] = value
            write(target, ledger)

        previous = static['models'][model]
        if not previous.get('serving') or not any(r.get('passed') for r in previous['serving']):
            record('status', 'BLOCKED_STATIC_MEASUREMENT_GATE')
            continue
        costs = calibration(model, previous['qualification'], record)
        if not costs or not all(r['passed'] for r in costs):
            record('status', 'BLOCKED_MEASURED_COST_GATE')
            continue
        for axis in ('token', 'loop'):
            label = 'token_partial' if axis == 'token' else 'loop_partial'
            cfg = configuration(model, label, {}, 32, 'unused', True)
            # Keep the ordinary startup request budget. Cycling short-prompt
            # cuts does not warm the measured full-length geometries and can
            # consume the finite native graph cache before serving begins.
            # Isolated costs retain first-use work. Do not subtract it.
            cfg.update(dynamic=True, arm='Token_Dynamic' if axis == 'token' else 'Loop_Dynamic')
            try:
                with quiet_analysis(model + '_' + axis + '_cost_table'):
                    path = table_for(model, axis, cfg, costs, previous['serving'])
            except AssertionError as error:
                record(axis + '_status', 'BLOCKED_COST_OR_NATIVE_DECODE_SLO: ' + str(error))
                continue
            contract = dict(dynamic_cost_table=str(path), dynamic_cost_table_SHA256=digest(path))
            partial_key = max(r['cost']['key'] for r in costs if r['axis'] == axis and r['cost']['d_continuation_s'] > 0)
            checks = []
            # Serial N8 precedes concurrent N32. Force the diagnostic branch
            # to test all ownership transitions under actual request overlap.
            # Real online selection and its SLO checks remain unchanged.
            for case, count in (('accept', 8), ('reject', 8), ('accept', 32), ('reject', 32)):
                test = copy.deepcopy(cfg)
                suffix = '_concurrent' if count >= 32 else ''
                test.update(contract, name=f'h_{model}_{axis}_forced_{case}{suffix}_state{count}', count=count,
                    phase='EXACT_DYNAMIC_BRANCH_DIAGNOSTIC', calibration_serial=count < 32,
                    arrival_offsets=[0.] * count, arrival_rate=None, arrival_seed=None,
                    request_caps=test['request_caps'][:count], prompt_ids=test['prompt_ids'][:count],
                    qualification_force_partial_key=partial_key,
                    qualification_reject_frontier=case == 'reject')
                save(test)
                reference = (f'c_{model}_native_timed_state32' if count == 32
                             else f'a_{model}_atomic_native_origin_v1_state8')
                check = qualify(test['name'], reference)
                if check['passed']:
                    with quiet_analysis(test['name'] + '_native_lifetime'):
                        check['lifetime'] = lifetime(test['name'])
                        from .dynamic_branch_gate import check as branch_gate
                        check['branch'] = branch_gate(test['name'], case, partial_key)
                        check['passed'] = check['lifetime']['passed'] and check['branch']['passed']
                checks.append(check)
                record(axis + '_branch_checks', checks)
                if not check['passed']:
                    break
            if len(checks) != 4 or not all(r['passed'] for r in checks):
                record(axis + '_status', 'BLOCKED_EXACT_BRANCH_GATE')
                continue
            test = copy.deepcopy(cfg)
            test.update(contract, name=f'h_{model}_{axis}_online_state32', phase='OBSERVED_ONLINE_N32_EXACTNESS')
            save(test)
            check = qualify(test['name'], f'c_{model}_native_timed_state32')
            if check['passed']:
                with quiet_analysis(test['name'] + '_native_lifetime'):
                    check['lifetime'] = lifetime(test['name'])
                    check['passed'] = check['lifetime']['passed']
            record(axis + '_online32', check)
            if not check['passed']:
                record(axis + '_status', 'BLOCKED_ONLINE_N32_GATE')
                continue
            record(axis + '_status', 'EXACT_ONLINE_N32_QUALIFIED')
            ready.append(dict(model=model, axis=axis, label=label, arm=cfg['arm'], contract=contract))
    # Finish the independent model/axis gates before N128 expansion. The most
    # informative paired loads precede the long low-load arrival windows.
    ledger['stage'] = 'PAIRED_N128_HEAVY_MEDIUM_LOW'
    write(target, ledger)
    for load in ('heavy', 'medium', 'low'):
        for task in ready:
            model, axis = task['model'], task['axis']
            row = ledger['models'][model]
            if row.get(axis + '_status') != 'EXACT_ONLINE_N32_QUALIFIED':
                row['serving'].append(dict(axis=axis, load=load, status='BLOCKED_EARLIER_ONLINE_SERVING_FAILURE'))
                write(target, ledger)
                continue
            test = serving_configuration(task, load)
            save(test)
            complete = execute(test['name'])
            result = dict(name=test['name'], model=model, axis=axis, load=load, complete=complete, passed=False)
            if complete:
                reference = next(r['name'] for r in static['models'][model]['serving'] if r.get('plan') == 'native'
                                 and r.get('load') == load and r.get('passed'))
                with quiet_analysis(test['name'] + '_online_metrics'):
                    result['output'] = evidence.compare(reference, test['name'])
                    result['work'] = evidence.full_work(test['name'])
                    result['lifetime'] = lifetime(test['name'])
                    result['metrics'] = summarize(test['name'])
                    result['accounting'] = accounting(test['name'])
                    result['passed'] = all(result[key]['passed'] for key in ('output', 'work', 'lifetime'))
            else:
                receipt = json.loads((ROOT / 'raw' / test['name'] / 'dispatcher_result.json').read_text())
                result['status'] = receipt.get('status', 'EXECUTION_FAILED')
            row['serving'].append(result)
            write(target, ledger)
            if not result['passed']:
                row[axis + '_status'] = ('NOT_RUN_TIME_BUDGET' if result.get('status') == 'NOT_RUN_TIME_BUDGET'
                                        else 'BLOCKED_ONLINE_SERVING_FAILURE')
                write(target, ledger)
    ledger['status'] = 'DYNAMIC_CAMPAIGN_FINISHED'
    write(target, ledger)


if __name__ == '__main__':
    main()
