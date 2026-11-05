"""Freeze calibration before confirmation and reject an inaccurate offline model."""
import random
import statistics
from .common import *
from .simulator import Simulation, lookup, value, percentile


def fit_and_freeze(model_name='oracle_cost_model_frozen.json', predictions_name='predictions'):
    initialize()
    path = ROOT / model_name
    if path.exists():
        return json.loads(path.read_text())
    if model_name == 'oracle_cost_model_frozen.json':
        assert not list((ROOT / 'raw').glob('d_*_heldout*/dispatcher_result.json')), 'Do not fit on confirmation outcomes'
    else:
        assert not list((ROOT / predictions_name).glob('*.json')), 'Freeze before this version predicts validation runs'
    from .costs import fit
    model = fit()
    for name, costs in model['models'].items():
        for method, action in costs['actions'].items():
            cfg = json.loads((ROOT / 'configs' / f'f_{name}_{method}_serial_cost8.json').read_text())
            frontier = cfg.get('native_frontier')
            query = cfg['length'] - frontier[1] if frontier else cfg['length']
            action['D_max_batch'] = min(cfg['D_slots'], cfg['D_token_budget'] // query)
            action['P_query_rows'] = {phase: (frontier[1] if frontier and
                (frontier[0] == 0 or phase == 'recurrent') else cfg['length']) for phase in action['P_steps']}
            action['D_query_rows'] = query if action['D_prompt'] else None
            action['frontier'] = frontier
            action['loop_cut'] = cfg.get('native_partial_cut')
        costs['graph_budget'] = cfg['deep_graph_budget']
    model.update(frozen_before_confirmation=(model_name == 'oracle_cost_model_frozen.json'),
        frozen_before_version_specific_predictions=True, model_file=model_name,
        fitting_inputs='Independent calibration only. No held-out outcomes enter fitting.',
        model_code_SHA256={name: digest(SOURCE / name) for name in ('costs.py', 'simulator.py', 'oracle.py')},
        validation_status='NOT_YET_VALIDATED',
        model_limitations=[
            'Serialized resource approximation. Physical stream overlap is not modeled exactly.',
            'Missing batch sizes use the nearest larger measured cost or linear extrapolation.',
            'Model residuals are not exclusive GPU or CPU time.',
            'Mixed actions use homogeneous batches. A mixed-frontier GPU scheduler is not qualified.',
            'The graph-count screen is necessary, not sufficient. Mixed-plan opaque graph/workspace HBM and warm-up feasibility are not GPU-validated.',
            'Calibration is finite. Static validation cannot prove every dynamic-plan error bound.'])
    write(path, model)
    return model


def predict_configuration(cfg, model_name='oracle_cost_model_frozen.json', predictions_name='predictions'):
    cfg.setdefault('deep_retire_geometry_hook_after_warmup', True)
    try:
        return _predict_configuration(cfg, model_name, predictions_name)
    except Exception:
        import traceback
        result = dict(name=cfg['name'], status='NOT_PREDICTED_MODEL_EXCEPTION',
            time=time.time(), traceback=traceback.format_exc(),
            scope='CPU prediction failed. Qualified GPU static comparison remains the higher priority. This run cannot validate an absent prediction.')
        write(ROOT / predictions_name / (cfg['name'] + '.json'), result)
        return result


def _predict_configuration(cfg, model_name='oracle_cost_model_frozen.json', predictions_name='predictions'):
    initialize()
    save(cfg)
    path = ROOT / predictions_name / (cfg['name'] + '.json')
    model_path = ROOT / model_name
    config_path = ROOT / 'configs' / (cfg['name'] + '.json')
    if not model_path.exists():
        result = dict(name=cfg['name'], status='NOT_PREDICTED_MODEL_UNAVAILABLE',
                      configuration_SHA256=digest(config_path), time=time.time())
        write(path, result)
        return result
    if path.exists():
        result = json.loads(path.read_text())
        if 'configuration_SHA256' in result:
            assert result['configuration_SHA256'] == digest(config_path)
        if 'cost_model_SHA256' in result:
            assert result['cost_model_SHA256'] == digest(model_path)
        return result
    folder = ROOT / 'raw' / cfg['name']
    if folder.exists():
        result = dict(name=cfg['name'], status='EXCLUDED_DISPATCH_PRECEDES_THIS_MODEL_VERSION',
            configuration_SHA256=digest(config_path), cost_model_SHA256=digest(model_path),
            cost_model_file=model_name, time=time.time(),
            scope='Retain earlier measured serving results. Do not create a retrospective prediction for this version.')
        write(path, result)
        return result
    frozen = json.loads(model_path.read_text())
    model = frozen['models'].get(cfg['model'])
    if model is None or cfg['method'] not in model['actions']:
        result = dict(name=cfg['name'], status='NOT_PREDICTED_MISSING_CALIBRATION',
            configuration_SHA256=digest(config_path), cost_model_SHA256=digest(model_path), time=time.time())
        write(path, result)
        return result
    slo = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[cfg['model'] + '_L' + str(cfg['length'])]
    result = Simulation(model, [1000 * value for value in cfg['arrival_offsets']],
                        [cfg['method']] * cfg['count'], slo).run()
    if folder.exists():
        result = dict(name=cfg['name'], status='EXCLUDED_DISPATCH_RACE_DURING_CPU_PREDICTION',
            configuration_SHA256=digest(config_path), cost_model_SHA256=digest(model_path),
            cost_model_file=model_name, time=time.time())
        write(path, result)
        return result
    result.update(name=cfg['name'], time=time.time(), configuration_SHA256=digest(config_path),
        cost_model_SHA256=digest(model_path),
        cost_model_file=model_name, predicted_before_GPU_dispatch=True,
        status='UNVALIDATED_OFFLINE_PREDICTION_NOT_MEASURED')
    write(path, result)
    return result


def start_block_intervals(rows):
    """Resample independent starts, not individual requests or correlated tokens."""
    output = {}
    for model in ('ouro', 'huginn'):
        for kind in ('marker_free', 'instrumented_service'):
            sample = [row for row in rows if row['model'] == model and row['load'] == 'medium'
                and ((not row['markers']) if kind == 'marker_free' else row['metric'] == 'service_TTFT_ms_p95')]
            blocks = {start: [row['absolute_relative_error'] for row in sample if row['start'] == start]
                      for start in sorted({row['start'] for row in sample})}
            result = dict(independent_starts=len(blocks), cells=len(sample),
                scope='Start-cluster percentile bootstrap. Resampling does not remove systematic model error. Fewer than three starts has no reported interval.')
            if len(blocks) >= 3:
                rng = random.Random(2026110529)
                keys = list(blocks)
                centers, tails = [], []
                for _ in range(2000):
                    values = [value for start in rng.choices(keys, k=len(keys)) for value in blocks[start]]
                    centers.append(statistics.median(values))
                    tails.append(percentile(values, .95))
                result.update(median_absolute_relative_error_CI95=[percentile(centers, .025), percentile(centers, .975)],
                              p95_absolute_relative_error_CI95=[percentile(tails, .025), percentile(tails, .975)])
            output[model + '_' + kind] = result
    return output


def validate(predictions_name='predictions'):
    initialize()
    rows = []
    false_safe = []
    rejected_markers = []
    fields = {'TTFT_p95_ms': 'TTFT_p95_ms', 'TPOT_p95_ms': 'TPOT_p95_ms',
              'ITL_p99_ms': 'ITL_p99_ms', 'E2E_p95_ms': 'E2E_p95_ms',
              'service_TTFT_ms_p95': 'service_TTFT_p95_ms'}
    for prediction in sorted((ROOT / predictions_name).glob('*.json')):
        predicted = json.loads(prediction.read_text())
        if 'metrics' not in predicted:
            continue
        name = predicted['name']
        gate_path = ROOT / 'raw' / name / 'deep_serving_gate.json'
        if not gate_path.exists():
            continue
        gate = json.loads(gate_path.read_text())
        if not gate.get('passed'):
            continue
        config_path = ROOT / 'configs' / (name + '.json')
        assert predicted['configuration_SHA256'] == digest(config_path), 'Validation configuration changed after prediction'
        if predicted.get('cost_model_file'):
            assert predicted['cost_model_SHA256'] == digest(ROOT / predicted['cost_model_file']), 'Frozen cost model changed'
        receipt = json.loads((gate_path.parent / 'dispatcher_result.json').read_text())
        dispatch_start = receipt['time'] - receipt['elapsed_seconds']
        assert predicted['time'] < dispatch_start, 'Prediction does not precede estimated subprocess dispatch'
        cfg = json.loads(config_path.read_text())
        actual = gate['metrics']
        if cfg['observe_service_timing']:
            control = ROOT / 'raw' / name.replace('_on128', '_off128') / 'deep_serving_gate.json'
            if not control.exists():
                rejected_markers.append(dict(name=name, reason='MATCHED_MARKER_FREE_CONTROL_MISSING'))
                continue
            baseline = json.loads(control.read_text())
            if not baseline.get('passed'):
                rejected_markers.append(dict(name=name, reason='MATCHED_CONTROL_NOT_QUALIFIED'))
                continue
            change = actual['response_span_s'] / baseline['metrics']['response_span_s'] - 1
            if abs(change) > .05:
                rejected_markers.append(dict(name=name, reason='GLOBAL_MARKER_INTERFERENCE_ABOVE_5_PERCENT',
                                            measured_change_percent=100 * change))
                continue
        if not predicted['predicted_before_GPU_dispatch']:
            raise RuntimeError('A retrospective prediction cannot enter held-out validation')
        if predicted['metrics']['Decode_SLO_screen'] and not actual['Decode_SLO_screen']:
            false_safe.append(name)
        for measured_key, predicted_key in fields.items():
            if measured_key not in actual or actual[measured_key] is None:
                continue
            observed = actual[measured_key]
            estimate = predicted['metrics'][predicted_key]
            rows.append(dict(name=name, model=cfg['model'], method=cfg['method'], start=cfg['independent_start'],
                load=cfg['load'], markers=cfg['observe_service_timing'], metric=measured_key,
                prediction_epoch=predicted['time'], estimated_dispatch_epoch=dispatch_start,
                measured=observed, predicted=estimate, relative_error=(estimate - observed) / observed,
                absolute_relative_error=abs(estimate - observed) / observed))
    clean = [row['absolute_relative_error'] for row in rows if not row['markers']]
    service = [row['absolute_relative_error'] for row in rows if row['metric'] == 'service_TTFT_ms_p95']
    median = statistics.median(clean) if clean else None
    tail = percentile(clean, .95) if clean else None
    service_median = statistics.median(service) if service else None
    service_tail = percentile(service, .95) if service else None
    coverage = {}
    service_coverage = {}
    for model_name in ('ouro', 'huginn'):
        for family in ('native', 'token', 'loop', '2d_'):
            starts = {row['start'] for row in rows if row['model'] == model_name and not row['markers']
                      and row['load'] == 'medium' and row['metric'] == 'TTFT_p95_ms'
                      and (row['method'] == family if family == 'native' else row['method'].startswith(family))}
            coverage[model_name + '_' + family] = len(starts)
            observed = {row['start'] for row in rows if row['model'] == model_name and row['markers']
                        and row['load'] == 'medium' and row['metric'] == 'service_TTFT_ms_p95'
                        and (row['method'] == family if family == 'native' else row['method'].startswith(family))}
            service_coverage[model_name + '_' + family] = len(observed)
    by_model = {}
    for model_name in ('ouro', 'huginn'):
        observed = [row['absolute_relative_error'] for row in rows if row['model'] == model_name and not row['markers']]
        instrumented = [row['absolute_relative_error'] for row in rows if row['model'] == model_name and row['metric'] == 'service_TTFT_ms_p95']
        by_model[model_name] = dict(
            median_absolute_relative_error=statistics.median(observed) if observed else None,
            p95_absolute_relative_error=percentile(observed, .95) if observed else None,
            service_median_absolute_relative_error=statistics.median(instrumented) if instrumented else None,
            service_p95_absolute_relative_error=percentile(instrumented, .95) if instrumented else None,
            error_screen_passed=bool(observed and instrumented) and statistics.median(observed) <= .15
                and percentile(observed, .95) <= .25 and statistics.median(instrumented) <= .15
                and percentile(instrumented, .95) <= .25)
    passed = (bool(clean and service) and median <= .15 and tail <= .25
              and service_median <= .15 and service_tail <= .25 and not false_safe
              and all(count >= 3 for count in coverage.values())
              and all(count >= 3 for count in service_coverage.values())
              and all(row['error_screen_passed'] for row in by_model.values()))
    result = dict(time=time.time(), predictions_directory=predictions_name,
        passed=passed, clean_cells=len(clean),
        median_absolute_relative_error=median, p95_absolute_relative_error=tail,
        service_median_absolute_relative_error=service_median, service_p95_absolute_relative_error=service_tail,
        medium_independent_start_coverage=coverage,
        accepted_service_independent_start_coverage=service_coverage,
        rejected_instrumented_controls=rejected_markers,
        by_model=by_model, start_block_error_intervals=start_block_intervals(rows),
        false_predicted_Decode_safe=false_safe, rows=rows,
        status='STATIC_VALIDATION_SCREEN_PASSED' if passed else 'MODEL_VALIDATION_FAILED_OR_INCOMPLETE',
        scope='Independent static confirmation predictions. Passing this screen is not proof of a globally optimal oracle or a qualified mixed-frontier runtime.')
    write(ROOT / 'oracle_static_validation.json', result)
    return result


def search_family(model, arrivals, slo, methods, initial=None):
    endpoints = ['native'] + (['zero'] if 'zero' in model['actions'] else [])
    candidates = list(dict.fromkeys(endpoints + methods))
    evaluations = 0
    best = None
    def scored(choices):
        nonlocal evaluations
        result = Simulation(model, arrivals, choices, slo).run()
        evaluations += 1
        metrics = result['metrics']
        score = (metrics.get('graph_budget_screen') is True, metrics['Decode_SLO_screen'],
                 metrics['joint_SLO_goodput'], -metrics['TTFT_p95_ms'])
        return score, choices[:], metrics
    for method in candidates:
        item = scored([method] * len(arrivals))
        if best is None or item[0] > best[0]:
            best = item
    if initial:
        for plan in initial:
            item = scored(plan)
            if item[0] > best[0]:
                best = item
    # Restricted offline search, not an exhaustive optimality certificate.
    block = 8
    budget = 96
    for left in range(0, len(arrivals), block):
        for method in candidates:
            if evaluations >= budget:
                break
            choices = best[1][:]
            choices[left:left + block] = [method] * len(choices[left:left + block])
            item = scored(choices)
            if item[0] > best[0]:
                best = item
        if evaluations >= budget:
            break
    return dict(choices=best[1], metrics=best[2], evaluations=evaluations,
        evidence='MODELED_RESTRICTED_OFFLINE_SEARCH_NOT_ONLINE_SPEEDUP',
        scope='Known future arrival trace and fixed output caps. Eight-request action blocks, 96-evaluation budget. Not exhaustive or globally optimal. Mixed-frontier GPU execution remains unqualified.')


def headroom(model_name='oracle_cost_model_frozen.json', predictions_name='predictions'):
    validation = validate(predictions_name)
    if not validation['passed']:
        result = dict(status='BLOCKED_MODEL_VALIDATION', headroom='NOT_ESTABLISHED',
            cost_model_file=model_name, cost_model_SHA256=digest(ROOT / model_name),
            predictions_directory=predictions_name,
            scope='Do not convert inaccurate static predictions into a feasible-oracle speedup.')
        write(ROOT / 'oracle_headroom.json', result)
        return result
    frozen = json.loads((ROOT / model_name).read_text())
    output = []
    for model_label, model in frozen['models'].items():
        cfg_path = ROOT / 'configs' / f'd_{model_label}_native_heldout_s1_off128.json'
        cfg = json.loads(cfg_path.read_text())
        arrivals = [1000 * v for v in cfg['arrival_offsets']]
        slo = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[model_label + '_L' + str(cfg['length'])]
        token = search_family(model, arrivals, slo, [m for m in model['actions'] if m.startswith('token')])
        loop = search_family(model, arrivals, slo, [m for m in model['actions'] if m.startswith('loop')])
        two = search_family(model, arrivals, slo,
            [m for m in model['actions'] if m not in ('native', 'zero')], [token['choices'], loop['choices']])
        output.append(dict(model=model_label, token=token, loop=loop, two_dimensional=two,
                           scope='Validated static-cost domain only. The candidate plan is not a measured online scheduler.'))
    result = dict(status='RESTRICTED_OFFLINE_CANDIDATES_AFTER_STATIC_VALIDATION', results=output,
        cost_model_file=model_name, predictions_directory=predictions_name)
    write(ROOT / 'oracle_headroom.json', result)
    return result


if __name__ == '__main__':
    headroom()
