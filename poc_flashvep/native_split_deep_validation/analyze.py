"""Analyze retained results without turning missing or rejected evidence into a win."""
import csv
import math
import statistics
from collections import Counter
from .common import *
from .granularity import account, structural_attention


def csv_write(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def ratio(values):
    result = dict(starts=len(values), start_ratios=values)
    if not values:
        return result
    if any(value <= 0 for value in values):
        result.update(geometric_mean_ratio=0. if 0. in values else None,
            scope='A zero goodput ratio cannot enter a log-ratio t interval. Zero values remain reported. No positive-only interval is substituted.')
        return result
    logs = [math.log(value) for value in values]
    center = statistics.mean(logs)
    result['geometric_mean_ratio'] = math.exp(center)
    critical = {3: 4.302652729911275, 4: 3.182446305284263, 5: 2.7764451051977987}
    if len(logs) in critical:
        half = critical[len(logs)] * statistics.stdev(logs) / math.sqrt(len(logs))
        result['ratio_CI95'] = [math.exp(center - half), math.exp(center + half)]
    result['scope'] = 'Paired independent-start log-ratio t interval. Requests and tokens are not independent starts. Small-sample normal-log-ratio assumption.'
    return result


def main():
    initialize()
    prior.parent.prior.preserve()
    inventory, serving, graphs, work, memory, tiles, failures, attention, warmups, payloads, state_graphs, wire_rows = [], [], [], [], [], [], [], [], [], [], [], []
    queue_summaries, queue_timeline = [], []
    handoff_requests, handoff_summaries = [], []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if not isinstance(cfg, dict) or 'name' not in cfg:
            continue
        folder = ROOT / 'raw' / cfg['name']
        receipt_path = folder / 'dispatcher_result.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        row = dict(name=cfg['name'], model=cfg['model'], method=cfg['method'], count=cfg['count'],
            phase=cfg['phase'], split=cfg['data_split'], markers=cfg['observe_service_timing'],
            complete=receipt.get('complete', False), status=receipt.get('status', 'NOT_COMPLETED'),
            elapsed_job_wall_seconds=receipt.get('elapsed_seconds'), configuration_SHA256=digest(path))
        inventory.append(row)
        attention.append(dict(name=cfg['name'], **structural_attention(cfg)))
        for basename in ('deep_primary_warmup.jsonl', 'deep_D_geometry_warmup.jsonl', 'graph_warmup_accounting.jsonl'):
            source = folder / basename
            if source.exists():
                for line in source.read_text().splitlines():
                    warmups.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                        graph_budget=cfg['deep_graph_budget'], source=basename, **json.loads(line)))
        if receipt and not receipt.get('complete'):
            log = (ROOT / 'logs' / (cfg['name'] + '.log')).read_text(errors='replace')
            errors = [line for line in log.splitlines()
                      if line.startswith(('ValueError:', 'TypeError:', 'RuntimeError:', 'AttributeError:',
                                          'AssertionError:', 'torch.OutOfMemoryError:'))]
            failures.append(dict(row, errors=errors,
                OOM='torch.OutOfMemoryError:' in log,
                measurement_started=(folder / 'measurement_begin.json').exists()))
        if receipt.get('complete'):
            if cfg.get('state_probe'):
                state_gate = folder / 'qualification.json'
                state_passed = json.loads(state_gate.read_text()).get('passed') if state_gate.exists() else None
                for audit in folder.glob('deep_graph_cache_*.json'):
                    value = json.loads(audit.read_text())
                    for geometry in value['geometries'].values():
                        if geometry['measured']:
                            state_graphs.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                                role=value['role'], graph_budget=cfg['deep_graph_budget'], qualification_passed=state_passed,
                                **geometry, scope='Recorded heavy-state test coverage. Unseen startup-warmed geometries do not inherit an exact-state claim. Not clean latency.'))
            begin = folder / 'measurement_begin.json'
            if begin.exists():
                since = json.loads(begin.read_text())['time']
                expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
                records = [json.loads(line) for source in folder.glob('migration_*.jsonl')
                           for line in source.read_text().splitlines()]
                records = [value for value in records if value['seed'] in expected and value['time'] >= since]
                if records:
                    payloads.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                        count=cfg['count'], transfer_records=len(records), measured_seed_coverage=len({value['seed'] for value in records}),
                        KV_bytes=sum(value['KV_bytes'] for value in records),
                        hidden_bytes=sum(value['hidden_bytes'] for value in records),
                        recorded_fragments=sum(value['fragments'] for value in records),
                        recorded_descriptors=sum(value['descriptors'] for value in records),
                        phase=cfg['phase'],
                        scope='Recorded source transfer regions and partition counts. Excludes control messages and network headers. Not exposed latency. Native paths without this record remain NOT_MEASURED in this component table.'))
        gate_path = folder / 'deep_serving_gate.json'
        if not gate_path.exists():
            continue
        gate = json.loads(gate_path.read_text())
        if not gate.get('complete'):
            continue
        if cfg['data_split'] == 'heldout' and not cfg.get('oracle_calibration'):
            from .handoff_receipts import analyze as handoff
            requests, summary = handoff(folder, cfg['count'])
            metadata = dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                markers=cfg['observe_service_timing'], start=cfg['independent_start'], load=cfg['load'])
            handoff_requests.extend(dict(metadata, **row) for row in requests)
            handoff_summaries.append(dict(metadata, **summary,
                scope='Native host receipt intervals and projected model-stream continuation gaps. Not exclusive GPU DMA, CPU work, or full physical buffer lifetime. Continuation gaps include scheduling, preparation, and device queueing. Negative observer-order intervals remain visible. Do not add percentiles.'))
        window_path = folder / 'window.json'
        if window_path.exists():
            from ..looped_pd_headroom_recovery import queues
            queues.ROOT = ROOT
            summary, timeline = queues.analyze(cfg['name'], json.loads(window_path.read_text()))
            queue_summaries.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                phase=cfg['phase'], **summary))
            queue_timeline.extend(timeline)
        serving.append(dict(gate['metrics'], name=cfg['name'], model=cfg['model'], method=cfg['method'],
            phase=cfg['phase'], data_split=cfg['data_split'], markers=cfg['observe_service_timing'],
            start=cfg['independent_start'], load=cfg['load'], count=cfg['count'],
            correctness_passed=gate['passed'], graph_coverage_passed=gate['performance_eligible_graphs'],
            calibration=bool(cfg.get('oracle_calibration')), cache_policy=cfg['deep_graph_policy'],
            oracle_validation_only=bool(cfg.get('oracle_validation_only')),
            graph_budget=cfg['deep_graph_budget']))
        work.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'], **gate['actual_work']))
        for audit in folder.glob('wire_payload_*.json'):
            value = json.loads(audit.read_text())
            native = gate['actual_work']['counters'][value['role']]
            measured = value['counts']['measured']
            total = measured['KV_bytes'] + measured['hidden_bytes']
            wire_rows.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'], role=value['role'],
                count=cfg['count'], phase=cfg['phase'], **measured,
                measured_bytes_match_native_counter=total == native['transport_bytes_sent'],
                submissions_match_native_counter=measured['accepted_submissions'] == native['transport_completed_count'],
                submitted_equals_completed_at_close=value['submitted_equals_completed_at_close'],
                scope=value['scope']))
        for item in gate['graphs']['roles']:
            graphs.append(dict(name=cfg['name'], model=cfg['model'], method=cfg['method'],
                **{key: value for key, value in item.items() if key != 'measured'}, **item['measured']))
        tiles.extend(account(cfg['name']))
        for pattern, moment in (('memory_inventory_initial_*.json', 'INITIAL_STAGE'),
                                ('native_factory_memory_*.json', 'AFTER_CONNECTOR')):
            for source in folder.glob(pattern):
                value = json.loads(source.read_text())
                denominator = 81559 * 1024**2
                amount = {key: number for key, number in value.items()
                          if key.endswith('_bytes') and isinstance(number, (float, int))}
                memory.append(dict(name=cfg['name'], role=value['role'], moment=moment,
                    **{key + '_GiB': number / 1024**3 for key, number in amount.items()},
                    **{key + '_physical_HBM_percent': 100 * number / denominator for key, number in amount.items()},
                    physical_HBM_GiB=denominator / 1024**3,
                    scope='Component and allocator totals overlap. Not an additive HBM breakdown or complete opaque graph attribution.'))
    for name, rows in (('configuration_inventory', inventory), ('serving_results', serving),
                       ('graph_cache_results', graphs), ('work_and_payload', work),
                       ('gpu_memory_gib_percent', memory), ('fixed_tile_accounting', tiles),
                       ('structural_attention_pairs', attention),
                       ('warmup_inventory', warmups),
                       ('transfer_payload_components', payloads), ('exact_state_graph_coverage', state_graphs),
                       ('actual_wire_payload_accounting', wire_rows),
                       ('P_ready_and_admission_summary', queue_summaries),
                       ('P_ready_and_admission_timeline', queue_timeline),
                       ('native_handoff_receipt_requests', handoff_requests),
                       ('native_handoff_receipt_summary', handoff_summaries),
                       ('failure_inventory', failures)):
        write(ROOT / (name + '.json'), rows)
        csv_write(ROOT / (name + '.csv'), rows)
    confirmation = [row for row in serving if row['data_split'] == 'heldout' and row['load'] == 'medium'
                    and row['correctness_passed'] and not row['calibration'] and not row['oracle_validation_only']]
    paired, markers = {}, []
    fields = ['TTFT_p95_ms', 'service_TTFT_ms_p95', 'E2E_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms',
              'joint_SLO_goodput', 'requests_per_second']
    for model in ('ouro', 'huginn'):
        paired[model] = {}
        for observed in (False, True):
            selected = [row for row in confirmation if row['model'] == model and row['markers'] == observed]
            native = {row['start']: row for row in selected if row['method'] == 'native'}
            methods = sorted({row['method'] for row in selected})
            section = paired[model]['marker_on' if observed else 'marker_free'] = {}
            for method in methods:
                candidates = {row['start']: row for row in selected if row['method'] == method}
                section[method] = {}
                for field in fields:
                    values = [candidates[start][field] / native[start][field]
                        for start in sorted(native.keys() & candidates.keys())
                        if field in native[start] and field in candidates[start]
                        and native[start][field] is not None and candidates[start][field] is not None
                        and native[start][field] > 0]
                    section[method][field] = ratio(values)
                    section[method][field]['undefined_zero_native_denominator_starts'] = [start
                        for start in sorted(native.keys() & candidates.keys())
                        if native[start].get(field) == 0]
            if not observed:
                continue
            for row in selected:
                controls = [value for value in confirmation if value['model'] == model
                    and value['method'] == row['method'] and value['start'] == row['start'] and not value['markers']]
                if not controls:
                    continue
                control = controls[0]
                wall = 100 * (row['response_span_s'] / control['response_span_s'] - 1)
                markers.append(dict(model=model, method=row['method'], start=row['start'],
                    on=row['name'], off=control['name'], wall_change_percent=wall,
                    TTFT_p95_change_percent=100 * (row['TTFT_p95_ms'] / control['TTFT_p95_ms'] - 1),
                    ITL_p99_change_percent=100 * (row['ITL_p99_ms'] / control['ITL_p99_ms'] - 1),
                    global_wall_screen_passed=abs(wall) <= 5,
                    scope='Global wall screen does not bound local request-tail perturbation.'))
    write(ROOT / 'paired_confirmation_ratios.json', paired)
    pairwise, partial_token_pairs = {}, {}
    frozen = ROOT / 'state/frozen_static_selection.json'
    selected = json.loads(frozen.read_text())['selected'] if frozen.exists() else {}
    token_reference_path = ROOT / 'state/frozen_safe_token_reference.json'
    if selected and not token_reference_path.exists() and not list((ROOT / 'raw').glob('d_*_heldout*')):
        references = {}
        for model in selected:
            candidates = [row for row in serving if row['model'] == model and row['data_split'] == 'discovery'
                and row['name'].startswith('c_') and row['count'] == 128
                and row['start'] == 1 and not row['markers'] and row['correctness_passed']
                and row['graph_coverage_passed'] and row['Decode_SLO_screen']
                and (row['method'] in ('native', 'zero') or row['method'].startswith('token'))]
            if candidates:
                best = max(candidates, key=lambda row: (row['joint_SLO_goodput'], -row['TTFT_p95_ms']))
                references[model] = dict(method=best['method'], discovery_run=best['name'])
        write(token_reference_path, dict(time=time.time(), references=references,
            frozen_before_heldout=True,
            scope='Choose the safe Token reference from discovery only. Native and feature-enabled zero-offload are both valid full-prefix endpoints. All paired endpoint controls remain reported.'))
    token_references = json.loads(token_reference_path.read_text())['references'] if token_reference_path.exists() else {}
    for model, choices in selected.items():
        token_choice = choices.get('token', {})
        diagnostic = token_choice.get('method')
        if not diagnostic:
            continue
        token = token_references.get(model, {}).get('method',
            diagnostic if token_choice.get('Decode_safe') else 'native')
        sample = [row for row in confirmation if row['model'] == model and not row['markers']]
        reference = {row['start']: row for row in sample if row['method'] == token}
        pairwise[model] = dict(reference=token, comparisons={},
            diagnostic_only_partial=None if token_choice.get('Decode_safe') else diagnostic,
            reference_frozen_before_heldout=model in token_references,
            scope='Safe Token references include Native and zero-offload full-prefix endpoints. A Decode-unsafe diagnostic partial is not a safe baseline. Cut validity is limited to the declared candidate budget.')
        for method in sorted({row['method'] for row in sample if row['method'] != token}):
            candidate = {row['start']: row for row in sample if row['method'] == method}
            item = pairwise[model]['comparisons'][method] = {}
            for field in fields:
                values = [candidate[start][field] / reference[start][field]
                    for start in sorted(reference.keys() & candidate.keys())
                    if isinstance(reference[start].get(field), (float, int))
                    and isinstance(candidate[start].get(field), (float, int)) and reference[start][field] > 0]
                item[field] = ratio(values)
                item[field]['undefined_zero_reference_denominator_starts'] = [start
                    for start in sorted(reference.keys() & candidate.keys()) if reference[start].get(field) == 0]
        partial_reference = {row['start']: row for row in sample if row['method'] == diagnostic}
        partial_token_pairs[model] = dict(reference=diagnostic, comparisons={},
            actual_Decode_screen_passes=sum(row['Decode_SLO_screen'] for row in partial_reference.values()),
            actual_reference_starts=len(partial_reference),
            scope='Direct comparison with the prospectively selected Token partial. Discovery eligibility does not replace the measured held-out Decode screen. This is separate from the frozen safe-endpoint reference.')
        for method in sorted({row['method'] for row in sample if row['method'] != diagnostic}):
            candidate = {row['start']: row for row in sample if row['method'] == method}
            item = partial_token_pairs[model]['comparisons'][method] = {}
            for field in fields:
                values = [candidate[start][field] / partial_reference[start][field]
                    for start in sorted(partial_reference.keys() & candidate.keys())
                    if isinstance(partial_reference[start].get(field), (float, int))
                    and isinstance(candidate[start].get(field), (float, int)) and partial_reference[start][field] > 0]
                item[field] = ratio(values)
    write(ROOT / 'paired_token_reference_ratios.json', pairwise)
    write(ROOT / 'paired_selected_token_partial_ratios.json', partial_token_pairs)
    zero_pairs = {}
    for model in ('ouro', 'huginn'):
        sample = [row for row in confirmation if row['model'] == model and not row['markers']]
        reference = {row['start']: row for row in sample if row['method'] == 'zero'}
        zero_pairs[model] = {}
        for method in sorted({row['method'] for row in sample if row['method'] != 'zero'}):
            candidate = {row['start']: row for row in sample if row['method'] == method}
            zero_pairs[model][method] = {}
            for field in fields:
                zero_pairs[model][method][field] = ratio([candidate[start][field] / reference[start][field]
                    for start in sorted(reference.keys() & candidate.keys())
                    if isinstance(candidate[start].get(field), (int, float))
                    and isinstance(reference[start].get(field), (int, float)) and reference[start][field] > 0])
    write(ROOT / 'paired_zero_path_reference_ratios.json', dict(results=zero_pairs,
        scope='Feature-enabled full-depth zero-offload control. Native versus zero separates the connector/transport package from actual recurrent redistribution. Compare start-level ratios, not differences of component percentiles.'))
    write(ROOT / 'marker_interference.json', markers)
    csv_write(ROOT / 'marker_interference.csv', markers)
    accepted = {row['on'] for row in markers if row['global_wall_screen_passed']}
    accepted_pairs, shifts = {}, {}
    for model in ('ouro', 'huginn'):
        observed = [row for row in confirmation if row['model'] == model and row['markers']
                    and row['name'] in accepted and row['graph_coverage_passed']]
        native = {row['start']: row for row in observed if row['method'] == 'native'}
        off = {(row['method'], row['start']): row for row in confirmation
               if row['model'] == model and not row['markers']}
        accepted_pairs[model], shifts[model] = {}, {}
        for method in sorted({row['method'] for row in observed}):
            candidate = {row['start']: row for row in observed if row['method'] == method}
            accepted_pairs[model][method], shifts[model][method] = {}, {}
            for field in fields:
                accepted_pairs[model][method][field] = ratio([candidate[start][field] / native[start][field]
                    for start in sorted(candidate.keys() & native.keys())
                    if isinstance(candidate[start].get(field), (int, float))
                    and isinstance(native[start].get(field), (int, float)) and native[start][field] > 0])
                differences = []
                for start in sorted(candidate.keys() & native.keys()):
                    left, right = off.get((method, start)), off.get(('native', start))
                    if not left or not right:
                        continue
                    values = [candidate[start].get(field), native[start].get(field), left.get(field), right.get(field)]
                    if all(isinstance(value, (int, float)) and value > 0 for value in values):
                        differences.append((values[0] / values[1]) / (values[2] / values[3]))
                shifts[model][method][field] = ratio(differences)
    write(ROOT / 'paired_marker_accepted_ratios.json', accepted_pairs)
    write(ROOT / 'marker_relative_advantage_shifts.json', dict(results=shifts,
        scope='Candidate/Native ratio under ON divided by its matched OFF ratio. Values below one indicate a larger apparent relative gain under markers. Counterbalanced finite paired runs do not remove all scheduler noise.'))
    from .prepare import ACTIONS
    present = {row['name']: row for row in inventory}
    planned = []
    def plan(name, model, method, cohort):
        actual = present.get(name)
        planned.append(dict(name=name, model=model, method=method, count=cohort,
            status=('COMPLETED_DISPATCH_CHECK_OTHER_GATES' if actual['complete'] else actual['status'])
                if actual else 'NOT_PREPARED_OR_NOT_RUN_TIME_BUDGET',
            scope='Planned condition. Dispatch completion alone is not a scientific or correctness pass.'))
    for model, methods in ACTIONS.items():
        for method in methods:
            for count in (8, 32):
                plan(f'b_{model}_{method}_reserved8_state{count}', model, method, count)
            for start in range(1, 4 if model == 'ouro' else 2):
                plan(f'c_{model}_{method}_discovery_s{start}_off128', model, method, 128)
            for mode, count in (('serial', 8), ('burst', 24)):
                plan(f'f_{model}_{method}_{mode}_cost{count}', model, method, count)
        choices = list(dict.fromkeys(['native', 'zero'] + [row['method'] for row in selected.get(model, {}).values()]
                                    + (['loop31'] if model == 'huginn' else [])))
        for method in choices:
            for start in range(1, 6 if model == 'huginn' else 4):
                for marker in (('off', 'on') if start <= 3 else ('off',)):
                    plan(f'd_{model}_{method}_heldout_s{start}_{marker}128', model, method, 128)
            for load in ('low', 'heavy'):
                plan(f'e_{model}_{method}_{load}_heldout_off128', model, method, 128)
    write(ROOT / 'planned_matrix_status.json', planned)
    csv_write(ROOT / 'planned_matrix_status.csv', planned)
    completed = [row for row in inventory if row['complete']]
    closed = (ROOT / 'state/gpu_campaign_closed.json').exists()
    summary = dict(time=time.time(), elapsed_hours=(time.time() - START) / 3600,
        status='GPU_MEASUREMENTS_CLOSED' if closed else 'IN_PROGRESS',
        configurations=len(inventory), completed_dispatches=len(completed), failed_dispatches=len(failures),
        completed_offered_requests=sum(row['count'] for row in completed),
        completed_count_distribution=dict(Counter(row['count'] for row in completed)),
        main_planned_conditions=len(planned),
        main_unprepared_conditions=sum(row['status'] == 'NOT_PREPARED_OR_NOT_RUN_TIME_BUDGET' for row in planned),
        marker_free_heldout_medium_runs=sum(not row['markers'] for row in confirmation),
        overall_deadline_epoch=DEADLINE, GPU_stop_epoch=GPU_DEADLINE,
        kernel_union='NOT_MEASURED_UNLESS_SEPARATE_PROFILE_IS_QUALIFIED',
        scope=('Final measurement analysis. Publication has a separate receipt. ' if closed else
               'Intermediate analysis. ') +
              'Dispatch completion is separate from correctness, graph, Decode, and model-validation gates.')
    write(ROOT / 'analysis_status.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
