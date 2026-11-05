"""Analyze retained measurements. Do not launch a model or infer missing costs."""
import csv
import math
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import *
from .dependency import Frontier


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def paired_ratio(values):
    result = dict(starts=len(values), ratios=values,
        scope='Paired independent-start ratios. Requests and tokens are not independent replications.')
    if not values:
        return result
    logs = [math.log(value) for value in values]
    result['geometric_mean_ratio'] = math.exp(statistics.mean(logs))
    if len(values) == 3:
        half = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
        result['ratio_CI95'] = [math.exp(statistics.mean(logs) - half), math.exp(statistics.mean(logs) + half)]
        result['assumption'] = 'Small-sample Student t interval on normal start-level log ratios. Selection uncertainty is not included.'
        result['equivalent_within_5_percent'] = result['ratio_CI95'][0] >= .95 and result['ratio_CI95'][1] <= 1.05
    return result


def graph_summary(folder):
    rows = []
    path = folder / 'native_prefill_graph_counts.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows.append(dict(role=row['role'], captures=row['captures'], replays=row['replays'],
                fallbacks_including_startup=row['fallbacks'],
                measured_successful_graph_calls=row['host_costs_after_measurement_marker']['calls'],
                scope='Fallback total includes startup. Measured successful graph calls are separate counters.'))
    return rows


def primitive_details(folder, cfg):
    by_role = {}
    for path in folder.glob('device_execution_*.json'):
        value = json.loads(path.read_text())
        selected = {}
        for row in value['rows']:
            if not row['measured'] or row['operation'] not in ('prelude', 'recurrent'):
                continue
            for request in row['requests']:
                if request['generated_tokens'] == 0 and request['token_count'] > 1:
                    selected.setdefault(request['seed'], []).append(row['stream_elapsed_ms'])
        by_role[value['role']] = dict(
            per_request_sum_ms={str(seed): sum(times) for seed, times in selected.items()},
            median_sum_ms=statistics.median(sum(times) for times in selected.values()) if selected else None)
    return dict(name=cfg['name'], model=cfg['model'], method=cfg['method'], streams=by_role,
        graphs=graph_summary(folder),
        scope='Serial disjoint calibration. CUDA event enclosures include launch gaps and waits. No kernel-union or exclusive FLOP measurement.')


def main():
    initialize()
    parent.prior.preserve()
    inventory, failures, serving, primitives, memory, graph_rows = [], [], [], [], [], []
    receipts = []
    for path in sorted((ROOT / 'configs').glob('*.json')):
        cfg = json.loads(path.read_text())
        if 'name' not in cfg:
            continue
        name = cfg['name']
        folder = ROOT / 'raw' / name
        receipt_path = folder / 'dispatcher_result.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        row = dict(name=name, model=cfg['model'], count=cfg['count'], phase=cfg.get('phase'),
            complete=receipt.get('complete', False), status=receipt.get('status', 'NOT_RUN'),
            elapsed_job_wall_seconds=receipt.get('elapsed_seconds'),
            configuration_SHA256=digest(path), gpus=cfg['gpus'])
        inventory.append(row)
        if receipt:
            receipts.append(receipt)
        if receipt and not receipt.get('complete'):
            log = (ROOT / 'logs' / (name + '.log')).read_text(errors='replace')
            exceptions = [line.strip() for line in log.splitlines()
                          if line.startswith(('ValueError:', 'TypeError:', 'AttributeError:', 'RuntimeError:', 'AssertionError:'))]
            failures.append(dict(row, exception=exceptions[-1] if exceptions else 'SEE_FULL_LOG'))
        if not receipt.get('complete'):
            continue
        gate = folder / 'serving_gate.json'
        if gate.exists():
            value = json.loads(gate.read_text())
            if name.startswith('e_'):
                from ..looped_pd_atomic_prefill.evidence import compare
                reference = f'c_{cfg["model"]}_native_medium_start1_128'
                comparison = compare(reference, name)
                write(folder / 'matched_current_native_output_comparison.json', comparison)
                assert comparison['passed']
            serving.append(dict(value.get('metrics', {}), name=name, model=cfg['model'], method=cfg['method'],
                load=cfg['load'], count=cfg['count'], start=cfg['independent_start'],
                markers=cfg['observe_service_timing'], passed=value['passed'],
                calibration=bool(cfg.get('calibration_serial')), runtime_control=name.startswith('e_')))
            if cfg.get('calibration_serial'):
                primitives.append(primitive_details(folder, cfg))
        for item in graph_summary(folder):
            graph_rows.append(dict(name=name, model=cfg['model'], **item))
        for item in folder.glob('memory_inventory_initial_*.json'):
            d = json.loads(item.read_text())
            # NVML's recorded physical denominator is 81559 MiB on this host.
            total = 81559 * 1024**2
            values = {key + '_GiB': amount / 1024**3 for key, amount in d.items()
                      if key.endswith('_bytes') and isinstance(amount, (int, float))}
            values.update({key + '_HBM_percent': 100 * amount / total for key, amount in d.items()
                           if key.endswith('_bytes') and isinstance(amount, (int, float))})
            memory.append(dict(name=name, model=cfg['model'], role=d['role'], snapshot='STAGE_INITIAL',
                               physical_HBM_GiB=total / 1024**3, **values))
        for item in folder.glob('native_factory_memory_*.json'):
            d = json.loads(item.read_text())
            total = 81559 * 1024**2
            values = {key + '_GiB': amount / 1024**3 for key, amount in d.items()
                      if key.endswith('_bytes') and isinstance(amount, (int, float))}
            values.update({key + '_HBM_percent': 100 * amount / total for key, amount in d.items()
                           if key.endswith('_bytes') and isinstance(amount, (int, float))})
            memory.append(dict(name=name, model=cfg['model'], role=d['role'], snapshot='POST_CONNECTOR_REGISTRATION',
                               physical_HBM_GiB=total / 1024**3, **values))
    write(ROOT / 'configuration_inventory.json', inventory)
    write(ROOT / 'failures.json', failures)
    write(ROOT / 'serving_results.json', serving)
    write(ROOT / 'primitive_costs.json', primitives)
    csv_write(ROOT / 'configuration_inventory.csv', inventory)
    csv_write(ROOT / 'failure_inventory.csv', failures)
    csv_write(ROOT / 'serving_results.csv', serving)
    csv_write(ROOT / 'memory_gib_percent.csv', memory)
    csv_write(ROOT / 'graph_cache_observations.csv', graph_rows)

    main_rows = [row for row in serving if not row['calibration'] and not row['runtime_control']]
    ratios, marker_gates, envelope = {}, [], []
    for model in ('ouro', 'huginn'):
        rows = [row for row in main_rows if row['model'] == model]
        native = {row['start']: row for row in rows if row['method'] == 'native'
                  and row['load'] == 'medium' and row['markers']}
        methods = sorted({row['method'] for row in rows if row['load'] == 'medium' and row['markers']})
        ratios[model] = {}
        for method in methods:
            chosen = {row['start']: row for row in rows if row['method'] == method
                      and row['load'] == 'medium' and row['markers']}
            ratios[model][method] = {metric: paired_ratio([chosen[start][metric] / native[start][metric]
                for start in sorted(chosen.keys() & native.keys())]) for metric in
                ('TTFT_p95_ms', 'service_TTFT_ms_p95', 'E2E_p95_ms', 'requests_per_second', 'joint_SLO_goodput')}
        on = {row['method']: row for row in rows if row['load'] == 'medium' and row['start'] == 1 and row['markers']}
        off = {row['method']: row for row in rows if not row['markers']}
        for method, b in off.items():
            a = on[method]
            change = 100 * (a['response_span_s'] / b['response_span_s'] - 1)
            marker_gates.append(dict(model=model, method=method, marker_on=a['name'], marker_off=b['name'],
                wall_change_percent=change, passed=abs(change) <= 5,
                TTFT_p95_change_percent=100 * (a['TTFT_p95_ms'] / b['TTFT_p95_ms'] - 1),
                scope='One matched global-wall screen. It does not bound request-tail perturbation or other loads/starts.'))
        for load in ('medium', 'heavy'):
            baseline = next(row for row in rows if row['method'] == 'native' and row['load'] == load
                            and row['start'] == 1 and row['markers'])
            for family, predicate in (('native', lambda m: m == 'native'),
                ('token', lambda m: m == 'native' or m.startswith('token')),
                ('loop', lambda m: m == 'native' or m.startswith('loop')),
                ('2d', lambda m: m == 'native' or m.startswith(('loop', 'token', '2d_')))):
                candidates = [row for row in rows if row['load'] == load and row['start'] == 1
                              and row['markers'] and row['passed'] and row['Decode_SLO_screen'] and predicate(row['method'])]
                best = min(candidates, key=lambda row: row['TTFT_p95_ms'])
                envelope.append(dict(model=model, load=load, family=family, selected=best['method'],
                    TTFT_p95_ms=best['TTFT_p95_ms'], service_TTFT_p95_ms=best['service_TTFT_ms_p95'],
                    goodput=best['joint_SLO_goodput'], TTFT_reduction_percent=100 * (1 - best['TTFT_p95_ms'] / baseline['TTFT_p95_ms']),
                    scope='Measured fixed-action envelope on executed, Decode-screen-passing points. Not a dynamic oracle or a bound over unmeasured Token cuts.',
                    token_partial_qualified=model == 'ouro'))
    write(ROOT / 'paired_start_ratios.json', ratios)
    write(ROOT / 'marker_interference.json', marker_gates)
    write(ROOT / 'measured_fixed_action_envelope.json', envelope)
    csv_write(ROOT / 'marker_interference.csv', marker_gates)
    csv_write(ROOT / 'measured_fixed_action_envelope.csv', envelope)
    structural = []
    for model, t, r in (('ouro', 4096, 4), ('huginn', 2048, 32)):
        cuts = range(r + 1)
        for c in cuts:
            for k in sorted({0, t // 4, t // 2, 3 * t // 4, t}):
                structural.append(dict(model=model, **Frontier(t, r, c, k).accounting(model)))
    write(ROOT / 'structural_frontiers_not_gpu_costs.json', structural)
    csv_write(ROOT / 'structural_frontiers_not_gpu_costs.csv', structural)
    kst = ZoneInfo('Asia/Seoul')
    first = min(row['time'] - row.get('elapsed_seconds', 0) for row in receipts)
    last = max(row['time'] for row in receipts)
    summary = dict(configurations=len(inventory), dispatches=len(receipts),
        completed_dispatches=sum(row['complete'] for row in receipts), failed_dispatches=len(failures),
        completed_count_distribution=dict(Counter(row['count'] for row in inventory if row['complete'])),
        completed_offered_requests=sum(row['count'] for row in inventory if row['complete']),
        main_serving_runs=len(main_rows), main_serving_requests=sum(row['count'] for row in main_rows),
        completed_N128_runs=sum(row['complete'] and row['count'] == 128 for row in inventory),
        completed_N128_requests=sum(row['count'] for row in inventory if row['complete'] and row['count'] == 128),
        GPU_first_dispatch_start=datetime.fromtimestamp(first, kst).isoformat(),
        GPU_last_dispatch_end=datetime.fromtimestamp(last, kst).isoformat(),
        GPU_phase_hours_from_campaign_start=(last - START) / 3600,
        summed_job_wall_hours=sum(row.get('elapsed_seconds', 0) for row in receipts) / 3600,
        analysis_generated=datetime.now(kst).isoformat(), closeout_restart='2026-11-05T09:42:31+09:00',
        original_overall_deadline=datetime.fromtimestamp(DEADLINE, kst).isoformat(),
        budget_scope='GPU jobs finished inside the GPU budget. Assistant capacity interruption delayed closeout beyond the original six-hour overall deadline. No post-deadline GPU experiment is launched.',
        dynamic_oracle='NOT_IMPLEMENTED_OR_VALIDATED', optimistic_compute_upper_bound='NOT_ESTABLISHED',
        kernel_union_SM_FLOPs='NOT_MEASURED_IN_THIS_CAMPAIGN')
    write(ROOT / 'analysis_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
