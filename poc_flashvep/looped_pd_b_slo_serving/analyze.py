"""Arrival-based SLO accounting. Unknown values never become measured zeros."""
import csv
import json
from collections import Counter, defaultdict
import numpy as np

from .common import ROOT, write
from .jobs import output_rows


def save(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n', restval='NOT_MEASURED')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def wilson(correct, total):
    if not total:
        return [None, None]
    p = correct / total
    z = 1.959963984540054
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [100 * (center - half), 100 * (center + half)]


def summarize(job):
    folder = ROOT / 'raw' / job['name']
    status_file = folder / 'status.json'
    state = json.loads(status_file.read_text()) if status_file.exists() else {'status': 'NOT_RUN'}
    rows = output_rows(job['name'])
    base = dict(run=job['name'], model=job['model'], arm=job['arm'], workload=job['workload'],
                execution_epoch=job.get('execution_epoch', 'original'),
                runtime_override=job.get('env', {}).get('B_REPAIR_VARIANT', 'NONE'),
                dataset=job.get('dataset', job['workload']),
                GPU_count=len(job['gpus']), physical_GPUs=job['gpus'], P=len(job['gpus']) // 2 if job['arm'] not in ('DP', 'DP_Mixed', 'Native', 'HF') else 0,
                D=len(job['gpus']) // 2 if job['arm'] in ('PD', 'PD_G1', 'B', 'GenericLayer') else 0,
                colocated_replicas=len(job['gpus']) if job['arm'] in ('DP', 'DP_Mixed') else 0,
                seed=job['seed'], offered_rate=job['rate'], planned_arrivals=job['count'], completed=len(rows),
                status=state['status'], error=state.get('error', ''), output_cap=job['cap'],
                diagnostic=job.get('diagnostic', False), fixed_depth=4 if job['model'] == 'ouro' else 32,
                source=('ACTUAL_OFFICIAL_HF_GPU_EXECUTION' if job.get('reference') else 'ACTUAL_RLT_GPU_EXECUTION') if rows else 'NOT_MEASURED',
                timing_class='HEAVY_PROFILE_NOT_CLEAN_SERVING' if job.get('python', '').endswith('profile_python.py') else 'DIAGNOSTIC_NOT_CLEAN_SERVING' if job.get('diagnostic') else 'ISOLATED_OPEN_LOOP_SERVING',
                accepted_lossless_speedup=False)
    percentiles = dict(base)
    for key in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'input_tokens', 'output_tokens', 'delivery_delay_ms'):
        values = [row[key] for row in rows if key in row]
        percentiles[key + '_mean'] = float(np.mean(values)) if values else 'NOT_MEASURED'
        for q in (50, 95, 99):
            percentiles[key + f'_p{q}'] = float(np.percentile(values, q)) if values else 'NOT_MEASURED'
    gaps = [1000 * (event['time'] - event['previous_time']) for row in rows for event in row.get('token_events', []) if not event['first']]
    for q in (50, 95, 99):
        percentiles[f'observed_ITL_ms_p{q}'] = float(np.percentile(gaps, q)) if gaps else 'NOT_MEASURED'
    percentiles['observed_ITL_ms_max'] = max(gaps) if gaps else 'NOT_MEASURED'
    percentiles['coalesced_callbacks'] = sum(event['new_tokens'] > 1 for row in rows for event in row.get('token_events', []))
    percentiles['ITL_definition'] = 'Observed emission gaps. No invented timestamps inside multi-token callbacks.'
    percentiles['p99_scope'] = 'EXPLORATORY_N_LT_100' if len(rows) < 100 else 'One run, not independent-run uncertainty'
    percentiles['percentile_population'] = 'Completed requests only. Missing, timed-out, or aborted requests are not assigned artificial finite latency.'
    if state['status'] != 'COMPLETE':
        percentiles['p99_scope'] = 'FAILED_OR_INCOMPLETE_RUN_SURVIVORS_ONLY'
    # The pinned RLT scheduler calls its EOS termination FinishReason.STOP.
    # It does not configure string stops for these experiments.
    reasons_available = bool(rows) and all('finish_reason' in row for row in rows)
    base.update(EOS_stops=sum(row.get('finish_reason') in ('eos', 'stop') for row in rows) if reasons_available else 'NOT_MEASURED',
                cap_stops=sum(row.get('finish_reason') == 'length' for row in rows) if reasons_available else 'NOT_MEASURED',
                finish_reason_counts=dict(Counter(row.get('finish_reason') for row in rows)),
                EOS_definition='Pinned RLT STOP is EOS termination. LENGTH is the safety cap. Raw reasons are preserved.')
    window_file = folder / 'window.json'
    window = None
    if window_file.exists():
        window = json.loads(window_file.read_text())
    elif job.get('rate', 0) > 0 and (folder / 'arrival_trace.json').exists():
        # Preserve the planned-arrival denominator when the engine fails before drain.
        # Never invent a successful drain or a failure-end GPU timestamp.
        trace = json.loads((folder / 'arrival_trace.json').read_text())
        start = rows[0]['arrival'] - trace['offsets'][rows[0]['sample_id']] if rows else None
        window = dict(start=start, window_seconds=trace['window_seconds'],
                      evaluation_end=start + trace['window_seconds'] if start is not None else None,
                      end=max((row['finish'] for row in rows), default=None),
                      actual_trace_rate=job['count'] / trace['window_seconds'] if trace['window_seconds'] else None)
        base['partial_window_scope'] = 'Failed or incomplete trace. Planned arrival window is known. Drain and cohort-end time are NOT_MEASURED.'
    if window is not None:
        duration = window['window_seconds']
        span = max(duration, window['actual_span_seconds']) if 'actual_span_seconds' in window else None
        base.update(window_seconds=duration, drain_seconds=window.get('drain_seconds', 'NOT_MEASURED'),
                    cohort_seconds=span if span is not None else 'NOT_MEASURED', actual_trace_rate=window.get('actual_trace_rate'),
                    completed_throughput_with_drain=len(rows) / span if span is not None and span > 0 else 'NOT_MEASURED',
                    output_tokens_s_with_drain=sum(row.get('output_tokens', 0) for row in rows) / span if span is not None and span > 0 else 'NOT_MEASURED')
        base['completed_inside_window_requests_s'] = sum(row['finish'] <= window['evaluation_end'] for row in rows) / duration if duration > 0 and window['evaluation_end'] is not None else 'NOT_MEASURED'
        observation_file = ROOT / 'logs/all_GPU_process_observations.jsonl'
        if observation_file.exists() and window.get('start') is not None and window.get('end') is not None:
            observed = [json.loads(line) for line in observation_file.read_text().splitlines() if line.strip()]
            observed = [row for row in observed if window['start'] <= row['monotonic'] <= window['end']]
            base['all_GPU_process_samples'] = len(observed)
            base['unregistered_GPU_process_samples'] = sum(any(not p['registered_experiment_session'] for p in row['processes']) for row in observed)
            base['isolation_evidence'] = 'READ_ONLY_10S_PROCESS_SAMPLES_NOT_CPU_EXCLUSIVITY' if observed else 'NO_FULL_FABRIC_OBSERVER_COVERAGE'
        lock = ROOT / 'configs/slo_lock.json'
        if lock.exists() and duration > 0:
            slo = json.loads(lock.read_text())[job['model']]
            success = []
            strict = []
            for row in rows:
                normal = row['finish_reason'] in ('eos', 'length', 'stop') and row['output_tokens'] > 0
                success.append(normal and row['ttft_ms'] <= slo['TTFT_ms'] and row['tpot_ms'] <= slo['TPOT_ms'])
                maximum_gap = max([1000 * (event['time'] - event['previous_time']) for event in row['token_events'] if not event['first']], default=0.)
                strict.append(normal and row['ttft_ms'] <= slo['TTFT_ms'] and maximum_gap <= slo['secondary_ITL_ms'])
            n = job['count']
            base.update(TTFT_limit_ms=slo['TTFT_ms'], TPOT_limit_ms=slo['TPOT_ms'],
                        qualified_requests=sum(success), joint_attainment_percent=100 * sum(success) / n,
                        joint_Wilson_95_percent=wilson(sum(success), n),
                        Wilson_scope='Descriptive binomial interval. Does not account for queue dependence or independent-start variation.',
                        TTFT_attainment_percent=100 * sum(row['ttft_ms'] <= slo['TTFT_ms'] for row in rows) / n,
                        TPOT_attainment_percent=100 * sum(row['tpot_ms'] <= slo['TPOT_ms'] for row in rows) / n,
                        qualified_requests_s_window=sum(success) / duration,
                        qualified_requests_s_with_drain=sum(success) / span if span is not None and span > 0 else 'NOT_MEASURED',
                        planned_but_incomplete_fraction=max(0, n - len(rows)) / n)
            base.update(secondary_strict_streaming_attainment_percent=100 * sum(strict) / n,
                        secondary_ITL_limit_ms=slo['secondary_ITL_ms'],
                        secondary_scope='TTFT and every observed emission gap. Does not select the primary winner.')
        queue = folder / 'queue.jsonl'
        samples = [json.loads(line) for line in queue.read_text().splitlines()] if queue.exists() else []
        samples = [row for row in samples if not row['warmup'] and duration * .25 <= row['elapsed'] <= duration]
        if len(samples) >= 8 and duration > 0:
            x = np.asarray([row['elapsed'] for row in samples])
            y = np.asarray([row['outstanding'] for row in samples])
            slope = float(np.polyfit(x, y, 1)[0])
            early = float(y[x <= duration * .5].mean()) if any(x <= duration * .5) else float(y[0])
            late = float(y[x >= duration * .75].mean()) if any(x >= duration * .75) else float(y[-1])
            check = slope <= .05 * job['rate'] and late - early <= max(3., .1 * job['rate'] * duration / 4)
            base.update(backlog_slope_requests_s=slope, backlog_early_mean=early, backlog_late_mean=late,
                        weak_queue_check=check,
                        queue_stability='NO_PERSISTENT_GROWTH_OBSERVED' if check and duration >= 300 else 'INSUFFICIENT_DURATION' if check else 'PERSISTENT_GROWTH_SUSPECTED',
                        queue_scope='Outstanding requests include execution. Heuristic is preregistered, not a proof of stationarity.')
    transfer = dict(base)
    after = folder / 'after_measured.json'
    before = folder / 'before_measured.json'
    if after.exists() and before.exists():
        a, b = json.loads(before.read_text()), json.loads(after.read_text())
        if all(isinstance(value, dict) for value in b.values()):
            for field in ('weight_bytes', 'derived_weight_pack_bytes', 'reserved_kv_bytes', 'peak_allocated_bytes'):
                transfer[field + '_by_worker'] = {key: value.get(field, 'NOT_MEASURED') for key, value in b.items()}
        for field in ('transport_bytes_sent', 'transport_completed_count', 'transport_completion_observation_seconds', 'issued_loop_rows'):
            if all(isinstance(value, dict) for value in b.values()):
                transfer[field] = sum(value.get(field, 0) - a.get(key, {}).get(field, 0) for key, value in b.items())
        if 'issued_loop_rows' in transfer and state['status'] == 'COMPLETE':
            expected = base['fixed_depth'] * sum(row.get('input_tokens', 0) + row.get('output_tokens', 0) - 1 for row in rows)
            transfer['structural_full_depth_loop_rows'] = expected
            transfer['issued_loop_rows_equal_structural_full_depth'] = transfer['issued_loop_rows'] == expected
            transfer['loop_row_scope'] = 'Observed recurrent-row submissions versus R*(prompt tokens + generated tokens - request count). Not FLOPs or latency.'
        if job['arm'] in ('PD', 'PD_G1', 'B') and state['status'] == 'COMPLETE' and 'transport_bytes_sent' in transfer:
            hidden_row = 4096 if job['model'] == 'ouro' else 21120
            transfer['structural_final_hidden_payload_bytes'] = len(rows) * hidden_row
            transfer['KV_payload_bytes_by_counter_minus_hidden'] = transfer['transport_bytes_sent'] - len(rows) * hidden_row
            transfer['payload_scope'] = 'Measured transport counter includes valid prompt KV and final hidden rows. KV-only value subtracts source-derived BF16 hidden-row sizes. Out-of-band control bytes are not measured.'
    peaks = list(folder.glob('NVML-summary-*.json'))
    if peaks:
        nvml = json.loads(peaks[-1].read_text())
        transfer['sampled_peak_HBM_bytes_by_GPU'] = nvml.get('measured_cohort_peak_sampled_HBM_bytes', {})
    closed = folder / 'closed_stats.json'
    if closed.exists():
        metrics = json.loads(closed.read_text()).get('metrics', [])
        exposed = []
        negative_handoff = []
        partitions = []
        request_map = {f'r0_q{row["sample_id"]}': row for row in rows}
        for row in metrics:
            t = row.get('timings', {})
            if not row['request_id'].startswith('r-1_') and 'd_received' in t and 'prefill_compute_completed' in t:
                value = 1000 * (t['d_received'] - t['prefill_compute_completed'])
                (exposed if value >= 0 else negative_handoff).append(value)
            keys = ('admitted', 'prefill_started', 'prefill_compute_completed', 'd_received', 'first_token', 'finished')
            actual = request_map.get(row['request_id'])
            if actual and all(key in t for key in keys):
                values = [t['admitted'], t['prefill_started'] - t['admitted'],
                          t['prefill_compute_completed'] - t['prefill_started'],
                          t['d_received'] - t['prefill_compute_completed'],
                          t['first_token'] - t['d_received'], t['finished'] - t['first_token'],
                          actual['e2e_ms'] / 1000 - t['finished']]
                if min(values) >= 0:
                    partitions.append(values)
        transfer['exposed_handoff_ms_mean'] = float(np.mean(exposed)) if exposed else 'NOT_MEASURED'
        transfer['exposed_handoff_ms_p95'] = float(np.percentile(exposed, 95)) if exposed else 'NOT_MEASURED'
        transfer['exposed_handoff_nonnegative_N'] = len(exposed)
        transfer['negative_handoff_notification_differences_N'] = len(negative_handoff)
        transfer['handoff_scope'] = 'Nonnegative host-notification differences only. Negative ordering differences remain raw evidence, not negative physical stall or speedup.'
        if partitions:
            totals = np.sum(np.asarray(partitions), axis=0)
            names = ('coordinator_queue', 'reservation_control', 'P_prefill_including_backpressure',
                     'uncovered_handoff', 'D_resume_first_token', 'remaining_decode', 'harness_delivery_residual')
            transfer['host_E2E_component_percent'] = dict(zip(names, (100 * totals / totals.sum()).tolist()))
            transfer['host_partition_N'] = len(partitions)
            transfer['host_partition_E2E_sum_seconds'] = float(totals.sum())
            transfer['host_partition_denominator'] = 'Sum of valid request E2E wall times. Not GPU utilization or cohort makespan.'
    transfer['transfer_time_scope'] = 'Host-observed cumulative transfer durations can overlap. They are not additive E2E or DMA time.'
    transfer['pure_GPU_attention_FFN_E2E_percent'] = 'NOT_MEASURED'
    return base, percentiles, transfer


def main():
    capacity, latency, transfers = [], [], []
    for path in sorted((ROOT / 'configs/jobs').glob('*.json')):
        job = json.loads(path.read_text())
        if not (ROOT / 'raw' / job['name'] / 'status.json').exists():
            continue
        a, b, c = summarize(job)
        capacity.append(a)
        latency.append(b)
        transfers.append(c)
    primary = [row for row in capacity if row['workload'] == 'W1' and not row['diagnostic']]
    save('03_primary_w1_capacity.csv', [row for row in primary if row['execution_epoch'] == 'original'])
    if any(row['execution_epoch'] != 'original' for row in primary):
        save('repair_epoch_w1_capacity.csv', [row for row in primary if row['execution_epoch'] != 'original'])
    save('04_latency_percentiles.csv', latency)
    save('05_workload_w2_w3_w4.csv', [row for row in capacity if row['workload'] in ('W2', 'W3', 'W4') and not row['diagnostic']])
    save('06_transfer_and_critical_path.csv', transfers)
    components = []
    for row in transfers:
        shared = {key: row[key] for key in ('run', 'model', 'arm', 'GPU_count', 'physical_GPUs', 'workload', 'status')}
        values = row.get('host_E2E_component_percent', {})
        if values:
            for component, percent in values.items():
                components.append(dict(shared, component=component, percent=percent,
                                       request_count=row['host_partition_N'],
                                       denominator_seconds=row['host_partition_E2E_sum_seconds'],
                                       evidence='MEASURED_HOST_REQUEST_WALL_TIME_PARTITION',
                                       limitation='Not a GPU-kernel, DMA, dispatch, or utilization breakdown. Not cohort makespan.'))
        else:
            components.append(dict(shared, component='host_request_phase_partition', percent='NOT_MEASURED',
                                   request_count=0, evidence='NOT_MEASURED'))
        components.append(dict(shared, component='GPU_attention_FFN_dispatch_idle_partition', percent='NOT_MEASURED',
                               evidence='NOT_MEASURED', limitation='No pure GPU critical-path attribution was collected.'))
    save('e2e_component_percent.csv', components)
    save('07_scale_2_4_8gpu.csv', [row for row in capacity if row['workload'] == 'W1' and not row['diagnostic']])
    from .jobs import parity
    paired = []
    baseline_pairs = []
    groups = defaultdict(dict)
    for path in sorted((ROOT / 'configs/jobs').glob('*.json')):
        job = json.loads(path.read_text())
        if job.get('diagnostic') or job.get('rate', 0) <= 0:
            continue
        key = (job['model'], tuple(job['gpus']), job['workload'], job['count'], job['rate'], job['seed'], job['dataset'],
               job.get('execution_epoch', 'original'))
        groups[key][job['arm']] = job
    for key, group in groups.items():
        if 'PD_G1' in group:
            for arm in ('DP', 'DP_Mixed', 'PD'):
                if arm not in group:
                    continue
                value = parity(group['PD_G1']['name'], group[arm]['name'])
                value.update(model=key[0], GPU_count=len(key[1]), workload=key[2],
                             rate=key[4], seed=key[5], control='PD_G1', comparison=arm, execution_epoch=key[7])
                for side, job in (('left', group['PD_G1']), ('right', group[arm])):
                    outputs = output_rows(job['name'])
                    value[side + '_generated_tokens_total'] = sum(row['output_tokens'] for row in outputs)
                    value[side + '_recorded_decode_loop_work'] = sum(sum(row['exit_depths']) for row in outputs)
                baseline_pairs.append(value)
        if 'B' not in group:
            continue
        for arm in ('DP', 'DP_Mixed', 'PD', 'PD_G1'):
            if arm not in group:
                continue
            value = parity(group[arm]['name'], group['B']['name'])
            value.update(model=key[0], GPU_count=len(key[1]), physical_GPUs=list(key[1]),
                         workload=key[2], planned_arrivals=key[3], rate=key[4], seed=key[5],
                         baseline=arm, candidate='B', execution_epoch=key[7])
            paired.append(value)
    save('paired_output_fidelity.csv', paired)
    save('baseline_output_fidelity.csv', baseline_pairs)
    write(ROOT / 'state/current_summary.json', dict(runs=len(capacity), statuses=dict(__import__('collections').Counter(row['status'] for row in capacity))))


if __name__ == '__main__':
    main()
