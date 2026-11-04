"""Derive per-request queue and service latency before computing quantiles."""
from .common import *


def individual_slo_attainment(rows, offered, slo):
    if offered <= 0 or len(rows) > offered:
        raise ValueError('Invalid offered population')
    ttft = sum(row['ttft_ms'] <= slo['TTFT_ms'] for row in rows)
    tpot = sum(row['tpot_ms'] <= slo['TPOT_ms'] for row in rows)
    return dict(TTFT_SLO_pass_count=ttft, TPOT_SLO_pass_count=tpot,
        TTFT_SLO_attainment=ttft / offered, TPOT_SLO_attainment=tpot / offered,
        SLO_attainment_denominator=offered)


def disjoint_latency_shares(records, outputs):
    e2e = sum(1000 * (row['finish'] - row['arrival']) for row in outputs)
    queue = sum(row['queue_ms'] for row in records)
    service = sum(row['service_TTFT_ms'] for row in records)
    decode = e2e - queue - service
    if e2e <= 0 or min(queue, service, decode) < -1e-5:
        raise ValueError('Invalid disjoint request-latency components')
    return dict(summed_request_E2E_ms=e2e,
        summed_request_queue_E2E_percent=100 * queue / e2e,
        summed_request_service_TTFT_E2E_percent=100 * service / e2e,
        summed_request_postfirst_Decode_E2E_percent=100 * decode / e2e,
        latency_share_scope='Disjoint host request intervals. Denominator is summed request E2E latency, '
                            'not wall time, GPU time, or a sum of component percentiles.')


def decompose(arrival, first_token, start):
    if not arrival <= start['time'] <= first_token:
        raise ValueError('Invalid actual-execution timestamp order')
    return dict(queue_ms=1000 * (start['time'] - arrival),
        service_TTFT_ms=1000 * (first_token - start['time']),
        TTFT_ms=1000 * (first_token - arrival),
        start_clock_uncertainty_ms=1000 * (start['upper'] - start['lower']) / 2)


def measured_device_rows(rows, expected, measurement_begin):
    result = {}
    for row in rows:
        # Initial startup requests deliberately use the same generation seeds.
        # Their distinct request IDs and pre-measurement submissions are not
        # duplicate measured requests. Keep the raw startup records intact.
        if row['seed'] not in expected or row['first_host_submit'] < measurement_begin:
            continue
        assert row['seed'] not in result, 'Duplicate measured device state for one role and seed'
        result[row['seed']] = row
    return result


def summarize(name):
    from ..fluidtoken_loop_split.static_screen import metrics as host_metrics
    from ..looped_pd_headroom_recovery.gates import percentile
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    rows = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    rows = [row for row in rows if not row.get('warmup')]
    expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
    assert {row['seed'] for row in rows} == expected, 'Include all offered requests or report the run as incomplete'
    devices = {'prefill': {}, 'decode': {}}
    measurement_begin = json.loads((folder / 'measurement_begin.json').read_text())['time']
    for path in folder.glob('service_device_times_*.json'):
        value = json.loads(path.read_text())
        target = devices[value['role']]
        selected = measured_device_rows(value['requests'], expected, measurement_begin)
        assert not (target.keys() & selected.keys()), 'Duplicate measured role across worker records'
        target.update(selected)
    if set(devices['prefill']) != expected:
        raise ValueError('Actual first Prefill execution is not measured for every request')
    phases = {row['request_id']: row for row in json.loads((folder / 'closed_stats.json').read_text())['metrics']}
    records = []
    for row in rows:
        seed = row['seed']
        rid = f"r{row['run']}_q{row['sample_id']}"
        first = next(event['time'] for event in row['token_events'] if event['first'])
        p = devices['prefill'][seed]
        values = decompose(row['arrival'], first, p['first_model_stream_entry'])
        # Native coordinator timestamps are relative to its request creation.
        # They are observed host boundaries, not kernel or DMA durations.
        phase = phases[rid]
        timings = phase['timings']
        observed_handoff = None
        if 'prefill_compute_completed' in timings and 'd_received' in timings:
            observed_handoff = 1000 * (timings['d_received'] - timings['prefill_compute_completed'])
        d = devices['decode'].get(seed)
        record = dict(request_id=rid, seed=seed, sample_id=row['sample_id'], arrival=row['arrival'],
            first_token=first, actual_P_model_stream_entry=p['first_model_stream_entry'],
            **values, P_prompt_device_enclosure_ms=p['prompt_device_enclosure_ms'],
            observed_handoff_ms=observed_handoff,
            D_prompt_device_enclosure_ms=d['prompt_device_enclosure_ms'] if d else None,
            D_continuation_to_first_token_ms=1000 * (first - d['first_model_stream_entry']['time']) if d else None)
        records.append(record)
    result = host_metrics(name)
    result.update(individual_slo_attainment(rows, cfg['count'], result['frozen_SLO']))
    result.update(disjoint_latency_shares(records, rows))
    result['input_tokens_per_second'] = result['total_tokens_per_second'] - result['output_tokens_per_second']
    coalesced = [event for row in rows for event in row['token_events']
                 if not event['first'] and event['new_tokens'] > 1]
    captures = [json.loads(line) for path in folder.glob('graph_captures_*.jsonl')
                for line in path.read_text().splitlines() if json.loads(line).get('measured_phase')]
    offered = [json.loads(line) for line in (folder / 'offered_requests.jsonl').read_text().splitlines()]
    assert len(offered) == cfg['count'] and {row['seed'] for row in offered} == expected
    scheduled_rate = (len(rows) - 1) / (max(r['arrival'] for r in rows) - min(r['arrival'] for r in rows))
    submitted_rate = (len(offered) - 1) / (max(r['actual_send_start'] for r in offered)
                                         - min(r['actual_send_start'] for r in offered))
    result.update(name=name, model=cfg['model'], arm=cfg['arm'],
        nominal_QPS=cfg['arrival_rate'], arrival_seed=cfg['arrival_seed'],
        realized_arrival_QPS=scheduled_rate, scheduled_trace_QPS=scheduled_rate,
        actual_submit_QPS=submitted_rate,
        arrival_rate_scope='The retained realized_arrival_QPS field measures the finite scheduled trace. actual_submit_QPS measures actual coordinator send starts. Neither is completed throughput.',
        service_start_evidence='Native model-stream entry CUDA event after input preparation',
        service_start_limit='Not a CUPTI first-kernel timestamp. Clock uncertainty is retained per request.',
        timing_scope='Device enclosures include launch gaps and possibly other requests. They are not exclusive GPU kernel times.',
        coalesced_nonfirst_delivery_events=len(coalesced),
        token_tail_scope=('One-token coordinator deliveries.' if not coalesced else
            'Coalesced deliveries exist. The retained contract uses event-average intervals. Exact individual generation-time tails are not measured.'),
        measured_graph_capture_count=len(captures),
        measured_graph_capture_host_seconds=sum(row['host_seconds'] for row in captures),
        profiler_overhead='NOT_MEASURED' if not cfg.get('service_timing_overhead_qualified') else cfg['service_timing_overhead_qualified'])
    for field in ('queue_ms', 'service_TTFT_ms', 'P_prompt_device_enclosure_ms',
                  'observed_handoff_ms', 'D_prompt_device_enclosure_ms', 'D_continuation_to_first_token_ms'):
        values = [row[field] for row in records if row[field] is not None]
        for q in (.5, .95, .99):
            result[f'{field}_p{int(q * 100)}'] = percentile(values, q) if values else None
    result['maximum_clock_uncertainty_ms'] = max(row['start_clock_uncertainty_ms'] for row in records)
    lag = [1000 * (row['actual_send_start'] - row['arrival']) for row in offered]
    for q in (.5, .95, .99):
        result[f'actual_submit_lateness_p{int(q * 100)}_ms'] = percentile(lag, q)
    result['maximum_actual_submit_lateness_ms'] = max(lag)
    write_derived(folder / 'service_latency_requests.json', records)
    write_derived(folder / 'serving_metrics.json', result)
    return result
