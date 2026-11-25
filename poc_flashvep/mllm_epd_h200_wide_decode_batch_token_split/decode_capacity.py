"""Measure D-only decode with identical prepared-NIXL cohorts at each cap."""

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.request import urlopen

from ..mllm_epd_h200_prefill_heavy_batch_matched import capacity as base
from ..mllm_epd_h200_prefill_heavy_batch_matched.trace import rpc


CAPS = (1, 2, 4, 8, 16, 32, 64, None)
COHORT_ORDER_SEED = 2026112404
STAGE_KV_LEASE_SECONDS = 180
MIN_INITIAL_LEASE_REMAINING = 10
TOTAL_KV_BYTES = 42 * base.GIB
KV_COHORT_LIMIT = 38 * base.GIB
OUTPUT_TOKEN_CONTRACT = 5269
MAX_MODEL_LENGTH = 32768
LOGICAL_MAXIMA = ('scheduled_requests', 'scheduler_generation_phase_requests', 'prepared_decode_requests')
PORTS = {'encoder': 18840, 'prefill': 18841, 'decode': 18842}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def cap_label(cap):
    return 'free' if cap is None else str(cap)


def normalized_caps(caps):
    values = tuple(None if cap == 'free' else int(cap) if cap is not None else None for cap in caps)
    if not values or len(set(values)) != len(values) or any(cap not in CAPS for cap in values):
        raise ValueError('Caps must be a unique subset of 1/2/4/8/16/32/64/free')
    return values


def kv_blocks(token_count):
    return (token_count + base.KV_BLOCK_TOKENS - 1) // base.KV_BLOCK_TOKENS


def planned_cohorts(rows, replay):
    """Pack the input order shared by every Decode cap under real block bounds."""
    row_ids = {row['request_id'] for row in rows}
    if len(rows) != 100 or set(replay) != row_ids or len(row_ids) != len(rows):
        raise ValueError('D-only sweep requires all 100 unique frozen requests')
    if any(type(count) is not int or count < 1 for count in replay.values()) or sum(replay.values()) != OUTPUT_TOKEN_CONTRACT:
        raise ValueError('D-only sweep requires the 5,269 frozen output tokens')
    cohorts, plans = [], []
    p_blocks = d_blocks = embedding_bytes = 0
    pending = []
    block_bytes = base.KV_BYTES_PER_TOKEN * base.KV_BLOCK_TOKENS

    def freeze():
        cohorts.append(list(pending))
        plans.append({'requests': len(pending), 'request_ids': [row['request_id'] for row in pending],
                      'input_tokens': sum(row['total_input_tokens'] for row in pending),
                      'output_tokens': sum(replay[row['request_id']] for row in pending),
                      'planned_P_export_blocks': p_blocks, 'planned_D_peak_blocks': d_blocks,
                      'planned_D_peak_bytes': d_blocks * block_bytes,
                      'planned_E_embedding_bytes': embedding_bytes})

    for row in rows:
        count = row['total_input_tokens']
        out = replay[row['request_id']]
        vision = row['visual_tokens'] * base.VISION_BYTES_PER_TOKEN
        prefill_blocks, decode_blocks = kv_blocks(count), kv_blocks(count + out)
        if count + out > MAX_MODEL_LENGTH or decode_blocks * block_bytes > KV_COHORT_LIMIT or vision > base.E_BUFFER_BYTES:
            raise ValueError(f'Request exceeds D-only admission policy: {row["request_id"]}')
        if pending and ((d_blocks + decode_blocks) * block_bytes > KV_COHORT_LIMIT
                        or embedding_bytes + vision > base.E_BUFFER_BYTES):
            freeze()
            pending = []
            p_blocks = d_blocks = embedding_bytes = 0
        pending.append(row)
        p_blocks += prefill_blocks
        d_blocks += decode_blocks
        embedding_bytes += vision
    if pending:
        freeze()
    return cohorts, plans


def verify_exported_blocks(exports, plan):
    """Verify full nonoverlapping NIXL P reservations before the drain."""
    if [export['request_id'] for export in exports] != plan['request_ids']:
        raise RuntimeError('NIXL pre-stage ids did not match the frozen ordered cohort')
    reserved = set()
    for export in exports:
        groups = export['kv_transfer_params']['remote_block_ids']
        if len(groups) != 1 or len(groups[0]) != kv_blocks(export['input_tokens']):
            raise RuntimeError(f'Unexpected P KV block geometry for {export["request_id"]}')
        for block in groups[0]:
            if not isinstance(block, int) or block <= 0 or block in reserved:
                raise RuntimeError('An outstanding P reservation overlapped or used the sentinel block')
            reserved.add(block)
    if len(reserved) != plan['planned_P_export_blocks']:
        raise RuntimeError('Exported P block count does not match the cohort plan')
    return len(reserved)


async def fifo_map(count, cap, send):
    """Keep a backlog, admitting the next input as soon as one slot finishes."""
    pool = min(cap, count) if cap is not None else count
    indices = iter(range(count))
    rows = [None] * count
    in_flight = 0
    max_active = 0
    failed = False

    async def worker():
        nonlocal in_flight, max_active, failed
        while not failed:
            index = next(indices, None)
            if index is None:
                return
            in_flight += 1
            max_active = max(max_active, in_flight)
            try:
                rows[index] = await send(index, in_flight)
            except BaseException:
                failed = True
                raise
            finally:
                in_flight -= 1

    await base.collect_all(*(worker() for _ in range(pool)))
    require(all(row is not None for row in rows), 'D-only FIFO did not drain the full cohort')
    return rows, max_active


async def drain_decode(session, port, cohort, names, exports, replay, client, cap, lease_seconds):
    ready = time.monotonic()
    url = f'http://127.0.0.1:{port}'

    async def send(index, current):
        row, name, export = cohort[index], names[index], exports[index]
        started = time.monotonic()
        lower_bound = lease_seconds - (started - export['submit'])
        require(lower_bound > MIN_INITIAL_LEASE_REMAINING,
                f'NIXL P reservation has less than {MIN_INITIAL_LEASE_REMAINING}s conservative TTL before D admission')
        output_count = replay[row['request_id']]
        payload = client.payload(row, name, output_count)
        payload['kv_transfer_params'] = export['kv_transfer_params']
        result = await client.one(base.DecodeSession(session, name), url, row, payload, started,
                                  name, index, evidence_sink=base.MemorySink())
        require(result['status'] == 'MEASURED_ONLINE' and result['output_tokens'] == output_count,
                f'D-only request failed exact output replay: {result.get("error", row["request_id"])}')
        age = result['token_times'][0] - export['submit']
        require(age < lease_seconds, 'First D output lacked a conservative P export lifetime witness')
        result.update(status='ISOLATED_GPU_MEASUREMENT', runtime_request_id=name, submit=result['dispatch'],
                      input_tokens=row['total_input_tokens'], D_only_batch_cap=cap_label(cap),
                      cohort_ready=ready, cohort_queue_wait_ms=(started - ready) * 1000,
                      D_HTTP_inflight_at_admission=current, conservative_P_lease_at_admission=lower_bound,
                      conservative_P_age_at_first_D_token=age)
        return result

    return await fifo_map(len(cohort), cap, send)


def geometry_summary(receipt, cap):
    require(receipt['valid'] and receipt['prepared_calls'] > 0, f'D prepared/scheduled counter invalid: {receipt["errors"]}')
    histograms = json.loads((Path(receipt['folder']) / 'histograms.json').read_text())
    maxima = receipt['decode_maxima']
    require(maxima['prepared_decode_requests'] > 0, 'No decode generation was observed')
    if cap is not None:
        require(all(maxima[key] <= cap for key in LOGICAL_MAXIMA), 'Actual D geometry exceeded the configured admission cap')
    padding = histograms['prepared_decode_padding_histogram']
    distribution = Counter()
    modes = Counter()
    for row in padding:
        distribution[row['decode_requests']] += row['calls']
        modes[row['graph_mode']] += row['calls']
    token_distribution = Counter()
    for row in histograms['prepared_query_histogram']:
        requests = row['decode_only_requests'] + row['mixed_requests']
        token_distribution[requests] += row['decode_query_rows'] * row['calls']
    return {'valid': True, 'maximum': maxima, 'actual_cap_reached': maxima['prepared_decode_requests'] == cap
            if cap is not None else None, 'prepared_decode_call_histogram': dict(sorted(distribution.items())),
            'decode_query_rows_weighted_histogram': dict(sorted(token_distribution.items())),
            'graph_mode_histogram': dict(sorted(modes.items()))}


def weighted_distribution(histogram):
    histogram = {key: weight for key, weight in sorted(histogram.items()) if weight > 0}
    total = sum(histogram.values())

    def quantile(fraction):
        cumulative = 0
        for key, weight in histogram.items():
            cumulative += weight
            if cumulative >= fraction * total:
                return key
        return None

    return {'histogram': histogram, 'total_weight': total, 'p50': quantile(.5), 'p95': quantile(.95),
            'mean': sum(key * weight for key, weight in histogram.items()) / total if total else None}


def http_timeline(rows, cap):
    events = {}
    for row in rows:
        events.setdefault(row['submit'], [0, 0])[0] += 1
        events.setdefault(row['finish'], [0, 0])[1] += 1
    last = min(events)
    pending, active = len(rows), 0
    seconds_by_inflight = Counter()
    binding = 0.0
    for point, (starts, ends) in sorted(events.items()):
        span = point - last
        seconds_by_inflight[active] += span
        if pending and active == cap:
            binding += span
        active += starts - ends
        pending -= starts
        last = point
    require(active == pending == 0, 'D HTTP overlap accounting did not drain fully')
    return {'inflight_histogram_seconds': dict(seconds_by_inflight),
            'cap_binding_with_pending_seconds': binding if cap is not None else None}


def worker_diagnostics(port, log_path):
    """Snapshot outside the timed window; worker GPU allocation peaks are cumulative."""
    result = {}
    try:
        worker, = base.rpc_get(port, 'worker_info')
        result['worker_max_memory_allocated_bytes_cumulative'] = worker['maximum_allocated_bytes']
        result['D_KV_storage_bytes'] = worker['unique_KV_storage_bytes']
    except (OSError, TimeoutError, KeyError, ValueError) as exc:
        result['worker_error'] = str(exc)
    try:
        with urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
            lines = response.read().decode().splitlines()
        for metric, label in (('vllm:num_preemptions_total', 'preemptions_total'),
                              ('vllm:kv_cache_usage_perc', 'KV_usage_fraction_snapshot')):
            values = [float(line.split()[-1]) for line in lines if line.startswith((metric + '{', metric + ' '))]
            result[label] = values[0] if len(values) == 1 else None
    except (OSError, ValueError) as exc:
        result['metrics_error'] = str(exc)
    if log_path is not None and log_path.is_file():
        result['D_worker_log_bytes'] = log_path.stat().st_size
    return result


def diagnostic_delta(before, after, log_path):
    a, b = (state.get('preemptions_total') for state in (before, after))
    log_start, log_end = (state.get('D_worker_log_bytes') for state in (before, after))
    oom = None
    if log_path is not None and log_start is not None and log_end is not None and log_end >= log_start:
        with log_path.open('rb') as stream:
            stream.seek(log_start)
            data = stream.read(log_end - log_start).lower().splitlines()
        markers = (b'outofmemoryerror', b'cuda out of memory', b'cuda error: out of memory')
        oom = sum(any(marker in line for marker in markers) for line in data)
    return {'preemption_counter_delta': int(b - a) if a is not None and b is not None and b >= a else None,
            'D_worker_OOM_diagnostic_log_lines': oom,
            'GPU_device_peak_memory_bytes': None, 'D_actual_peak_KV_usage_bytes': None,
            'scope': 'Counter snapshots outside D drain. Allocator peak is cumulative since worker startup. '
                     'OOM matching counts logged lines. GPU/device and KV usage peaks were not sampled.',
            'before': before, 'after': after}


def rate_row(records, cohorts, cap, repeat):
    seconds = sum(row['duration_seconds'] for row in cohorts)
    output = sum(row['output_tokens'] for row in records)
    require(len(records) == 100 and output == OUTPUT_TOKEN_CONTRACT, 'D-only output scope changed')
    geometry = [row['actual_batch']['maximum'] for row in cohorts]
    distribution = Counter()
    decode_query_rows = Counter()
    inflight_seconds = Counter()
    modes = Counter()
    for cohort in cohorts:
        distribution.update(cohort['actual_batch']['prepared_decode_call_histogram'])
        decode_query_rows.update(cohort['actual_batch']['decode_query_rows_weighted_histogram'])
        inflight_seconds.update(cohort['HTTP_timeline']['inflight_histogram_seconds'])
        modes.update(cohort['actual_batch']['graph_mode_histogram'])
    itl = [value for row in records for value in row['ITL_ms']]
    binding_seconds = sum(row['HTTP_timeline']['cap_binding_with_pending_seconds'] for row in cohorts) if cap is not None else None
    return {'evidence': 'ISOLATED_GPU_MEASUREMENT', 'cap': cap_label(cap), 'repeat': repeat,
            'requests': len(records), 'tokens': output, 'seconds': seconds,
            'requests_per_second': len(records) / seconds, 'tokens_per_second': output / seconds,
            'service_p50_ms': statistics.median((row['finish'] - row['submit']) * 1000 for row in records),
            'mean_TBT_ms': statistics.mean(itl) if itl else None, 'ITL_p99_ms': base.percentile(itl, .99),
            'TPOT_p95_ms': base.percentile([row['TPOT_ms'] for row in records], .95),
            'KV_input_tokens_p50': base.percentile([row['input_tokens'] for row in records], .5),
            'KV_input_tokens_p95': base.percentile([row['input_tokens'] for row in records], .95),
            'maximum_D_HTTP_inflight': max(row['maximum_D_HTTP_inflight'] for row in cohorts),
            'maximum_actual_D_geometry': {key: max(row[key] for row in geometry) for key in geometry[0]},
            'actual_cap_reached': any(row['actual_batch']['actual_cap_reached'] for row in cohorts) if cap is not None else None,
            'prepared_decode_call_histogram': dict(sorted(distribution.items())),
            'graph_mode_histogram': dict(sorted(modes.items())),
            'actual_D_batch_distributions': {
                'forward_step_weighted': weighted_distribution(distribution),
                'request_forward_occurrence_weighted': weighted_distribution({key: key * value for key, value in distribution.items()}),
                'decode_query_rows_weighted': weighted_distribution(decode_query_rows),
                'GPU_time_weighted': None, 'unique_request_weighted': None,
            },
            'D_HTTP_inflight_time_weighted': weighted_distribution(inflight_seconds),
            'D_HTTP_cap_binding_with_pending_seconds': binding_seconds,
            'D_HTTP_cap_binding_with_pending_fraction': binding_seconds / seconds if binding_seconds is not None else None,
            'cohorts': cohorts,
            'scope': 'Cohort drain time includes D import and SSE; original E/P preparation and EC release are excluded. '
                     'Prepared geometry and padding are observed. Finite-cohort stage measurement only.'}


async def measure_decode_sweep(root, rows, replay, caps=CAPS, *, lease_seconds=STAGE_KV_LEASE_SECONDS,
                               ports=None, repeats=1, tag='capacity_decode_wide'):
    """Measure all specified admission arms inside exclusively owned E/P/D workers."""
    import aiohttp
    from ..mllm_epd_prefill_heavy_token_split import client

    root = Path(root)
    caps = normalized_caps(caps)
    if lease_seconds != STAGE_KV_LEASE_SECONDS or repeats < 1 or not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Use the fixed 180s isolated NIXL lease, positive repeat count and a safe tag')
    ports = PORTS if ports is None else ports
    encoder, prefill, decode = (ports[key] for key in ('encoder', 'prefill', 'decode'))
    cohort_rows, plans = planned_cohorts(rows, replay)
    for port in (prefill, decode):
        worker, = await asyncio.to_thread(base.rpc_get, port, 'worker_info')
        scheduler = worker['scheduler']
        require(worker['unique_KV_storage_bytes'] == TOTAL_KV_BYTES and scheduler['B'] == 8192 and scheduler['S'] == 1024,
                'D-only protocol requires P/D B8192/S1024 with a 42GiB KV pool')
    raw = root / 'raw' / tag
    raw.mkdir()
    config = {'caps': [cap_label(cap) for cap in caps], 'repeat_count': repeats,
              'caller_main100_order_seed': COHORT_ORDER_SEED, 'cohort_plans': plans, 'kv_pool_bytes': TOTAL_KV_BYTES,
              'kv_cohort_limit_bytes': KV_COHORT_LIMIT, 'block_tokens': base.KV_BLOCK_TOKENS,
              'kv_bytes_per_token': base.KV_BYTES_PER_TOKEN, 'NIXL_stage_lease_seconds': lease_seconds,
              'lease_reserve_before_D_admission_seconds': MIN_INITIAL_LEASE_REMAINING,
              'counts_SHA256': hashlib.sha256(json.dumps(replay, sort_keys=True).encode()).hexdigest(),
              'scope': 'Cap is open FIFO D HTTP admission; engine S remains 1024. All caps use identical full Main100 cohorts.'}
    base.write_json(root / 'configs' / f'{tag}.json', config)
    previous, = await asyncio.to_thread(rpc, encoder, 'ec_ipc/inline_policy', {
        'enabled': True, 'consumer_role': 'prefill',
    })
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=600, sock_read=300)
    results = []
    logs = list((root / 'logs').glob('*_decode.log'))
    d_log = logs[0] if len(logs) == 1 else None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        for repeat in range(repeats):
            rotation = repeat % len(caps)
            for cap in caps[rotation:] + caps[:rotation]:
                arm = f'{tag}_{cap_label(cap)}_r{repeat}'
                arm_rows, arm_cohorts = [], []
                diagnostics_before = await asyncio.to_thread(worker_diagnostics, decode, d_log)
                for index, (cohort, plan) in enumerate(zip(cohort_rows, plans, strict=True)):
                    identifier = f'{arm}_c{index:02d}'
                    names = [f'{identifier}_{item:04d}' for item in range(len(cohort))]
                    jobs = list(zip(cohort, names, strict=True))
                    encoded = await base.collect_all(*(base.stage_request(session, encoder, row, name, 'E:full', client,
                                                                         output_count=replay[row['request_id']])
                                                       for row, name in jobs))
                    exports = await base.collect_all(*(base.stage_request(session, prefill, row, name, 'P', client,
                                                                         export=True, output_count=replay[row['request_id']])
                                                       for row, name in jobs))
                    blocks = verify_exported_blocks(exports, plan)
                    base.write_json(raw / f'{identifier}_prepared.json', {'encoder': encoded, 'prefill_export': exports})
                    async with base.batch_counter(decode, identifier, decode_only=True) as counter:
                        completed, active = await drain_decode(session, decode, cohort, names, exports,
                                                               replay, client, cap, lease_seconds)
                    base.write_json(raw / f'{identifier}_requests.json', completed)
                    end, = counter['end']
                    geometry = geometry_summary(end, cap)
                    require(cap is None or active <= cap, 'Aggregate active D HTTP stream cap exceeded')
                    release = await base.release(session, encoder, names)
                    metrics = base.cohort_metrics(completed, 'decode')
                    metrics.update(actual_batch=geometry, maximum_D_HTTP_inflight=active, P_export_blocks=blocks,
                                   planned_D_peak_KV_bytes=plan['planned_D_peak_bytes'],
                                   HTTP_timeline=http_timeline(completed, cap),
                                   minimum_conservative_lease_remaining=min(row['conservative_P_lease_at_admission'] for row in completed),
                                   EC_release=release, counter=counter)
                    base.write_json(raw / f'{identifier}_metrics.json', metrics)
                    arm_rows.extend(completed)
                    arm_cohorts.append(metrics)
                result = rate_row(arm_rows, arm_cohorts, cap, repeat)
                diagnostics_after = await asyncio.to_thread(worker_diagnostics, decode, d_log)
                result['diagnostics'] = diagnostic_delta(diagnostics_before, diagnostics_after, d_log)
                base.write_json(root / 'analysis' / f'{arm}.json', result)
                print('DECODE_CAPACITY', cap_label(cap), result['requests_per_second'],
                      result['tokens_per_second'], result['maximum_actual_D_geometry'], flush=True)
                results.append(result)
    await asyncio.to_thread(rpc, encoder, 'ec_ipc/inline_policy', {
        'enabled': previous['before'] is not None, 'consumer_role': previous['before'] or 'prefill',
    })
    summary = {'evidence': 'ISOLATED_GPU_MEASUREMENT', 'config': config, 'arms': results}
    base.write_json(root / 'analysis' / f'{tag}.json', summary)
    return summary
