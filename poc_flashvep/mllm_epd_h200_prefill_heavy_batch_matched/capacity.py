"""Isolate each H200 stage while preserving the real image and KV paths."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

from .trace import kernel_intervals, merged, elapsed, profile_session, rpc


GIB = 1024 ** 3
VISION_BYTES_PER_TOKEN = 2 * 3584
KV_BYTES_PER_TOKEN = 2 * 2 * 28 * 4 * 128
KV_BLOCK_TOKENS = 128
E_BUFFER_BYTES = 6 * GIB
D_MAX_REQUESTS = 12
D_MAX_OUTPUT_TOKENS = 620
P_EXPORT_MINIMUM_REMAINING = 5.0
P_EXPORT_DEFAULT_TTL = 30.0
PULL_PARAMS = {'do_remote_decode': True, 'do_remote_prefill': False,
               'remote_engine_id': None, 'remote_block_ids': None,
               'remote_host': None, 'remote_port': None}


def groups(rows, max_requests=None, replay=None):
    """Keep original order and bound live encoder allocation and NIXL credit."""
    cohorts, pending = [], []
    embedding_bytes = kv_bytes = output_tokens = 0
    for row in rows:
        visual = row['visual_tokens'] * VISION_BYTES_PER_TOKEN
        tokens = row['total_input_tokens']
        padded = (tokens + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS * KV_BLOCK_TOKENS
        kv = padded * KV_BYTES_PER_TOKEN
        generated = replay[row['request_id']] if replay is not None else 0
        if visual > E_BUFFER_BYTES or kv > 32 * GIB:
            raise ValueError(f'Single request exceeds capacity probe budget: {row["request_id"]}')
        if pending and (embedding_bytes + visual > E_BUFFER_BYTES or (max_requests is not None and
                        (len(pending) == max_requests or kv_bytes + kv > 32 * GIB
                         or output_tokens + generated > D_MAX_OUTPUT_TOKENS))):
            cohorts.append(pending)
            pending = []
            embedding_bytes = kv_bytes = output_tokens = 0
        pending.append(row)
        embedding_bytes += visual
        kv_bytes += kv
        output_tokens += generated
    if pending:
        cohorts.append(pending)
    return cohorts


def headers(base, stage):
    return {'x-request-id': base + ':' + stage,
            'x-campaign-metadata-key': hashlib.sha256(base.encode()).hexdigest(),
            'x-campaign-metadata-mode': 'grid_only'}


def post_rpc(session, port, endpoint, body):
    return session.post(f'http://127.0.0.1:{port}/campaign/{endpoint}', json=body)


async def json_reply(request, stage):
    async with request as response:
        if response.status != 200:
            raise RuntimeError(f'{stage} {response.status}: {(await response.text())[:1000]}')
        return await response.json()


async def stage_request(session, port, row, name, stage, client, export=False):
    body = client.payload(row, name, None, stream=False)
    body['max_tokens'] = 1
    if export:
        body['kv_transfer_params'] = dict(PULL_PARAMS)
    submit = time.monotonic()
    response = await json_reply(session.post(f'http://127.0.0.1:{port}/v1/chat/completions',
                                              json=body, headers=headers(name, stage)), stage)
    finish = time.monotonic()
    input_tokens = response['usage']['prompt_tokens']
    if input_tokens != row['total_input_tokens']:
        raise RuntimeError(f'Prompt length changed for {row["request_id"]}')
    result = {'request_id': row['request_id'], 'runtime_request_id': name,
              'submit': submit, 'finish': finish, 'input_tokens': input_tokens,
              'visual_tokens': row['visual_tokens'], 'image_count': row['image_count']}
    if export:
        params = response.get('kv_transfer_params') or {}
        if not params.get('do_remote_prefill') or not params.get('remote_block_ids'):
            raise RuntimeError(f'Missing original P NIXL export for {row["request_id"]}')
        if params['remote_num_tokens'] != input_tokens:
            raise RuntimeError(f'Incomplete P NIXL export for {row["request_id"]}')
        result['kv_transfer_params'] = params
    return result


async def release(session, encoder, names):
    # Release only after the cohort's last consumer has completed and acknowledged its clone.
    for name in names:
        result = await json_reply(post_rpc(session, encoder, 'ec_ipc/release', {
            'key': headers(name, 'E')['x-campaign-metadata-key'],
        }), 'E release')
        if len(result) != 1:
            raise RuntimeError('Expected one encoder release receipt')
    state = await asyncio.to_thread(rpc_get, encoder, 'ec_ipc/state')
    if len(state) != 1 or state[0]['live_images'] or state[0]['outstanding_exports']:
        raise RuntimeError('Encoder ownership remains live after cohort')
    return state[0]


def rpc_get(port, path):
    from urllib.request import urlopen
    with urlopen(f'http://127.0.0.1:{port}/campaign/{path}', timeout=180) as response:
        return json.load(response)


class DecodeSession:
    """Use original SSE reader and append exact grid-only metadata headers."""

    def __init__(self, session, name):
        self.session = session
        self.name = name

    def post(self, url, **kwargs):
        kwargs['headers'] = headers(self.name, 'D')
        return self.session.post(url, **kwargs)


class MemorySink:
    def record(self, row):
        # client.one already returns this result. No per-request Lustre write in the D timing window.
        pass


async def drain_decode(session, port, rows, names, exports, replay, client, batch_one):
    async def send(index):
        row, name, export = rows[index], names[index], exports[index]
        if time.monotonic() - export['finish'] >= P_EXPORT_DEFAULT_TTL - P_EXPORT_MINIMUM_REMAINING:
            raise TimeoutError(f'NIXL export for {row["request_id"]} too close to default 30s expiry')
        payload = client.payload(row, name, replay[row['request_id']])
        payload['kv_transfer_params'] = export['kv_transfer_params']
        url = f'http://127.0.0.1:{port}'
        result = await client.one(DecodeSession(session, name), url, row, payload,
                                  time.monotonic(), name, index, evidence_sink=MemorySink())
        if result['status'] != 'MEASURED_ONLINE' or result['output_tokens'] != replay[row['request_id']]:
            raise RuntimeError(f'D-only replay failed: {result.get("error", row["request_id"])}')
        result['status'] = 'ISOLATED_GPU_MEASUREMENT'
        result['runtime_request_id'] = name
        result['submit'] = result['dispatch']
        result['input_tokens'] = row['total_input_tokens']
        return result

    if batch_one:
        return [await send(index) for index in range(len(rows))]
    return await collect_all(*(send(index) for index in range(len(rows))))


async def collect_all(*awaitables):
    """Let submitted requests drain before raising, including background failures."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


@asynccontextmanager
async def batch_counter(port, name, decode_only=False):
    options = {'name': name}
    if decode_only:
        options['decode_only'] = True
    begin = await asyncio.to_thread(rpc, port, 'batch_counter/begin', options)
    record = {'begin': begin}
    try:
        yield record
    finally:
        record['end'] = await asyncio.to_thread(rpc, port, 'batch_counter/end', {})


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values) - 1) * quantile
    lower = int(offset)
    return values[lower] + (offset - lower) * (values[min(lower + 1, len(values) - 1)] - values[lower])


def cohort_metrics(rows, stage):
    end = max(row['finish'] for row in rows)
    start = min(row['submit'] for row in rows)
    token_field = {'encoder': 'visual_tokens', 'prefill': 'input_tokens', 'decode': 'output_tokens'}[stage]
    duration = end - start
    raw_tokens = sum(row[token_field] for row in rows)
    return {'duration_seconds': duration, 'begin_monotonic': start, 'end_monotonic': end,
            'requests': len(rows), 'tokens': raw_tokens, 'requests_per_second': len(rows) / duration,
            'tokens_per_second': raw_tokens / duration, 'token_field': token_field,
            'service_p50_ms': statistics.median((row['finish'] - row['submit']) * 1000 for row in rows),
            'images_per_second': sum(row['image_count'] for row in rows) / duration if stage == 'encoder' else None,
            'ITL_p99_ms': percentile([value for row in rows for value in row.get('ITL_ms', [])], .99),
            'TPOT_p95_ms': percentile([row['TPOT_ms'] for row in rows if 'TPOT_ms' in row], .95)}


def attach_actual_batch(result, end_counter, stage, batch_one):
    histograms = json.loads((Path(end_counter['folder']) / 'histograms.json').read_text())
    result['actual_batch_valid'] = end_counter['valid']
    if stage == 'encoder':
        encoded = histograms['encoder_output_histogram']
        actual = sum(row['output_visual_rows'] * row['calls'] for row in encoded)
        result['actual_visual_embedding_rows'] = actual
        result['actual_visual_embedding_rows_per_second'] = actual / result['duration_seconds']
        result['actual_visual_rows_match_manifest'] = bool(encoded) and actual == result['tokens']
        result['maximum_actual_encoder_image_batch'] = max((row['output_images'] for row in encoded), default=None)
        return
    prepared = histograms['prepared_query_histogram']
    if stage == 'prefill':
        actual = sum(row['prompt_query_rows'] * row['calls'] for row in prepared)
        result['actual_prompt_query_rows'] = actual
        result['actual_prompt_rows_match_manifest'] = bool(prepared) and actual == result['tokens']
        result['maximum_actual_prefill_request_batch'] = max((row['requests'] for row in prepared), default=None)
        return
    maxima = end_counter['decode_maxima']
    result['maximum_actual_prepared_decode_requests'] = maxima['prepared_decode_requests']
    result['maximum_actual_scheduled_generation_requests'] = maxima['scheduler_generation_phase_requests']
    result['batch_one_observed'] = end_counter['valid'] and (
        maxima['scheduled_requests'] == maxima['prepared_decode_requests'] == 1
        and maxima['scheduler_generation_phase_requests'] <= 1
    )
    if batch_one and not result['batch_one_observed']:
        raise RuntimeError('D-only Batch1 lacked a valid Actual Prepared/Scheduled Batch1 observation')


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


async def measure_stages(root, rows, stage, *, replay=None, batch_one=False, profile=False,
                         ports=None, tag=None):
    """Measure completed cohorts, excluding each cohort's off-stage construction."""
    import aiohttp
    from poc_flashvep.mllm_epd_prefill_heavy_token_split import client

    root = Path(root)
    ports = ports or {'encoder': 18840, 'prefill': 18841, 'decode': 18842}
    encoder, prefill, decode = (ports[key] for key in ('encoder', 'prefill', 'decode'))
    if stage not in ports or not rows:
        raise ValueError('Select a real E/P/D stage and nonempty cohort')
    if stage == 'decode' and (not replay or set(replay) != {row['request_id'] for row in rows}):
        raise ValueError('D-only requires original equal-output counts for every request')
    if profile and stage != 'prefill':
        raise ValueError('Isolated kernel profile is defined for P only')
    tag = tag or f'capacity_{stage}' + ('_b1' if batch_one else '_free')
    if not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Unsafe capacity trace tag')
    raw = root / 'raw' / tag
    raw.mkdir()
    cohorts = groups(rows, D_MAX_REQUESTS if stage == 'decode' else None, replay if stage == 'decode' else None)
    write_json(root / 'configs' / f'{tag}.json', {
        'stage': stage, 'batch_one': batch_one, 'cohorts': [[row['request_id'] for row in cohort] for cohort in cohorts],
        'E_buffer_budget_bytes': E_BUFFER_BYTES, 'KV_bytes_per_token': KV_BYTES_PER_TOKEN,
        'native_P_export_only_for_D_preparation': stage == 'decode', 'profile_first_cohort': profile,
        'D_P_export_lease_seconds': P_EXPORT_DEFAULT_TTL if stage == 'decode' else None,
        'D_output_tokens_per_cohort_limit': D_MAX_OUTPUT_TOKENS if stage == 'decode' else None,
    })
    original_policy, = await asyncio.to_thread(rpc, encoder, 'ec_ipc/inline_policy', {
        'enabled': stage != 'encoder', 'consumer_role': 'prefill',
    })
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=1600)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    stats = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        for index, cohort in enumerate(cohorts):
            cohort_tag = f'{tag}_{index:02d}'
            names = [f'{cohort_tag}_{item:04d}' for item in range(len(cohort))]
            jobs = list(zip(cohort, names, strict=True))

            async def encode():
                return await collect_all(*(stage_request(session, encoder, row, name, 'E:full', client)
                                           for row, name in jobs))

            async def prefill_rows(export=False):
                return await collect_all(*(stage_request(session, prefill, row, name, 'P', client, export)
                                           for row, name in jobs))

            prepared = {}
            if stage != 'encoder':
                prepared['encoder'] = await encode()
            if stage == 'decode':
                prepared['prefill_export'] = await prefill_rows(export=True)
            write_json(raw / f'{cohort_tag}_prepared.json', prepared)

            async def measure():
                if stage == 'encoder':
                    return await encode()
                if stage == 'prefill':
                    return await prefill_rows()
                return await drain_decode(session, decode, cohort, names, prepared['prefill_export'],
                                          replay, client, batch_one)

            trace = None
            async with batch_counter(ports[stage], cohort_tag, decode_only=stage == 'decode') as counter:
                if profile and index == 0:
                    async with profile_session({stage: ports[stage]}, cohort_tag, state_interval_ms=50) as trace:
                        values = await measure()
                else:
                    values = await measure()
            result = cohort_metrics(values, stage)
            result['counter'] = counter
            end_counter, = counter['end']
            attach_actual_batch(result, end_counter, stage, batch_one)
            if trace is not None:
                result['profile_receipts'] = trace
                kernels, coverage, clock = kernel_intervals(root / 'raw/profiles' / cohort_tag / stage)
                begin, end = result['begin_monotonic'], result['end_monotonic']
                if not clock['passed'] or not coverage[0] <= begin < end <= coverage[1]:
                    raise RuntimeError('Isolated P kernel observer did not cover measurement')
                result['P_kernel_busy_fraction'] = elapsed(merged(kernels, begin, end)) / (end - begin)
                result['P_kernel_evidence'] = 'DIAGNOSTIC unqualified isolated P profile'
            write_json(raw / f'{cohort_tag}_rows.json', values)
            result['encoder_release'] = await release(session, encoder, names)
            write_json(raw / f'{cohort_tag}_metrics.json', result)
            stats.append(result)
    await asyncio.to_thread(rpc, encoder, 'ec_ipc/inline_policy', {
        'enabled': original_policy['before'] is not None,
        'consumer_role': original_policy['before'] or 'prefill',
    })
    duration = sum(record['duration_seconds'] for record in stats)
    count = sum(record['requests'] for record in stats)
    tokens = sum(record['tokens'] for record in stats)
    result = {'evidence': 'ISOLATED_GPU_MEASUREMENT', 'stage': stage, 'batch_one': batch_one,
              'requests': count, 'tokens': tokens, 'seconds': duration,
              'requests_per_second': count / duration, 'tokens_per_second': tokens / duration,
              'cohorts': stats, 'scope': {
                  'encoder': 'Concurrent original image-encoder HTTP. No inline transport, and releases follow the measured interval.',
                  'prefill': 'Original E embeddings pre-staged, real decoder model P step with immediate local KV release; no D or exported KV retention.',
                  'decode': 'Official NIXL imports from fully pre-staged P KV with equal original output counts. Includes import and dispatch gaps. P/E preparation excluded.',
              }[stage], 'observer_qualification': 'PENDING for aggregate counter overhead and steady-state capacity'}
    if stage == 'encoder' and all(row['actual_batch_valid'] and row['actual_visual_rows_match_manifest'] for row in stats):
        result['actual_visual_embedding_rows_per_second'] = sum(row['actual_visual_embedding_rows'] for row in stats) / duration
    write_json(root / 'analysis' / f'{tag}.json', result)
    return result
