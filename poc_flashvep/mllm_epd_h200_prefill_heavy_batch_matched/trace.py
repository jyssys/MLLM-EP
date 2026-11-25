"""Build and qualify the pinned kernel-only collector on the current host."""

import argparse
import asyncio
import csv
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen


COLLECTOR_PATH = 'poc_flashvep/reports/mllm_epd_prefill_heavy_token_split_20261124/references/kernel_only_collector_v3/activity_kernel_only_v3.cc'
COLLECTOR_SHA256 = 'fb1dbc4f54f21190ba0b67602ebd640253ce08e3a177a106c66ac5906ac9d38e'
COMPARISON_METRICS = ('duration_seconds', 'TTFT_p50_ms', 'TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms')


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_collector(repo, root, runtime):
    """Compile the exact tracked C++ source using the installed CUDA 12.9 headers."""
    target = Path(root).resolve() / 'references/kernel_only_collector_v3'
    receipt = target / 'build.json'
    if receipt.exists():
        value = json.loads(receipt.read_text())
        if value.get('source_SHA256') != COLLECTOR_SHA256 or any(
            file_sha256(Path(value[key])) != value[key + '_SHA256']
            for key in ('source', 'library', 'CUPTI_library')
        ):
            raise RuntimeError('Existing collector receipt failed the hash check')
        return value

    code = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{COLLECTOR_PATH}'])
    if hashlib.sha256(code).hexdigest() != COLLECTOR_SHA256:
        raise RuntimeError('Pinned collector source hash changed')
    target.mkdir(parents=True)
    source = target / Path(COLLECTOR_PATH).name
    source.write_bytes(code)
    library = target / 'activity.so'
    runtime = Path(runtime).resolve()
    cupti = runtime / 'lib/python3.14/site-packages/nvidia/cuda_cupti'
    cuda = runtime / 'targets/x86_64-linux'
    cupti_library = cupti / 'lib/libcupti.so.12'
    compiler = runtime / 'bin/x86_64-conda-linux-gnu-c++'
    command = [str(compiler), '-std=c++17', '-O2', '-fPIC', '-shared',
               '-I' + str(cuda / 'include'), '-I' + str(cupti / 'include'), str(source),
               str(cupti_library), '-Wl,-rpath,' + str(cupti_library.parent),
               '-o', str(library)]
    subprocess.run(command, check=True)
    paths = {'source': source, 'library': library, 'CUPTI_library': cupti_library}
    result = {key: str(path) for key, path in paths.items()}
    result.update({key + '_SHA256': file_sha256(path) for key, path in paths.items()})
    result['command'] = command
    result['scope'] = 'Concurrent CUDA kernels only for campaign collection. Clean/profile/clean qualification required.'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def rpc(port, path, body):
    request = Request(f'http://127.0.0.1:{port}/campaign/{path}',
                      json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


@asynccontextmanager
async def profile_session(ports, name, state_interval_ms=50):
    """Collect three kernel streams and rate-limited scheduler state outside timed setup."""
    receipts = {}
    try:
        for role, port in ports.items():
            value = await asyncio.to_thread(rpc, port, 'profile/begin', {
                'name': name, 'include_copies': False, 'state_interval_ms': state_interval_ms,
            })
            receipts[role] = {'begin': value}
        yield receipts
    finally:
        roles = list(receipts)
        results = await asyncio.gather(*(
            asyncio.to_thread(rpc, ports[role], 'profile/end', {}) for role in roles
        ), return_exceptions=True)
        for role, result in zip(roles, results, strict=True):
            if isinstance(result, BaseException):
                receipts[role]['close_error'] = str(result)
            else:
                receipts[role]['end'] = result


def cpc_gate(clean_before, profile, clean_after, receipts, threshold=.05):
    comparisons = {}
    for key in COMPARISON_METRICS:
        a, b, c = clean_before.get(key), profile.get(key), clean_after.get(key)
        center = (a + c) / 2 if a is not None and c is not None else 0
        spread = abs(c - a) / center if center else None
        overhead = abs(b / center - 1) if center and b is not None else None
        comparisons[key] = {'clean_before': a, 'profile': b, 'clean_after': c,
                            'clean_relative_spread': spread, 'profile_relative_deviation': overhead,
                            'passed': spread is not None and overhead is not None
                            and max(spread, overhead) <= threshold}
    collectors_pass = len(receipts) == 3 and all(
        len(value.get('end', [])) == 1 and value['end'][0]['passed']
        and value['end'][0]['counters']['all_kernels'] > 0 for value in receipts.values()
    )
    passed = collectors_pass and all(value['passed'] for value in comparisons.values())
    return {'passed': passed, 'collector_passed': collectors_pass, 'threshold': threshold,
            'metrics': comparisons, 'evidence': 'TRACE_ANALYSIS' if passed else 'DIAGNOSTIC'}


def merged(intervals, left, right):
    intervals = sorted((max(left, a), min(right, b)) for a, b in intervals if a < right and b > left and a < b)
    output = []
    for a, b in intervals:
        if output and output[-1][1] >= a:
            output[-1] = (output[-1][0], max(output[-1][1], b))
        else:
            output.append((a, b))
    return output


def complement(intervals, left, right):
    starts = [left] + [b for _, b in intervals]
    ends = [a for a, _ in intervals] + [right]
    return [(a, b) for a, b in zip(starts, ends, strict=True) if a < b]


def intersect(first, second):
    i = j = 0
    overlap = []
    while i < len(first) and j < len(second):
        a, b = first[i]
        c, d = second[j]
        if max(a, c) < min(b, d):
            overlap.append((max(a, c), min(b, d)))
        if b <= d:
            i += 1
        else:
            j += 1
    return overlap


def elapsed(intervals):
    return sum(b - a for a, b in intervals)


def kernel_intervals(folder):
    roi = json.loads((folder / 'actual_ROI.json').read_text())
    a, b = roi['left'], roi['right']
    anchor = lambda item: (item['host_lower_ns'] + item['host_upper_ns']) / 2
    scale = (anchor(b) - anchor(a)) / (b['cupti_ns'] - a['cupti_ns'])
    max_bracket_ns = max(item['host_upper_ns'] - item['host_lower_ns'] for item in (a, b))
    if not .99 <= scale <= 1.01 or max_bracket_ns > 1_000_000:
        raise RuntimeError(f'Unreliable cross-stage clock alignment in {folder}')

    def host_seconds(cupti_ns):
        return (anchor(a) + scale * (int(cupti_ns) - a['cupti_ns'])) / 1e9

    with (folder / 'kernels.csv').open() as stream:
        intervals = [(host_seconds(row['start_ns']), host_seconds(row['end_ns']))
                     for row in csv.DictReader(stream) if row['kind'] == 'kernel']
    return intervals, (anchor(a) / 1e9, anchor(b) / 1e9), {
        'passed': roi['passed'], 'clock_scale': scale, 'max_bracket_ns': max_bracket_ns,
        'records': len(intervals),
    }


def scheduler_intervals(folder, left, right):
    record = json.loads((folder / 'scheduler.json').read_text())
    samples = sorted(record['rows'], key=lambda row: row['begin_ns'])
    cap = record['sampling_interval_ms'] / 1000 * 1.5
    pairs = [(row, row['begin_ns'] / 1e9, next_row['begin_ns'] / 1e9)
             for row, next_row in zip(samples, samples[1:])]
    observed = merged(((a, min(b, a + cap)) for _, a, b in pairs), left, right)
    waiting = merged(((a, min(b, a + cap)) for row, a, b in pairs if row['before']['waiting']), left, right)
    pending = [row['before'] for row in samples if left <= row['begin_ns'] / 1e9 <= right]
    return waiting, {
        'observations': len(pending), 'window_fraction_with_recent_sample': elapsed(observed) / (right - left),
        'window_fraction_recent_sample_waiting': elapsed(waiting) / (right - left),
        'waiting_nonempty_fraction_per_sample': sum(bool(row['waiting']) for row in pending) / len(pending)
        if pending else None,
        'waiting_max': max((row['waiting'] for row in pending), default=None),
        'active_decode_max': max((row['decode_active'] for row in pending), default=None),
        'free_KV_blocks_min': min((row['free_KV_blocks'] for row in pending
                                   if row['free_KV_blocks'] is not None), default=None),
        'scope': 'Conditional Scheduler.schedule sampling. A sample is held at most 1.5 intervals. Unobserved state remains unknown.',
    }


def summarize_trace(profile_dir, client_rows, proxy_rows):
    """Use the central 20-80% scheduled-arrival window and expose raw union intervals."""
    arrivals = sorted(row['scheduled_arrival'] for row in client_rows)
    if len(arrivals) < 5:
        raise ValueError('Insufficient arrivals for a central online window')
    left, right = (arrivals[int(len(arrivals) * fraction)] for fraction in (.2, .8))
    if left >= right:
        raise ValueError('Degenerate central arrival window')
    duration = right - left
    busy, clocks, scheduler, waiting = {}, {}, {}, {}
    for role in ('encoder', 'prefill', 'decode'):
        folder = Path(profile_dir) / role
        intervals, coverage, clocks[role] = kernel_intervals(folder)
        if not (coverage[0] <= left and right <= coverage[1]):
            raise ValueError(f'Collector does not fully cover the central arrival window: {role}')
        busy[role] = merged(intervals, left, right)
        waiting[role], scheduler[role] = scheduler_intervals(folder, left, right)

    d_idle = complement(busy['decode'], left, right)
    d_http = merged(((row['D_submit'], row['finished']) for row in proxy_rows
                     if 'D_submit' in row and 'finished' in row), left, right)
    d_pending = merged(((row['P_ready'], row['D_submit']) for row in proxy_rows
                        if 'P_ready' in row and 'D_submit' in row), left, right)
    d_http_and_pending_idle = complement(merged(d_http + d_pending, left, right), left, right)
    p_ready_to_submit = merged(((row['E_complete'], row['P_submit']) for row in proxy_rows
                                if 'E_complete' in row and 'P_submit' in row), left, right)
    p = busy['prefill']
    p_pending = merged(((row['P_submit'], row['P_ready']) for row in proxy_rows
                        if 'P_submit' in row and 'P_ready' in row), left, right)
    p_starved_proxy = intersect(complement(p, left, right), complement(p_pending, left, right))
    d_quiet = intersect(d_idle, d_http_and_pending_idle)
    by_id = {row['runtime_request_id']: row for row in client_rows}
    rates = {}
    for role, key, field in (('encoder', 'E_complete', 'input_tokens_manifest'),
                             ('prefill', 'P_ready', 'input_tokens_manifest'),
                             ('decode', 'finished', 'output_tokens')):
        completed = [by_id[row['runtime_request_id']] for row in proxy_rows
                     if row.get('status') == 'MEASURED_ONLINE' and left <= row.get(key, -1) < right]
        rates[role] = {'requests': len(completed), 'requests_per_second': len(completed) / duration,
                       'tokens_per_second': sum(row[field] for row in completed) / duration if role != 'encoder' else None,
                       'token_field': field if role != 'encoder' else 'visual rows require actual encoder evidence'}
    simultaneous = {
        'P_kernel_D_kernel_idle': intersect(p, d_idle),
        'P_kernel_D_kernel_idle_no_D_HTTP_or_postprefill_wait': intersect(p, d_quiet),
        'P_ready_before_HTTP_and_D_kernel_idle': intersect(p_ready_to_submit, d_idle),
        'P_kernel_and_P_ready_before_HTTP_and_D_kernel_idle': intersect(intersect(p, p_ready_to_submit), d_idle),
        'P_recent_scheduler_waiting_D_kernel_idle': intersect(waiting['prefill'], d_idle),
        'P_kernel_and_recent_scheduler_waiting_D_kernel_idle': intersect(intersect(p, waiting['prefill']), d_idle),
    }
    return {
        'window_rule': 'Scheduled arrivals at observed indices floor(N*.20) to floor(N*.80). End-of-trace drain excluded.',
        'window_monotonic': [left, right], 'duration_seconds': duration, 'clocks': clocks,
        'kernel_busy_fraction': {role: elapsed(values) / duration for role, values in busy.items()},
        'kernel_intervals': busy, 'D_physical_idle_windows_seconds': [b - a for a, b in d_idle],
        'D_no_HTTP_or_postprefill_wait_windows_seconds': [b - a for a, b in d_http_and_pending_idle],
        'D_no_HTTP_or_postprefill_wait_fraction': elapsed(d_http_and_pending_idle) / duration,
        'D_physical_idle_no_HTTP_or_postprefill_wait_windows_seconds': [b - a for a, b in d_quiet],
        'P_embedding_ready_before_HTTP_fraction': elapsed(p_ready_to_submit) / duration,
        'P_kernel_idle_no_P_HTTP_fraction': elapsed(p_starved_proxy) / duration,
        'scheduler_samples': scheduler, 'online_completed_stage_rates': rates,
        'simultaneous': {key: {'seconds': elapsed(values), 'fraction_all': elapsed(values) / duration,
                              'fraction_P_busy': elapsed(values) / elapsed(p) if p and key.startswith('P_kernel') else None}
                         for key, values in simultaneous.items()},
        'boundary_scope': 'D HTTP retains conservative activity through proxy finish. P ready-to-submit only a host queue before HTTP; engine scheduler and KV backlog need separate evidence.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_collector(args.repo, args.root, args.runtime), indent=2))


if __name__ == '__main__':
    main()
