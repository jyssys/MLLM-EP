"""Same-three-GPU Natural/replay baselines with fresh independent starts."""
import asyncio
import json
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

from .client import execute, resolve_manifest, load_replay_counts
from .common import ROOT, REPO, PYTHON, digest, preserve, write, experiment_deadline
from .frontend_supply_control import wait_ready, retire
from .ipc_ec_control import launch as native_launch, stop_proxy, ownership_empty
from .llm_probe_control import rpc
from .metrics import summarize
from .source_snapshot import snapshot

URL = 'http://127.0.0.1:18847'
REFERENCE = ROOT / 'raw/reference_main400_natural_template_v2_v0_requests.jsonl'


def launch(family, tag):
    if family == 'native':
        names, proxy = native_launch(tag, 'cuda_ipc')
    else:
        ec = Path('/dev/shm') / f'mllm_epd_20261124_{tag}'
        assert not ec.exists()
        roles = [('agg', 4), ('agg', 5), ('agg', 6)] if family == 'agg' else \
                [('encoder', 4), ('pd', 5), ('pd', 6)]
        names = []
        for role, gpu in roles:
            name = f'{tag}_{role[0].upper()}{gpu}'
            names.append(name)
            args = [str(PYTHON), '-m', 'poc_flashvep.mllm_epd_prefill_heavy_token_split.servers',
                'start', '--name', name, '--gpu', str(gpu), '--role', role, '--observe',
                '--port', str(18840 + gpu - 4), '--side-port', str(19840 + gpu - 4),
                '--ec-path', str(ec), '--ec-transport', 'cuda_ipc']
            if role == 'encoder':
                args += ['--encoder-native-ops', '--compiled-encoder', '--encoder-sync-scheduling',
                         '--mm-cache-gb', '0', '--renderer-workers', '2']
            subprocess.run(args, cwd=REPO, check=True)
        wait_ready(names)
        ports = [18840, 18841, 18842] if family == 'agg' else [18841, 18842]
        command = [str(PYTHON), '-m', 'poc_flashvep.mllm_epd_prefill_heavy_token_split.proxy',
            '--mode', 'agg' if family == 'agg' else 'e_pd', '--name', tag + '_proxy',
            '--encoder', 'http://127.0.0.1:18840', '--prefill', 'http://127.0.0.1:18841',
            '--decoders', ','.join(f'http://127.0.0.1:{p}' for p in ports), '--port', '18847']
        if family == 'e_pd':
            command += ['--processor-handoff', 'grid_only', '--release-encoder-ipc']
        lease = json.loads((ROOT / 'state' / f'server_{names[0]}.json').read_text())
        with (ROOT / 'logs' / f'{tag}_proxy.log').open('xb') as stream:
            proxy = subprocess.Popen(command, cwd=lease['executed_source_cwd'], stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write(ROOT / 'state' / f'{tag}_proxy.json', dict(pid=proxy.pid, sid=proxy.pid, command=command,
            models=names, family=family, executed_source_cwd=lease['executed_source_cwd']))
        deadline = time.time() + 30
        while True:
            assert proxy.poll() is None, 'Inspect the terminal owned proxy before any restart'
            try:
                with urllib.request.urlopen(URL + '/health', timeout=2) as reply:
                    if reply.status == 200:
                        break
            except OSError:
                pass
            if time.time() >= deadline:
                raise TimeoutError('Inspect the same live proxy. Observation expiry is not terminal.')
            time.sleep(.5)
    if family != 'agg':
        for port in [18841, 18842]:
            rpc(port, 'ec_ipc/ack_policy', dict(mode='file'))
        role = 'prefill' if family == 'native' else 'pd'
        value = rpc(18840, 'ec_ipc/inline_policy', dict(enabled=True, consumer_role=role))
        assert len(value) == 1 and value[0]['inline_role'] == role
    return names, proxy


async def correctness(family, tag):
    manifest = ROOT / 'data/calibration_manifest_template_v2.json'
    all_rows = json.loads(manifest.read_text())['requests']
    rows = [all_rows[i] for i in [0, 1, 4, 6, 12, 17, 20, 31]]
    teacher = ROOT / 'raw/frontend_B5_cache0_T2_N8_natural_v0_requests.jsonl'
    reference = {r['request_id']: r for r in map(json.loads, teacher.read_text().splitlines())}
    comparisons = []
    for label, selected in [('N1', rows[:1]), ('N8', rows)]:
        values = await execute(selected, [URL], f'{tag}_{label}_natural_v0', .25, 2026112402, None, True,
            metadata=dict(evidence='CORRECTNESS_ONLY', family=family, manifest_SHA256=digest(manifest)))
        assert len(values) == len(selected) and all(r['status'] == 'MEASURED_ONLINE' for r in values)
        for row in values:
            ref = reference[row['request_id']]
            comparisons.append(dict(phase=label, request_id=row['request_id'],
                strict_prompt=row.get('prompt_token_ids') == ref.get('prompt_token_ids'),
                strict_tokens=row['token_ids'] == ref['token_ids']))
        assert all(r['strict_prompt'] for r in comparisons), 'Original tokenization changed'
    write(ROOT / 'analysis' / f'{tag}_correctness.json', dict(time=time.time(), comparisons=comparisons,
        prompt_identity_passed=True, full_output_strict=all(r['strict_tokens'] for r in comparisons),
        teacher_SHA256=digest(teacher), evidence='CORRECTNESS_ONLY', full_task_quality='PENDING',
        scope='Cross-batch/placement BI0 output differences are recorded, not overwritten. '
              'Strict prompt identity alone does not establish output quality. Original stock forwards remain unchanged.'))
    if family != 'agg':
        ownership_empty(tag + '_correctness', 'cuda_ipc')


async def measure(family, tag, label, qps, replay=False):
    path = resolve_manifest(None)
    rows = json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    counts = load_replay_counts(REFERENCE) if replay else None
    run = f'{tag}_{label}_q{qps}_v0'
    result = await execute(rows, [URL], run, qps, 2026112402, counts,
        metadata=dict(evidence='MEASURED_ONLINE_WITH_QUALITY_PENDING', family=family,
            manifest_SHA256=digest(path), GPU_group=[4, 5, 6], source_credit='No artificial source credit',
            D_cap='No artificial D cap', actual_scheduler='Stock vLLM continuous, chunked prefill, B8192/S1024',
            scope='Same three-GPU budget. Natural EOS and replay remain separate. Finite-cohort arrival and drain.'))
    assert len(result) == 100 and all(r['status'] == 'MEASURED_ONLINE' for r in result)
    if replay:
        assert all(r['output_tokens'] == counts[r['request_id']] for r in result)
    if family != 'agg':
        ownership_empty(run, 'cuda_ipc')
    value = summarize(run)
    print('BASELINE_C14_RESULT', family, run, {key: value[key] for key in
        ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'gate_attainment']}, flush=True)
    return value


def main():
    preserve()
    for role, gpu in [('E', 4), ('P', 5), ('D', 6)]:
        lease = json.loads((ROOT / 'state' / f'server_ec_PROFILE_C13_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc', str(lease['guardian'])).exists()
    plan = ROOT / 'configs/main100_baseline_confirmation_C14_v0.json'
    assert not plan.exists(), 'Inspect an existing campaign instead of restarting it'
    source = snapshot('main100_baseline_confirmation_C14_v0')
    write(plan, dict(time=time.time(), status='FROZEN_BEFORE_NEW_BASELINE_RESULTS',
        source_archive_SHA256=source['archive_SHA256'], manifest_SHA256=digest(resolve_manifest(None)),
        families=['agg', 'e_pd', 'native'], GPU_group=[4, 5, 6], independent_starts=3,
        primary_natural_QPS=[2, 4], natural_discovery_QPS=[.5, 1, 3, 6],
        replay_confirmation_QPS=2, order_seed=2026112404, arrival_seed=2026112402,
        reuse='Native q2 Natural and replay independent N3 from C11. Do not repeat that eligible runtime measurement.',
        native_new_scope='Natural q4 N3 and remaining Natural discovery points.',
        common_runtime='Original model/processor, compiled full28, BF16 BI0, no prefix cache, '
            'P/D or replica B8192/S1024, KV42GiB, block128, no source-credit/D-cap restriction. '
            'Disaggregated E cache0/T2/B114688, exact early GPU embedding delivery and completion ACK.',
        quality_gate='PENDING. Incomplete official MMLongDoc extraction cannot become a quality PASS.',
        no_generic_token_split=True, no_new_prefill_insertion_into_decode=True))
    for family in ['agg', 'e_pd', 'native']:
        for start in [1, 2, 3]:
            assert time.time() < experiment_deadline(), 'Large-experiment budget ended'
            tag = f'base_C14_{family}_S{start}'
            names = []
            proxy = None
            results = []
            try:
                names, proxy = launch(family, tag)
                asyncio.run(correctness(family, tag))
                asyncio.run(measure(family, tag, 'warmup_replay100', 2, True))
                if family != 'native':
                    results.append(asyncio.run(measure(family, tag, 'natural100', 2)))
                    results.append(asyncio.run(measure(family, tag, 'replay100', 2, True)))
                results.append(asyncio.run(measure(family, tag, 'natural100', 4)))
                if start == 1:
                    for qps in [.5, 1, 3, 6]:
                        results.append(asyncio.run(measure(family, tag, 'natural100', qps)))
                write(ROOT / 'analysis' / f'{tag}.json', dict(time=time.time(), status='COMPLETE',
                    family=family, independent_start=start, rows=results, full_task_quality='PENDING'))
            except BaseException as error:
                write(ROOT / 'state' / f'{tag}_failure.json', dict(time=time.time(), status='FAILED_EXECUTION',
                    error=type(error).__name__ + ': ' + str(error), partial_results=results))
                raise
            finally:
                if proxy is not None:
                    stop_proxy(proxy)
                if names:
                    retire([name for name in names if '_E4' not in name])
                    retire([name for name in names if '_E4' in name])
    preserve()
    write(ROOT / 'state/main100_baseline_confirmation_C14_complete.json', dict(time=time.time(), status='COMPLETE'))


if __name__ == '__main__':
    main()
