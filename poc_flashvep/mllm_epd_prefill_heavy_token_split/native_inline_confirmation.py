"""Independent Native EPD starts after the one-factor EC publication control."""
import argparse
import asyncio
import json
import random
import time
from types import SimpleNamespace

from .client import execute, load_replay_counts, resolve_manifest
from .common import ROOT, digest, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import launch, stop_proxy, correctness, ownership_empty
from .llm_probe_control import rpc
from .metrics import summarize
from .profile_control import run as profile_run
from .source_snapshot import snapshot

URL = 'http://127.0.0.1:18847'
REFERENCE = ROOT / 'raw/reference_main400_natural_template_v2_v0_requests.jsonl'


async def measure(tag, label, qps, natural=False):
    path = resolve_manifest(None)
    rows = json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    counts = None if natural else load_replay_counts(REFERENCE)
    run = f'{tag}_{label}_q{qps}_v0'
    values = await execute(rows, [URL], run, qps, 2026112402, counts,
        metadata=dict(evidence='DIAGNOSTIC_C3', manifest_SHA256=digest(path),
            EC_transport='cuda_ipc', native_inline_descriptors=True,
            completion_ACK='file', independent_initialization=tag,
            output_contract='Natural EOS' if natural else 'Per-request equal-output replay',
            quality_gate='PENDING', scope='Native runtime validation. No token split or extra prefill insertion.'))
    assert len(values) == 100 and all(r['status'] == 'MEASURED_ONLINE' for r in values)
    if counts is not None:
        assert all(r['output_tokens'] == counts[r['request_id']] for r in values)
    ownership_empty(run, 'cuda_ipc')
    value = summarize(run)
    print('NATIVE_INDEPENDENT_RESULT', run,
          {k: value[k] for k in ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'gate_attainment']}, flush=True)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    preserve()
    tag = f'ec_NATIVE_C11_S{args.start}'
    plan = ROOT / 'configs' / f'{tag}.json'
    assert not plan.exists(), 'Do not restart an existing initialization under the same identity'
    prior = ROOT / 'analysis/native_IPC_inline_control_v0.json'
    control = json.loads(prior.read_text())
    assert len(control['rows']) == 3 and all(r['completed_count'] == 100 for r in control['rows'])
    source = snapshot(tag)
    write(plan, dict(time=time.time(), status='FROZEN_BEFORE_THIS_START',
        source_archive_SHA256=source['archive_SHA256'], prior_control_SHA256=digest(prior),
        planned_independent_starts=3, independent_start=args.start, GPU_group=[4, 5, 6],
        model='Qwen2.5-VL-7B original BF16 BI0', main_manifest_SHA256=digest(resolve_manifest(None)),
        output_modes=['Natural EOS', 'Equal-output replay'], main_QPS=2,
        capacity_pilot_QPS=[4, 6] if args.start == 1 else ([.5, 1, 3] if args.start == 3 else []),
        profile_QPS=4 if args.start == 2 else None,
        fixed='Original E, all28 LLM layers, exact MRoPE, B8192/S1024, KV42GiB, block128, no prefix cache.',
        safety='Fresh per-consumer descriptors, synchronized consumer clone, completion file ACK, and D-complete release.',
        source_selection='C10 A/B/A Native publication control. Not a generic token-split result.',
        quality_gate='Full task-quality remains PENDING. Do not claim primary serving success from execution alone.'))
    names = []
    proxy = None
    results = []
    try:
        names, proxy = launch(tag, 'cuda_ipc')
        for port in [18841, 18842]:
            rpc(port, 'ec_ipc/ack_policy', dict(mode='file'))
        value = rpc(18840, 'ec_ipc/inline_policy', dict(enabled=True))
        assert len(value) == 1 and value[0]['inline_role'] == 'prefill'
        asyncio.run(correctness(tag, 'cuda_ipc'))
        asyncio.run(measure(tag, 'warmup_replay100', 2))
        for label, natural in [('natural100', True), ('replay100', False)]:
            results.append(asyncio.run(measure(tag, label, 2, natural)))
        capacity_qps = [4, 6] if args.start == 1 else ([.5, 1, 3] if args.start == 3 else [])
        if capacity_qps:
            for qps in capacity_qps:
                results.append(asyncio.run(measure(tag, 'capacity_replay100', qps)))
        if args.start == 2:
            profile = SimpleNamespace(manifest=resolve_manifest(None), replay_from=REFERENCE,
                server_names=','.join(names), order_seed=2026112404, qps=4, seed=2026112402,
                name=tag + '_CPC_q4_v0', url=URL,
                worker_urls='http://127.0.0.1:18840,http://127.0.0.1:18841,http://127.0.0.1:18842',
                observer_kind='profile', observe_client_host=False, freeze_client_static_heap=False)
            asyncio.run(profile_run(profile))
            ownership_empty(tag + '_profile', 'cuda_ipc')
        write(ROOT / 'analysis' / f'{tag}.json', dict(time=time.time(), status='COMPLETE',
            evidence='DIAGNOSTIC_C3', independent_start=args.start, rows=results,
            quality_gate='PENDING', primary_success=False))
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


if __name__ == '__main__':
    main()
