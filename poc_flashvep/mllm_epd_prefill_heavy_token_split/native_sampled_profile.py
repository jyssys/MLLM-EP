"""One-factor observer control after the independent Native performance starts."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .common import ROOT, digest, preserve, write, experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import launch, stop_proxy, correctness, ownership_empty
from .llm_probe_control import rpc
from .native_inline_confirmation import measure, REFERENCE, URL
from .profile_control import run as profile_run
from .source_snapshot import snapshot
from .client import resolve_manifest

TAG = 'ec_PROFILE_C12'


def wait_for_prior():
    deadline = min(time.time() + 1800, experiment_deadline())
    while True:
        leases = [json.loads((ROOT / 'state' / f'server_ec_NATIVE_C11_S3_{role}{gpu}.json').read_text())
                  for role, gpu in [('E', 4), ('P', 5), ('D', 6)]]
        if all(row['finished'] and not Path('/proc', str(row['guardian'])).exists() for row in leases):
            assert json.loads((ROOT / 'analysis/ec_NATIVE_C11_S3.json').read_text())['status'] == 'COMPLETE'
            return
        for row in leases:
            proc = Path('/proc', str(row['guardian']))
            if proc.exists():
                assert proc.stat().st_uid == os.getuid()
                assert b'mllm_epd_prefill_heavy_token_split.servers' in (proc / 'cmdline').read_bytes()
        if time.time() >= deadline:
            raise TimeoutError('Inspect the same prior leases. Do not create overlapping server epochs.')
        time.sleep(1)


def main():
    global TAG
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default=TAG)
    TAG=parser.parse_args().tag
    preserve()
    plan = ROOT / 'configs' / f'{TAG}.json'
    assert not plan.exists(), 'Do not repeat an existing observer identity'
    source = snapshot(TAG)
    write(plan, dict(time=time.time(), status='FROZEN_BEFORE_MEASUREMENT',
        source_archive_SHA256=source['archive_SHA256'], QPS=4, requests=100, state_interval_ms=50,
        prior_profile='ec_NATIVE_C11_S2_CPC_q4_v0',
        prior_failure='Profile TTFTp50 +11.416%, ITLp99 +5.186% relative to surrounding clean mean.',
        ranked_hypotheses=['Per-step Python state collection', 'CUPTI activity recording', 'Temporal execution variation'],
        one_factor='Sample scheduler/execute host state at most once per50ms. Keep every GPU kernel/copy record.',
        unchanged=['Original model', 'Dataset', 'Arrival', 'Output replay', 'EC transport', 'SLO', 'GPU group'],
        observer_gate_max=.05, no_candidate_offloading=True,
        note='State samples do not capture every transient batch or instantaneous queue maximum.'))
    wait_for_prior()
    names = []
    proxy = None
    try:
        names, proxy = launch(TAG, 'cuda_ipc')
        for port in [18841, 18842]:
            rpc(port, 'ec_ipc/ack_policy', dict(mode='file'))
        rpc(18840, 'ec_ipc/inline_policy', dict(enabled=True))
        for name in names:
            lease = json.loads((ROOT / 'state' / f'server_{name}.json').read_text())
            assert Path('/proc', str(lease['pid']), 'cwd').resolve() == Path(lease['executed_source_cwd'])
        asyncio.run(correctness(TAG, 'cuda_ipc'))
        asyncio.run(measure(TAG, 'warmup_replay100', 4))
        args = SimpleNamespace(manifest=resolve_manifest(None), replay_from=REFERENCE,
            server_names=','.join(names), order_seed=2026112404, qps=4, seed=2026112402,
            name=TAG + '_CPC_q4_v0', url=URL,
            worker_urls='http://127.0.0.1:18840,http://127.0.0.1:18841,http://127.0.0.1:18842',
            observer_kind='profile', state_interval_ms=50,
            observe_client_host=False, freeze_client_static_heap=False)
        asyncio.run(profile_run(args))
        ownership_empty(TAG, 'cuda_ipc')
        write(ROOT / 'state' / f'{TAG}_complete.json', dict(time=time.time(), status='COMPLETE',
            qualification_SHA256=digest(ROOT / 'analysis' / f'{args.name}_qualification.json')))
    finally:
        if proxy is not None:
            stop_proxy(proxy)
        if names:
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__ == '__main__':
    main()
