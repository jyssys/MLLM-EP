"""Launch frozen qualified workers under the shared quiet-lane lease."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import zipfile

from .common import (BASE, MODEL_REVISION, PYTHON, QUIET_LOCK, REPO, ROOT,
                     digest, git, initialize, preserve, resources, write)


def execute(name, restore=False):
    initialize()
    preserve()
    for path, expected in json.loads((ROOT / 'state/frozen_predecessor_hashes.json').read_text()).items():
        assert digest(REPO / path) == expected, path
    for path, expected in json.loads((ROOT / 'state/installed_runtime_sources.json').read_text())['hashes'].items():
        assert digest(path) == expected, path
    assert not (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    config_path = ROOT / 'configs' / (name + '.json')
    config = json.loads(config_path.read_text())
    required = ['configs/verified_ByteLevel_text_priming.json'] if config.get('verified_ByteLevel_priming') else []
    if config.get('canonical_new_KV_block_order'):
        required.append('state/canonical_block_runtime_contract.json')
    if config.get('gpu_phase_mask_norm'):
        required.append('analysis/phase_mask_norm_GPU_gate.json')
    if config.get('direct_phase_norm'):
        required.append('analysis/exact_phase_norm_GPU_gate.json')
    for path in required:
        assert (ROOT / path).is_file(), 'Missing dependency before GPU launch: ' + path
    folder = ROOT / 'raw' / name
    receipt = folder / 'dispatcher_result.json'
    if receipt.exists():
        return json.loads(receipt.read_text())['complete']
    revision = ROOT / 'state/user_primary_scope_revision_v2.json'
    if revision.exists() and config['arm'] == 'DP2':
        assert config['token_budget'] == 8192 and config['D_slots'] == 1024, \
            'The user stopped future non-primary Aggregated settings. Preserve old data without launching this configuration.'
    assert not folder.exists(), 'Preserve failed/incomplete runs; use a new name'
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if receipt.exists():
            return json.loads(receipt.read_text())['complete']
        if revision.exists() and config['arm'] == 'DP2':
            assert config['token_budget'] == 8192 and config['D_slots'] == 1024
        assert not folder.exists()
        data = json.loads((ROOT / 'data' / config['dataset']).read_text())
        assert len(data) >= config['count']
        assert all(len(row['input_ids']['qwen8']) + cap <= 40960
                   for row, cap in zip(data[:config['count']], config['request_caps']))
        folder.mkdir()
        write(folder / 'configuration.json', config)
        packages = ('duetserve_agg_decode', 'dense_ar_slo_layer_pd', 'dense_ar_elastic_layer_pd',
                    'dense_ar_dynamic_layer_pd', 'dense_ar_layerwise_pd',
                    'looped_pd_atomic_prefill', 'looped_pd_abc')
        files = [p for package in packages for p in (REPO / 'poc_flashvep' / package).rglob('*.py')
                 if '__pycache__' not in p.parts]
        archive = folder / 'source.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, str(path.relative_to(REPO)))
            bundle.write(REPO / 'poc_flashvep/duetserve_agg_decode/site_bootstrap.py', 'sitecustomize.py')
        frozen = folder / 'worker_source'
        frozen.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(frozen)
        write(folder / 'source_hashes.json', {str(frozen / p.relative_to(REPO)): digest(p) for p in files})
        write(folder / 'input_provenance.json', dict(model_revision=MODEL_REVISION,
            dataset_SHA256=digest(ROOT / 'data' / config['dataset']),
            SLO_SHA256=digest(ROOT / 'configs/frozen_SLO_regimes.json'),
            config_SHA256=digest(config_path), GPU_mapping=config['gpus']))
        env = dict(os.environ, PYTHONPATH=str(frozen), DENSE_FROZEN_ROOT=str(frozen),
            DENSE_CONFIG=str(config_path), DUETSERVE_EVIDENCE_BOOTSTRAP='1',
            DENSE_SOURCE_HASHES_FILE=str(folder / 'source_hashes.json'), VLLM_BATCH_INVARIANT='1',
            CUDA_VISIBLE_DEVICES=','.join(map(str, config['gpus'])),
            PATH=str(PYTHON.parent) + os.pathsep + os.environ.get('PATH', ''), OMP_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        owner = resources()
        owner.stop(list(range(8)))
        write(folder / 'isolation_before.json', dict(time=time.time(), processes=owner.processes(),
            host_load=os.getloadavg(), CPU_affinity=sorted(os.sched_getaffinity(0)),
            scope='All approved burners stopped under the shared lease. Unknown GPU owners are not terminated.'))
        module = config.get('entry_module', 'poc_flashvep.dense_ar_slo_layer_pd.native')
        command = [str(PYTHON), '-u', '-m', module, str(config_path)]
        child = None
        started = time.time()
        interrupted = [False]
        def stop(*_):
            interrupted[0] = True
        handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (ROOT / 'logs' / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=frozen, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'command.json', dict(time=started, command=command, pid=child.pid,
                    cwd=str(frozen), source_archive_SHA256=digest(archive), publication_parent=BASE,
                    worktree_HEAD=git('rev-parse', 'HEAD'), GPU_mapping=config['gpus']))
                write(ROOT / 'state/GPU_lease_current.json', dict(name=name, status='GPU_JOB_RUNNING',
                    pid=child.pid, start_epoch=started, GPUs=config['gpus'],
                    scope='Written only by the process that holds the shared quiet-lane lease. Other controllers can remain queued.'))
                print('START', name, child.pid, flush=True)
                term = None
                while child.poll() is None:
                    if interrupted[0] or time.time() > started + config.get('wall_timeout', 1800):
                        assert os.getsid(child.pid) == child.pid
                        if term is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            term = time.time()
                        elif time.time() - term > 20:
                            os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(0.5)
                state = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
                exits = json.loads((folder / 'worker_exits.json').read_text()) if (folder / 'worker_exits.json').exists() else {}
                observers = [json.loads(p.read_text()) for p in folder.glob('NVML-summary-*.json')]
                isolation = bool(observers) and all(v.get('isolation_passed', False) for v in observers)
                complete = child.returncode == 0 and state.get('status') == 'COMPLETE' and exits.get('passed') and isolation
                write(receipt, dict(time=time.time(), complete=bool(complete), returncode=child.returncode,
                    watchdog=term is not None, elapsed_seconds=time.time() - started,
                    normal_worker_exit=exits.get('passed', False), timing_isolation_passed=isolation))
                print('END', name, bool(complete), flush=True)
                return bool(complete)
        finally:
            if child is not None and child.poll() is None:
                assert os.getsid(child.pid) == child.pid
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if child is not None:
                remaining = [row for row in owner.processes() if os.getsid(row['pid']) == child.pid]
                if remaining:
                    write(folder / 'owned_orphan_cleanup.json', dict(time=time.time(), processes=remaining,
                        scope='Only remaining GPU processes in this recorded experiment session receive termination. Other sessions remain untouched.'))
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    until = time.monotonic() + 15
                    while any(os.getsid(row['pid']) == child.pid for row in owner.processes()) and time.monotonic() < until:
                        time.sleep(.5)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            write(ROOT / 'state/GPU_lease_current.json', dict(name=name, status='RELEASED',
                end_epoch=time.time(), GPUs=config['gpus']))
            preserve()
            if restore and not owner.processes():
                owner.restore(list(range(8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    try:
        for name in args.names:
            if not execute(name):
                break
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
