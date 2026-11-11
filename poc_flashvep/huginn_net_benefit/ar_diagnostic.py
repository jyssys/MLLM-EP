"""Reuse stock vLLM/NIXL AR diagnostics under the exclusive campaign owner."""
import argparse
import fcntl
import random
import shutil
import signal
import subprocess
from types import SimpleNamespace
import zipfile
from .common import *

AR_OLD = REPO / 'poc_flashvep/reports/looped_pd_atomic_prefill_qps_crossover'


def ar_environment(gpus):
    from ..looped_pd_abc.common import environment
    env = environment(gpus)
    for key in ('HEADROOM_RECOVERY_WORKER', 'FLUIDTOKEN_LOOP_WORKER', 'BC_WORKER',
            'ATOMIC_PD_WORKER', 'ATOMIC_NATIVE_COLOCATED'):
        env.pop(key, None)
    env.update(NET_BENEFIT_AR='1', VLLM_BATCH_INVARIANT='0',
        PYTHONPATH=str(SOURCE / 'ar_bootstrap')+os.pathsep+str(REPO))
    return env


def configure_ar():
    from ..looped_pd_atomic_prefill import common
    common.ROOT = ROOT
    common.environment = ar_environment
    # The imported stock worker binds ROOT and environment at import time.
    from ..looped_pd_atomic_prefill import ar_worker
    ar_worker.ROOT = ROOT
    ar_worker.environment = ar_environment


def run_stock_coordinator(cfg):
    """Observe real child exits without changing the stock execution path."""
    configure_ar()
    from ..looped_pd_atomic_prefill import ar_worker
    workers = {}

    def spawn(command, *args, **kwargs):
        role = command[command.index('--role')+1]
        assert role not in workers
        child = subprocess.Popen(command, *args, **kwargs)
        workers[role] = child
        return child

    module = ar_worker.subprocess
    ar_worker.subprocess = SimpleNamespace(Popen=spawn, STDOUT=subprocess.STDOUT,
                                          TimeoutExpired=subprocess.TimeoutExpired)
    try:
        ar_worker.coordinator(cfg)
    finally:
        ar_worker.subprocess = module
        exits = {role:dict(pid=child.pid, returncode=child.poll()) for role,child in workers.items()}
        passed = set(exits) == {'prefill','decode'} and all(
            item['returncode'] == 0 for item in exits.values())
        write(ROOT/'raw'/cfg['name']/'AR_worker_exits.json', dict(passed=passed, workers=exits))
    assert passed, 'The stock AR workers did not both exit successfully'


def complete(folder, cfg):
    status = folder/'status.json'
    exits = folder/'AR_worker_exits.json'
    requests = folder/'requests.jsonl'
    if not all(path.exists() for path in (status, exits, requests)):
        return False
    try:
        rows = [json.loads(line) for line in requests.read_text().splitlines()]
        return (json.loads(status.read_text())['status'] == 'COMPLETE'
                and json.loads(exits.read_text())['passed'] and len(rows) == cfg['count']
                and {row['sample_id'] for row in rows} == set(range(cfg['count']))
                and all(row['input_tokens'] == cfg['length'] and row['output_tokens'] == cfg['cap']
                        and len(row['output_ids']) == cfg['cap']
                        and not row.get('warmup', False) for row in rows))
    except (KeyError, ValueError, OSError, TypeError):
        return False


def prepare():
    initialize()
    dataset = 'test_qwen8_L8000.json'
    target = ROOT / 'data' / dataset
    if not target.exists():
        shutil.copy2(AR_OLD / 'data' / dataset, target)
    source = AR_OLD / 'configs/diagnostic_qwen8_async_pd_l8000_o200_first_v1.json'
    base = json.loads(source.read_text())
    names = []
    for qps in (2., 4., 6.):
        name = f'ar_native_qwen8_L8000_O200_qps{int(qps)}_seed2026110947'
        rng = random.Random(2026110947)
        offsets = [0.]
        for _ in range(31):
            offsets.append(offsets[-1]+rng.expovariate(qps))
        cfg = dict(base, name=name, model='qwen8', count=32, request_caps=[200]*32,
            prompt_ids=base.get('prompt_ids', [])[:32], arrival_offsets=offsets,
            arrival_rate=qps, arrival_seed=2026110947, dataset=dataset,
            seed=2026111011, phase='AR_DIAGNOSTIC_NOT_HUGINN_CONFIRMATION',
            P_slots=8, D_slots=32, gpus=[0, 1], chunk=8000, atomic_prompt=True,
            token_budget=32000, ar_async_scheduling=False, ar_batch_invariant=False,
            ar_enforce_eager=False, timeout=600, nsys=False, nsys_P_only=False,
            warmup_count=2, native_ar_reuse_source_sha256=digest(source),
            scientific_scope='Stock vLLM/NIXL 1P1D AR diagnosis. Not DuetServe or FluidPD implementation. Not Huginn evidence.')
        for key in ('ar_cudagraph_mode', 'ar_attention_config', 'ar_execution_family'):
            cfg.pop(key, None)
        save(cfg)
        names.append(name)
    write(ROOT / 'state/ar_diagnostic_jobs.json', names)
    print(' '.join(names))


def run(name):
    initialize()
    assert time.time() < LONG_START_CUTOFF, 'Do not start an AR cell inside the closure reserve'
    cfg = json.loads((ROOT / 'configs' / (name+'.json')).read_text())
    assert cfg['model'] == 'qwen8' and cfg['phase'] == 'AR_DIAGNOSTIC_NOT_HUGINN_CONFIRMATION'
    from ..looped_pd_autonomous_headroom.common import resources
    owner = resources()
    with (ROOT / 'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        folder = ROOT / 'raw' / name
        assert not folder.exists(), 'Preserve existing AR evidence'
        owner.stop(list(range(8)))
        assert not owner.processes(), 'Another GPU process occupies the diagnostic quiet lane'
        folder.mkdir()
        write(folder / 'configuration.json', cfg)
        write(folder / 'isolation.json', dict(time=time.time(), processes=owner.processes(),
            gpus=cfg['gpus'], scope='Exclusive all-GPU quiet lane. No concurrent Huginn or burner work.'))
        paths = sorted((REPO / 'poc_flashvep/looped_pd_atomic_prefill').glob('*.py'))
        paths += sorted(SOURCE.rglob('*.py'))
        with zipfile.ZipFile(folder / 'ar_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(REPO)))
        write(folder / 'source_hashes.json', {str(path): digest(path) for path in paths})
        command = ['/home/esjung/.venvs/looped-pd-abc/bin/python', '-m',
            'poc_flashvep.huginn_net_benefit.ar_diagnostic', '--worker', name]
        env = ar_environment(cfg['gpus'])
        env.update(ARC_CONFIG=str(ROOT / 'configs' / (name+'.json')))
        start = time.monotonic()
        with (ROOT / 'logs' / (name+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            write(ROOT / 'state/active.json', dict(name=name, pid=child.pid, gpus=cfg['gpus'], time=time.time()))
            write(folder / 'command.json', dict(command=command, pid=child.pid,
                configuration_sha256=digest(env['ARC_CONFIG']), time=time.time()))
            try:
                child.wait(timeout=cfg['timeout']+660)
            except BaseException:
                assert os.getsid(child.pid) == child.pid
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                write(folder / 'dispatcher_result.json', dict(returncode=child.returncode,
                    complete=child.returncode == 0 and complete(folder, cfg),
                    elapsed_seconds=time.monotonic()-start, time=time.time()))
                write(ROOT / 'state/active.json', dict(status='IDLE', time=time.time()))
                if not owner.processes():
                    owner.restore(list(range(8)))
        assert child.returncode == 0 and complete(folder, cfg), 'AR diagnostic failed. Preserve its log.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        run_stock_coordinator(json.loads((ROOT / 'configs' / (args.worker+'.json')).read_text()))
    else:
        run(args.name)


if __name__ == '__main__':
    main()
