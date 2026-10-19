"""Guarded fixed-R4 DP4 cohort control. No native model operator is replaced."""
import argparse
import copy
from multiprocessing import Pipe
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

from poc_ouro_parallelism_deep_discovery.contract import GPUS, MODELS, PROJECT, ROOT, read, sha, validate_mask, write
from poc_ouro_parallelism_deep_discovery.dp4_cohort import assignments, collect_messages, run_cohort


def freeze_plan(size, protocol_path, output):
    protocol = read(protocol_path)
    assert protocol['status'] == 'FROZEN_BEFORE_RESULTS'
    base_path = ROOT / protocol['base_protocol']
    assert sha(base_path) == protocol['base_protocol_sha256']
    base = read(base_path)
    assert base['status'] == 'FROZEN_BEFORE_RESULTS'
    assert base['cohort_sizes'] == [1, 2, 4, 8] and base['contexts'] == [1024, 4096]
    assert base['generated_tokens'] == 128 and base['async_scheduling']
    assert not base['chunked_prefill'] and not base['prefix_caching']
    assert protocol['physical_gpus'] == GPUS and protocol['replicas'] == 4
    assert protocol['models'][size]['revision'] == MODELS[size]['revision']
    short = size.replace('.', '').replace('B', '')
    source = ROOT / f'outputs/control_async_base{short}_P1_v1/{size}_R4_TP1'
    inputs = read(source / 'workload.json')['inputs']
    token = inputs[0]['teacher_ids'][0]
    assert token != 0 and 0 <= token < 49152
    model_config = read(MODELS[size]['path'] / 'config.json')
    assert model_config['eos_token_id'] == 0
    template = read(source / 'engine_config.json')
    assert Path(template['model']).resolve() == MODELS[size]['path'].resolve()
    assert template['tensor_parallel_size'] == 1 and template['dtype'] == 'bfloat16'
    assert template['hf_overrides']['total_ut_steps'] == 4 and template['async_scheduling']
    assert not template['enable_chunked_prefill'] and not template['enable_prefix_caching']
    assert not template['enable_expert_parallel'] and not template['enforce_eager']
    assert not template.get('speculative_config') and template.get('pipeline_parallel_size', 1) == 1
    cases = []
    for count in base['cohort_sizes']:
        for context in base['contexts']:
            case = dict(C=count, context=context, requests=[dict(cohort_slot=index,
                source_request_id=inputs[index % len(inputs)]['id'],
                prompt_token_ids=inputs[index % len(inputs)]['prompt_ids'][str(context)]) for index in range(count)])
            assignments(case)
            cases.append(case)
    workers = []
    for worker, gpu in enumerate(GPUS):
        directory = output / f'workers/replica{worker}_gpu{gpu}'
        environment = {name: str(directory / suffix) for name, suffix in (
            ('VLLM_CACHE_ROOT', 'vllm_cache'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
            ('TRITON_CACHE_DIR', 'triton_cache'), ('TORCH_EXTENSIONS_DIR', 'torch_extensions'))}
        config = copy.deepcopy(template)
        config.update(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            max_num_seqs=base['max_num_seqs'], max_num_batched_tokens=base['max_num_batched_tokens'],
            kv_cache_memory_bytes=base['kv_cache_memory_bytes_per_rank'],
            worker_extension_cls='poc_ouro_parallelism_deep_discovery.cohort_probe.CohortProbe')
        config['compilation_config']['cache_dir'] = str(directory / 'compile_cache')
        workers.append(dict(worker=worker, physical_gpu=gpu, output=str(directory),
                            cache_environment=environment, engine_config=config))
    implementation = ['dp4_cohort.py', 'scripts/native_dp4_cohort_control.py',
                      'scripts/native_dp4_cohort_worker.py', 'cohort_probe.py', 'runtime.py', 'contract.py']
    plan = dict(status='FROZEN_BEFORE_WORKER_START', size=size, R=4, TP=1, DP=4,
        parent_pgid=os.getpgid(0), guard_output=str(output),
        dp4_protocol=protocol, dp4_protocol_sha256=sha(protocol_path),
        native_protocol=base, native_protocol_sha256=sha(base_path),
        model={key: str(value) if isinstance(value, Path) else value for key, value in MODELS[size].items()},
        source_hashes={str(path): sha(path) for path in (source / 'engine_config.json', source / 'workload.json')},
        implementation_sha256={name: sha(ROOT / name) for name in implementation},
        fixed_token_id=token, cases=cases, workers=workers)
    for worker in workers:
        directory = Path(worker['output'])
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / 'engine_config.json', worker['engine_config'])
    write(output / 'dp4_plan.json', plan)
    return plan


def cleanup(processes, connections, timeout):
    """Stop only explicitly created child PIDs. The outer guard owns their session."""
    for connection in connections.values():
        try:
            connection.send(dict(kind='STOP'))
        except (OSError, EOFError):
            pass
    deadline = time.monotonic() + timeout
    records = []
    for worker, process in processes.items():
        forced = False
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            forced = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        records.append(dict(worker=worker, pid=process.pid, exit_code=process.returncode, forced=forced))
    for connection in connections.values():
        connection.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'artifacts/native_dp4_cohort_protocol_v1.json')
    args = parser.parse_args()
    assert validate_mask() == GPUS
    output = Path(os.environ['LOOPED_KV_OUTPUT']).resolve()
    assert output.is_relative_to(ROOT / 'outputs')
    guard = read(output / 'run_contract.json')
    assert guard['mask'] == ','.join(map(str, GPUS)) and os.getpgid(0) == os.getpid()
    assert 'poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_control' in guard['command']
    assert not os.environ.get('OURO_CAMPAIGN_STORAGE_CONTROL')
    assert os.environ.get('POC4_FAMILY') != 'tied'
    assert not (output / 'dp4_plan.json').exists()
    plan = None
    processes, connections, logs, transcript = {}, {}, [], []
    status = dict(success=False, phase='STARTING', quality='NOT_RUN', candidate_speedup='NOT_RUN')
    write(output / 'status.json', status)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        plan = freeze_plan(args.size, args.protocol, output)
        for specification in plan['workers']:
            worker = specification['worker']
            parent_end, child_end = Pipe(duplex=True)
            connections[worker] = parent_end
            environment = dict(os.environ, **specification['cache_environment'],
                CUDA_VISIBLE_DEVICES=str(specification['physical_gpu']),
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False',
                VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_DP_SIZE='1', VLLM_DP_RANK='0',
                VLLM_DP_RANK_LOCAL='0', VLLM_DP_MASTER_PORT=str(29640 + worker))
            log = (Path(specification['output']) / 'console.log').open('x')
            logs.append(log)
            command = [sys.executable, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_worker',
                       '--plan', str(output / 'dp4_plan.json'), '--worker', str(worker),
                       '--control-fd', str(child_end.fileno())]
            try:
                processes[worker] = subprocess.Popen(command, cwd=PROJECT, env=environment,
                    pass_fds=(child_end.fileno(),), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            finally:
                child_end.close()
        loaded = collect_messages(connections, 'LOADED', plan['dp4_protocol']['startup_timeout_seconds'],
                                  transcript=transcript)
        write(output / 'loaded_workers.json', loaded)
        trial = 0
        timeout = plan['dp4_protocol']['trial_timeout_seconds']
        warmups = []
        for index, case in enumerate(plan['cases']):
            for repeat in range(plan['native_protocol']['warmup_repeats']):
                warmups.append(run_cohort(connections, case, index, 16, 'WARMUP', trial, timeout, transcript))
                trial += 1
        write(output / 'warmups.json', warmups)
        status['phase'] = 'CLEAN'
        write(output / 'status.json', status)
        rng = random.Random(plan['native_protocol']['order_seed'])
        order = []
        for repeat in range(plan['native_protocol']['repeats']):
            indices = list(range(len(plan['cases'])))
            rng.shuffle(indices)
            for index in indices:
                case = plan['cases'][index]
                value = run_cohort(connections, case, index, 128, 'CLEAN', trial, timeout, transcript)
                trial += 1
                record = dict(value, size=args.size, R=4, TP=1, DP=4, physical_gpus=GPUS,
                    C=case['C'], context=case['context'], repeat=repeat, evidence_class='LIVE_CLOSED_COHORT',
                    quality='NOT_RUN', placement='Static cohort_slot modulo 4. No migration.')
                write(output / f"C{case['C']}_N{case['context']}_repeat{repeat}.json", record)
                order.append(dict(trial_id=value['trial_id'], C=case['C'], context=case['context'], repeat=repeat))
                print('NATIVE_DP4_COHORT', args.size, case['C'], case['context'], repeat, flush=True)
        write(output / 'trial_order.json', order)
        status['phase'] = 'TRACE'
        write(output / 'status.json', status)
        traces = []
        for index, case in enumerate(plan['cases']):
            value = run_cohort(connections, case, index, 128, 'TRACE', trial, timeout, transcript)
            trial += 1
            traces.append(dict(value, C=case['C'], context=case['context'],
                               timing='EXCLUDED_FROM_HEADLINE. Separate schedule trace.'))
        write(output / 'schedule_trace.json', traces)
        for connection in connections.values():
            connection.send(dict(kind='STOP'))
        collect_messages(connections, 'STOPPED', plan['dp4_protocol']['shutdown_timeout_seconds'], transcript=transcript)
        status.update(success=True, phase='COMPLETE')
    except BaseException as error:
        status.update(success=False, phase='FAILED', error_type=type(error).__name__)
        write(output / 'failure.json', dict(error_type=type(error).__name__, traceback=traceback.format_exc()))
    finally:
        shutdown_timeout = 60 if plan is None else plan['dp4_protocol']['shutdown_timeout_seconds']
        stopped = cleanup(processes, connections, shutdown_timeout)
        if any(row['forced'] or row['exit_code'] != 0 for row in stopped):
            status.update(success=False, phase='CLEANUP_FAILED')
        write(output / 'worker_cleanup.json', stopped)
        write(output / 'control_transcript.json', transcript)
        write(output / 'status.json', status)
        for log in logs:
            log.close()
    if not status['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
