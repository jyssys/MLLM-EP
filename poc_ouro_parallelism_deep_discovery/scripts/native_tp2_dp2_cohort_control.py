"""Guarded conventional TP2xDP2 control. No native model operator changes."""
import argparse
import copy
from multiprocessing import Pipe
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import traceback

from poc_ouro_parallelism_deep_discovery.contract import GPUS, MODELS, PROJECT, ROOT, read, sha, validate_mask, write
from poc_ouro_parallelism_deep_discovery.dp4_cohort import collect_messages
from poc_ouro_parallelism_deep_discovery.scripts.native_dp4_cohort_control import cleanup
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import (
    GROUPS, assignments, inspect_process_leases, run_cohort, validate_groups, validate_ownership)


def frontend_environment(specification):
    """Preserve proc-environment guards before native process-title setup."""
    environment = dict(os.environ, **specification['cache_environment'],
        CUDA_VISIBLE_DEVICES=','.join(map(str, specification['physical_gpus'])),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        SPT_NOENV='1', VLLM_DP_SIZE='1', VLLM_DP_RANK='0', VLLM_DP_RANK_LOCAL='0',
        VLLM_DP_MASTER_PORT=str(29740 + specification['worker']))
    for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        environment.pop(key, None)
    return environment


def freeze_plan(size, protocol_path, output):
    protocol = read(protocol_path)
    assert protocol['status'] == 'FROZEN_BEFORE_RESULTS'
    base_path = ROOT / protocol['base_protocol']
    assert sha(base_path) == protocol['base_protocol_sha256']
    base = read(base_path)
    assert base['status'] == 'FROZEN_BEFORE_RESULTS'
    assert base['cohort_sizes'] == [1, 2, 4, 8] and base['contexts'] == [1024, 4096]
    assert base['generated_tokens'] == 128 and base['warmup_repeats'] == 3 and base['repeats'] == 5
    assert base['async_scheduling'] and not base['chunked_prefill'] and not base['prefix_caching']
    assert base['kv_cache_memory_bytes_per_rank'] == protocol['kv_cache_memory_bytes_per_rank'] == 52 * 1024 ** 3
    assert protocol['physical_gpus'] == GPUS
    validate_groups(protocol['physical_groups'])
    assert protocol['replicas'] == protocol['tensor_parallel_size_per_replica'] == 2
    assert protocol['native_data_parallel_size_per_engine'] == 1
    assert protocol['models'][size]['revision'] == MODELS[size]['revision']
    short = size.replace('.', '').replace('B', '')
    sources = {tp: ROOT / f'outputs/control_async_base{short}_P{tp}_v1/{size}_R4_TP{tp}' for tp in [1, 2]}
    inputs1, inputs2 = (read(sources[tp] / 'workload.json')['inputs'] for tp in [1, 2])
    assert inputs1 == inputs2, 'TP1 and TP2 frozen workloads differ'
    templates = {tp: read(sources[tp] / 'engine_config.json') for tp in [1, 2]}
    assert dict(templates[1], tensor_parallel_size=2) == templates[2], 'Unexpected TP template difference'
    template = templates[2]
    assert Path(template['model']).resolve() == MODELS[size]['path'].resolve()
    assert template['tensor_parallel_size'] == 2 and template['dtype'] == 'bfloat16'
    assert template['distributed_executor_backend'] == 'mp'
    assert template['hf_overrides']['total_ut_steps'] == 4 and template['async_scheduling']
    assert not template['enable_chunked_prefill'] and not template['enable_prefix_caching']
    assert not template['enable_expert_parallel'] and not template['enforce_eager']
    assert not template.get('speculative_config') and template.get('pipeline_parallel_size', 1) == 1
    token = inputs2[0]['teacher_ids'][0]
    assert 0 < token < 49152 and all(row['teacher_ids'][0] == token for row in inputs2)
    assert read(MODELS[size]['path'] / 'config.json')['eos_token_id'] == 0
    cases = []
    for count in base['cohort_sizes']:
        for context in base['contexts']:
            case = dict(C=count, context=context, requests=[dict(cohort_slot=index,
                source_request_id=inputs2[index % len(inputs2)]['id'],
                prompt_token_ids=inputs2[index % len(inputs2)]['prompt_ids'][str(context)]) for index in range(count)])
            assignments(case)
            cases.append(case)
    workers = []
    for worker, group in enumerate(GROUPS):
        directory = output / f'workers/replica{worker}_gpus{group[0]}_{group[1]}'
        environment = {name: str(directory / suffix) for name, suffix in (
            ('VLLM_CACHE_ROOT', 'vllm_cache'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
            ('TRITON_CACHE_DIR', 'triton_cache'), ('TORCH_EXTENSIONS_DIR', 'torch_extensions'))}
        config = copy.deepcopy(template)
        config.update(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
            max_num_seqs=base['max_num_seqs'], max_num_batched_tokens=base['max_num_batched_tokens'],
            kv_cache_memory_bytes=base['kv_cache_memory_bytes_per_rank'],
            worker_extension_cls='poc_ouro_parallelism_deep_discovery.hybrid_cohort_probe.HybridCohortProbe')
        config['compilation_config']['cache_dir'] = str(directory / 'compile_cache')
        workers.append(dict(worker=worker, physical_gpus=list(group), output=str(directory),
                            cache_environment=environment, engine_config=config))
    implementation = ['tp2_dp2_cohort.py', 'scripts/native_tp2_dp2_cohort_control.py',
        'scripts/native_tp2_dp2_cohort_worker.py', 'dp4_cohort.py', 'scripts/native_dp4_cohort_control.py',
        'cohort_probe.py', 'runtime.py', 'contract.py', 'process_lease_diagnostics.py', 'hybrid_cohort_probe.py']
    plan = dict(status='FROZEN_BEFORE_WORKER_START', size=size, R=4, TP=2, DP=2,
        parent_pgid=os.getpgid(0), parent_sid=os.getsid(0), guard_output=str(output),
        tp2_dp2_protocol=protocol, tp2_dp2_protocol_sha256=sha(protocol_path),
        native_protocol=base, native_protocol_sha256=sha(base_path),
        model={key: str(value) if isinstance(value, Path) else value for key, value in MODELS[size].items()},
        source_hashes={str(source / name): sha(source / name) for source in sources.values()
                       for name in ['engine_config.json', 'workload.json']},
        exact_tp1_tp2_inputs_equal=True, exact_templates_equal_except_tp=True,
        process_title_observability=dict(SPT_NOENV='1', scope='Frontend exec environment and descendants',
            strict_process_lease_checks='UNCHANGED', model_operators='UNCHANGED'),
        implementation_sha256={name: sha(ROOT / name) for name in implementation},
        fixed_token_id=token, cases=cases, workers=workers)
    for specification in workers:
        directory = Path(specification['output'])
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / 'engine_config.json', specification['engine_config'])
    write(output / 'tp2_dp2_plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'artifacts/native_tp2_dp2_cohort_protocol_v1.json')
    args = parser.parse_args()
    assert validate_mask() == GPUS
    output = Path(os.environ['LOOPED_KV_OUTPUT']).resolve()
    assert output.is_relative_to(ROOT / 'outputs')
    guard = read(output / 'run_contract.json')
    assert guard['mask'] == ','.join(map(str, GPUS))
    assert os.getpgid(0) == os.getsid(0) == os.getpid()
    assert 'poc_ouro_parallelism_deep_discovery.scripts.native_tp2_dp2_cohort_control' in guard['command']
    assert not os.environ.get('OURO_CAMPAIGN_STORAGE_CONTROL') and os.environ.get('POC4_FAMILY') != 'tied'
    assert not (output / 'tp2_dp2_plan.json').exists()
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
            environment = frontend_environment(specification)
            log = (Path(specification['output']) / 'console.log').open('x')
            logs.append(log)
            command = [sys.executable, '-m', 'poc_ouro_parallelism_deep_discovery.scripts.native_tp2_dp2_cohort_worker',
                '--plan', str(output / 'tp2_dp2_plan.json'), '--worker', str(worker),
                '--control-fd', str(child_end.fileno())]
            try:
                processes[worker] = subprocess.Popen(command, cwd=PROJECT, env=environment,
                    pass_fds=(child_end.fileno(),), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            finally:
                child_end.close()
        timeout = plan['tp2_dp2_protocol']['trial_timeout_seconds']
        loaded = collect_messages(connections, 'LOADED', plan['tp2_dp2_protocol']['startup_timeout_seconds'], transcript=transcript)
        for worker, message in loaded.items():
            specification = plan['workers'][worker]
            assert message['pid'] == processes[worker].pid
            assert message['physical_gpus'] == specification['physical_gpus']
            validate_ownership(specification['physical_gpus'], MODELS[args.size]['layers'], message['ownership'])
            inspect_process_leases([message['pid']] + [row['pid'] for row in message['ownership']],
                plan['parent_pgid'], plan['parent_sid'], plan['guard_output'], ancestor_pid=message['pid'])
        write(output / 'loaded_workers.json', loaded)
        trial, warmups = 0, []
        for index, case in enumerate(plan['cases']):
            for repeat in range(plan['native_protocol']['warmup_repeats']):
                warmups.append(run_cohort(connections, case, index, 16, 'WARMUP', trial, timeout, transcript))
                trial += 1
        write(output / 'warmups.json', warmups)
        status['phase'] = 'CLEAN'
        write(output / 'status.json', status)
        rng, order = random.Random(plan['native_protocol']['order_seed']), []
        for repeat in range(plan['native_protocol']['repeats']):
            indices = list(range(len(plan['cases'])))
            rng.shuffle(indices)
            for index in indices:
                case = plan['cases'][index]
                value = run_cohort(connections, case, index, 128, 'CLEAN', trial, timeout, transcript)
                trial += 1
                record = dict(value, size=args.size, R=4, TP=2, DP=2, physical_gpus=GPUS,
                    physical_groups=[list(group) for group in GROUPS], C=case['C'], context=case['context'],
                    repeat=repeat, evidence_class='LIVE_CLOSED_COHORT', quality='NOT_RUN',
                    placement='Static cohort_slot modulo 2. Two native TP2 engines. No migration.')
                write(output / f"C{case['C']}_N{case['context']}_repeat{repeat}.json", record)
                order.append(dict(trial_id=value['trial_id'], C=case['C'], context=case['context'], repeat=repeat))
                print('NATIVE_TP2_DP2_COHORT', args.size, case['C'], case['context'], repeat, flush=True)
        write(output / 'trial_order.json', order)
        status['phase'] = 'TRACE'
        write(output / 'status.json', status)
        traces = []
        for index, case in enumerate(plan['cases']):
            value = run_cohort(connections, case, index, 128, 'TRACE', trial, timeout, transcript)
            trial += 1
            traces.append(dict(value, C=case['C'], context=case['context'], timing='EXCLUDED_FROM_HEADLINE. Separate trace.'))
        write(output / 'schedule_trace.json', traces)
        for connection in connections.values():
            connection.send(dict(kind='STOP'))
        collect_messages(connections, 'STOPPED', plan['tp2_dp2_protocol']['shutdown_timeout_seconds'], transcript=transcript)
        status.update(success=True, phase='COMPLETE')
    except BaseException as error:
        status.update(success=False, phase='FAILED', error_type=type(error).__name__)
        write(output / 'failure.json', dict(error_type=type(error).__name__, traceback=traceback.format_exc()))
    finally:
        stopped = cleanup(processes, connections, 60 if plan is None else plan['tp2_dp2_protocol']['shutdown_timeout_seconds'])
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
