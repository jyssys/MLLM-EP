"""Guarded C07 eager physical fixture. This is not closed-loop speculation."""
import argparse
import copy
import os
from pathlib import Path
import time
import traceback

from poc_ouro_parallelism_deep_discovery.contract import (
    ROOT, GPUS, UUIDS, MODELS, NATIVE_PYTHON, read, sha, validate_mask, write)
from poc_ouro_parallelism_deep_discovery.c07_fixture import sealed_reference_cohort, selectors
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--tp', type=int, choices=[1, 4], required=True)
    parser.add_argument('--context', type=int, choices=[1024, 4096], default=1024)
    parser.add_argument('--row', type=int, choices=range(4), default=0)
    parser.add_argument('--depth', type=int, choices=range(4), default=0)
    parser.add_argument('--repeats', type=int, default=5)
    return parser.parse_args(argv)


def engine_config(source, output, size, tp):
    config = copy.deepcopy(source)
    if (Path(config['model']).resolve() != MODELS[size]['path'].resolve()
            or config['tensor_parallel_size'] != tp
            or config['hf_overrides']['total_ut_steps'] != 4
            or config['enable_chunked_prefill'] or config['enable_prefix_caching']
            or config['enable_expert_parallel'] or not config['async_scheduling']):
        raise ValueError('The source engine configuration differs from the pinned native R4 protocol.')
    config.update(enforce_eager=True, max_num_seqs=4, max_num_batched_tokens=32768,
                  kv_cache_memory_bytes=32 * 1024 ** 3,
                  worker_extension_cls='poc_ouro_parallelism_deep_discovery.c07_fixture_probe.C07FixtureProbe')
    config['compilation_config'] = dict(mode=0, cudagraph_mode='NONE', cache_dir=str(output / 'compile_cache'))
    return config


def freeze_request(workload, context):
    inputs = workload['inputs']
    if not inputs:
        raise ValueError('The frozen workload is empty.')
    token = inputs[0]['teacher_ids'][0]
    if not isinstance(token, int) or not 0 < token < 49152:
        raise ValueError('The allowed output token must be a frozen non-EOS token.')
    requests = []
    for row in range(4):
        source = inputs[row % len(inputs)]
        prompt = list(source['prompt_ids'][str(context)])
        if len(prompt) != context or not all(isinstance(item, int) and 0 <= item < 49152 for item in prompt):
            raise ValueError('A frozen prompt differs from the required token layout.')
        requests.append(dict(submitted_row=row, source_request_id=source['id'], prompt_token_ids=prompt))
    return dict(context=context, output_length=2, allowed_token_id=token, requests=requests,
                identity='Four distinct native requests. Prompt contents can repeat.')


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.repeats <= 32:
        raise ValueError('The eager service requires one through32 repeats.')
    if args.tp == 4 and (args.row != 0 or args.depth != 0):
        raise ValueError('TP4 always selects distinct rows/depths0/1/2/3.')
    if os.environ.get('OURO_CAMPAIGN_STORAGE_CONTROL') or os.environ.get('POC4_FAMILY') == 'tied':
        raise RuntimeError('Storage and tied-tape interventions are not permitted in this fixture.')
    root = Path(os.environ['LOOPED_KV_OUTPUT']).resolve()
    if not root.is_relative_to(ROOT / 'outputs') or not (root / 'run_contract.json').exists():
        raise RuntimeError('Run the fixture through the campaign gpu_session guard.')
    if os.getpgid(0) != os.getpid():
        raise RuntimeError('The fixture runner must own the guarded process group.')
    output = root / f'{args.size}_C07_TP{args.tp}'
    output.mkdir(parents=True, exist_ok=False)
    status = dict(success=False, phase='SETUP', parity='NOT_RUN', quality='NOT_RUN',
                  timing_scope='Eager physical pass only. No strongest-native or speculation claim.')
    write(output / 'status.json', status)
    llm, armed = None, False
    try:
        mask = validate_mask()
        if len(mask) != args.tp:
            raise ValueError('The guard mask does not match TP.')
        short = args.size.replace('.', '').replace('B', '')
        source = ROOT / f'outputs/control_async_base{short}_P{args.tp}_v1/{args.size}_R4_TP{args.tp}'
        request = freeze_request(read(source / 'workload.json'), args.context)
        config = engine_config(read(source / 'engine_config.json'), output, args.size, args.tp)
        roots = {key: str(output / directory) for key, directory in [
            ('VLLM_CACHE_ROOT', 'vllm_cache'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor_cache'),
            ('TRITON_CACHE_DIR', 'triton_cache')]}
        for key, directory in roots.items():
            if Path(directory).exists():
                raise FileExistsError('A fixture cache root already exists.')
            os.environ[key] = directory
        os.environ['C07_FIXTURE_OUTPUT'] = str(output)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
        write(output / 'cache_environment.json', dict(before_vllm_import=True, variables=roots,
              compile_cache=config['compilation_config']['cache_dir'], enforce_eager=True, compilation_mode=0))
        write(output / 'engine_config.json', config)
        write(output / 'request.json', request)
        native_root = Path(NATIVE_PYTHON).parents[1] / 'lib/python3.14/site-packages'
        native_sources = {str(native_root / relative): sha(native_root / relative) for relative in [
            'vllm/model_executor/models/ouro.py', 'vllm/forward_context.py',
            'vllm/v1/attention/backends/flash_attn.py', 'vllm/distributed/parallel_state.py',
            'vllm/v1/worker/gpu_model_runner.py', 'vllm/v1/worker/gpu/model_runner.py',
            'vllm/v1/worker/gpu/input_batch.py', 'vllm/v1/engine/core.py',
            'vllm/v1/engine/core_client.py', 'vllm/v1/core/sched/scheduler.py',
            'vllm/entrypoints/llm.py', 'vllm/entrypoints/offline_utils.py']}
        write(output / 'pre_run_contract.json', dict(status='FROZEN_BEFORE_ENGINE_LOAD',
            model={key: str(value) if isinstance(value, Path) else value for key, value in MODELS[args.size].items()},
            tp=args.tp, physical_gpus=mask, selected=selectors(args.tp, args.row, args.depth),
            row_order='Captured native input_batch request order. Submitted order is not assumed.',
            source_config=str(source / 'engine_config.json'), source_config_sha256=sha(source / 'engine_config.json'),
            source_workload_sha256=sha(source / 'workload.json'),
            runner_sha256=sha(Path(__file__)), worker_sha256=sha(ROOT / 'c07_fixture_probe.py'),
            helper_sha256=sha(ROOT / 'c07_fixture.py'), repeats=args.repeats,
            source_cohort='Native pause_scheduler(keep,False), enqueue four, native resume, wait_for_completion. Untimed fixture construction.',
            native_sources_sha256=native_sources,
            parity='Bitwise BF16 output and current/history KV words. Any difference is UNRESOLVED.',
            lease='No serving calls after generation. Disjoint non-null pages in existing slot0 allocations.',
            exclusions=['No scheduler', 'No branch verification', 'No graph capture', 'No quality claim']))
        configure_native_imports()
        from vllm import LLM, SamplingParams
        status['phase'] = 'ENGINE_LOAD'
        write(output / 'status.json', status)
        llm = LLM(**config)
        ownership = llm.collective_rpc('inspect_ownership', timeout=180)
        write(output / 'ownership.json', ownership)
        if (len(ownership) != args.tp or {row['GPU_uuid'].removeprefix('GPU-') for row in ownership}
                != {UUIDS[GPUS.index(gpu)] for gpu in mask}
                or not all(row['cache_slots'] == 4 * MODELS[args.size]['layers'] for row in ownership)):
            raise RuntimeError('The native GPU or4L-cache ownership differs from the frozen contract.')
        write(output / 'capture_start.json', llm.collective_rpc('c07_capture_start', timeout=180,
              args=(args.context, args.row, args.depth)))
        armed = True
        status['phase'] = 'CAPTURE_REQUEST'
        write(output / 'status.json', status)
        start = time.perf_counter_ns()
        seal = {}
        try:
            results = sealed_reference_cohort(llm,
                [dict(prompt_token_ids=row['prompt_token_ids']) for row in request['requests']],
                SamplingParams(temperature=0, max_tokens=2, ignore_eos=True,
                               allowed_token_ids=[request['allowed_token_id']]), seal)
        finally:
            write(output / 'source_cohort_seal.json', seal)
        capture_wall = time.perf_counter_ns() - start
        if (len(results) != 4 or any(list(row.outputs[0].token_ids) != [request['allowed_token_id']] * 2
                                    for row in results) or llm.llm_engine.has_unfinished_requests()):
            raise RuntimeError('The native C4 request did not finish with the frozen output tokens.')
        write(output / 'capture_request.json', dict(wall_ns=capture_wall,
            scope='Untimed fixture source. Includes native prefill, decode, copies, sampling, and IPC.',
            results=[dict(request_id=row.request_id, output_ids=list(row.outputs[0].token_ids)) for row in results]))
        captures = llm.collective_rpc('c07_capture_finish', timeout=180, args=(True,))
        armed = False
        write(output / 'capture_finish.json', captures)
        if len(captures) != args.tp or not all(row['phase'] == 'QUIESCENT_CAPTURED' for row in captures):
            raise RuntimeError('No suitable native C4 cached-decode invocation reached every TP rank.')
        if any(row['request_ids'] != captures[0]['request_ids']
               or row['selected'] != captures[0]['selected'] for row in captures):
            raise RuntimeError('TP ranks captured different request ordering or depth selections.')
        status['phase'] = 'FIXTURE_PREPARATION'
        write(output / 'status.json', status)
        write(output / 'preparation.json', llm.collective_rpc('c07_prepare_fixture', timeout=600))
        status['phase'] = 'EAGER_PHYSICAL_SERVICE'
        write(output / 'status.json', status)
        start = time.perf_counter_ns()
        service = llm.collective_rpc('c07_service', timeout=600, args=(args.repeats,))
        write(output / 'service.json', dict(ranks=service, rpc_wall_ns=time.perf_counter_ns() - start,
              scope='RPC wall includes parity, input restoration, barriers, and all repeats. Not one-pass latency.'))
        write(output / 'fixture_manifests.json', llm.collective_rpc('c07_export', timeout=600))
        status['parity'] = 'BITWISE_PASS' if all(row['phase'] == 'BITWISE_PASS' for row in service) else 'UNRESOLVED'
        status.update(success=status['parity'] == 'BITWISE_PASS', phase=status['parity'])
        write(output / 'status.json', status)
        if not status['success']:
            raise RuntimeError('The eager physical fixture has an unresolved bitwise difference.')
    except BaseException as error:
        if llm is not None:
            if armed:
                try:
                    write(output / 'capture_finish_after_failure.json',
                          llm.collective_rpc('c07_capture_finish', timeout=120, args=(False,)))
                except BaseException:
                    write(output / 'capture_finish_failure.json', dict(traceback=traceback.format_exc()))
            if not (output / 'fixture_manifests.json').exists():
                try:
                    write(output / 'fixture_manifests_after_failure.json', llm.collective_rpc('c07_export', timeout=600))
                except BaseException:
                    write(output / 'fixture_export_failure.json', dict(traceback=traceback.format_exc()))
        write(output / 'failure.json', dict(error_type=type(error).__name__, phase=status['phase'],
                                          traceback=traceback.format_exc()))
        status.update(success=False, error_type=type(error).__name__)
        write(output / 'status.json', status)
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=60)
                write(output / 'teardown.json', dict(engine_core_shutdown=True))
            except BaseException:
                write(output / 'teardown.json', dict(engine_core_shutdown=False, traceback=traceback.format_exc()))


if __name__ == '__main__':
    main()
