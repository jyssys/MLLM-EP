"""One persistent native TP2 group inside the parent's guarded session."""
import argparse
import dataclasses
from multiprocessing.connection import Connection
import os
from pathlib import Path
import time
import traceback

from poc_ouro_parallelism_deep_discovery.contract import MODELS, read, write
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import (
    assignments, inspect_process_leases, serve_worker, validate_ownership)
from poc_ouro_parallelism_deep_discovery.process_lease_diagnostics import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=range(2), required=True)
    parser.add_argument('--control-fd', type=int, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    worker = args.worker
    specification = plan['workers'][worker]
    output = Path(specification['output'])
    connection = Connection(args.control_fd)
    llm, success, failure, shutdown_failure = None, False, None, None
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, specification['physical_gpus']))
        assert os.getpgid(0) == plan['parent_pgid'] and os.getsid(0) == plan['parent_sid']
        assert os.environ['LOOPED_KV_OUTPUT'] == plan['guard_output']
        assert not os.environ.get('OURO_CAMPAIGN_STORAGE_CONTROL') and os.environ.get('POC4_FAMILY') != 'tied'
        for key, value in specification['cache_environment'].items():
            assert os.environ[key] == value
        expected = dict(LOOPED_KV_OUTPUT=plan['guard_output'],
            CUDA_VISIBLE_DEVICES=','.join(map(str, specification['physical_gpus'])),
            VLLM_WORKER_MULTIPROC_METHOD='spawn')
        write(output / 'guard_identity_before_llm.json', snapshot(os.getpid(), expected, 'FRONTEND_BEFORE_LLM'))
        write(output / 'cache_environment.json', dict(specification['cache_environment'],
            compilation_cache_dir=specification['engine_config']['compilation_config']['cache_dir'],
            CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], LOOPED_KV_OUTPUT=os.environ['LOOPED_KV_OUTPUT'],
            process_group=os.getpgid(0), session=os.getsid(0)))
        from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports
        configure_native_imports()
        from vllm import LLM, SamplingParams

        llm = LLM(**specification['engine_config'])
        ownership = llm.collective_rpc('inspect_ownership', timeout=120)
        write(output / 'ownership.json', ownership)
        write(output / 'guard_identity_after_llm.json', snapshot(os.getpid(), expected, 'FRONTEND_AFTER_LLM'))
        write(output / 'guard_identity_rpc.json', llm.collective_rpc('inspect_guard_identity', timeout=120,
                                                                   kwargs=dict(expected=expected)))
        write(output / 'guard_identity_external.json', [snapshot(row['pid'], expected, 'DECLARED_TP_RANK_EXTERNAL')
                                                       for row in ownership])
        validate_ownership(specification['physical_gpus'], MODELS[plan['size']]['layers'], ownership)
        leases = inspect_process_leases([os.getpid()] + [row['pid'] for row in ownership],
            plan['parent_pgid'], plan['parent_sid'], plan['guard_output'], ancestor_pid=os.getpid())
        write(output / 'ownership.json', ownership)
        write(output / 'process_leases.json', leases)
        connection.send(dict(kind='LOADED', worker=worker, pid=os.getpid(),
            physical_gpus=specification['physical_gpus'], ownership=ownership))

        def prepare(command):
            assert command['lane'] in ('WARMUP', 'CLEAN', 'TRACE')
            assert command['length'] == (16 if command['lane'] == 'WARMUP' else 128)
            assert type(command['case_index']) is int and 0 <= command['case_index'] < len(plan['cases'])
            requests = assignments(plan['cases'][command['case_index']])[worker]
            prompts = [dict(prompt_token_ids=request['prompt_token_ids']) for request in requests]
            params = SamplingParams(temperature=0, max_tokens=command['length'],
                ignore_eos=True, allowed_token_ids=[plan['fixed_token_id']])
            if command['lane'] == 'TRACE':
                llm.collective_rpc('cohort_trace_start', timeout=120)
            return dict(command=command, requests=requests, prompts=prompts, params=params), [r['cohort_slot'] for r in requests]

        def execute(context):
            command, records = context['command'], []
            released_ns, wall_ns = time.perf_counter_ns(), 0
            started_ns, finished_ns, trace = None, None, None
            try:
                if context['requests']:
                    started_ns = time.perf_counter_ns()
                    results = llm.generate(context['prompts'], context['params'], use_tqdm=False)
                    finished_ns = time.perf_counter_ns()
                    wall_ns = finished_ns - started_ns
                    assert len(results) == len(context['requests'])
                    for request, frozen in zip(results, context['requests'], strict=True):
                        assert len(request.outputs) == 1
                        assert list(request.prompt_token_ids) == frozen['prompt_token_ids']
                        assert list(request.outputs[0].token_ids) == [plan['fixed_token_id']] * command['length']
                        records.append(dict(cohort_slot=frozen['cohort_slot'], request_id=request.request_id,
                            source_request_id=frozen['source_request_id'], output_tokens=command['length'],
                            finish_reason=request.outputs[0].finish_reason,
                            metrics=None if request.metrics is None else dataclasses.asdict(request.metrics)))
            finally:
                if command['lane'] == 'TRACE':
                    trace = llm.collective_rpc('cohort_trace_finish', timeout=120)
            return dict(worker=worker, physical_gpus=specification['physical_gpus'],
                assigned_slots=[r['cohort_slot'] for r in context['requests']], generated_tokens=len(records) * command['length'],
                request_records=records, local_generate_wall_ns=wall_ns,
                local_generate_start_ns=started_ns, local_generate_end_ns=finished_ns, release_received_ns=released_ns,
                output_identity='ALL_FROZEN_CONSTANT_TOKEN_IDS', schedule_trace=trace)

        serve_worker(connection, worker, prepare, execute)
        success = True
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, traceback=traceback.format_exc())
        write(output / 'failure.json', failure)
        try:
            connection.send(dict(kind='ERROR', worker=worker, failure=failure))
        except (OSError, EOFError):
            pass
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=plan['tp2_dp2_protocol']['shutdown_timeout_seconds'] / 2)
            except BaseException as error:
                shutdown_failure = dict(error_type=type(error).__name__, traceback=traceback.format_exc())
                write(output / 'shutdown_failure.json', shutdown_failure)
        success = success and shutdown_failure is None
        write(output / 'status.json', dict(success=success, pid=os.getpid(), worker=worker,
            failure=failure, shutdown_failure=shutdown_failure, quality='NOT_RUN'))
        if success:
            try:
                connection.send(dict(kind='STOPPED', worker=worker))
            except (OSError, EOFError):
                success = False
        connection.close()
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
