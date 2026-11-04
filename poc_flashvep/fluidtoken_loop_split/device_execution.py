"""Optional stream-event enclosures for isolated method calibration.

Events include device waits and launch gaps between the two markers.
They are not kernel unions, exclusive E2E components, or clean serving results.
"""
import os
import time
from .common import ROOT, write


def install(cfg):
    if not cfg.get('observe_device_execution'):
        return
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    execute, close = ModelRunner._execute, StageEngine.close
    samples = []
    marker = ROOT / 'raw' / cfg['name'] / 'measurement_begin.json'
    measured = [False]

    def observed(runner, scheduled_batch, prepared):
        if not measured[0]:
            measured[0] = marker.exists()
        batch = prepared.batch
        metadata = dict(host_submit_monotonic=time.monotonic(), measured=measured[0],
            operation=batch.operation.value, logical_rows=batch.num_rows, loops=batch.loop_count,
            requests=[dict(request_id=s.request.request_id, seed=s.request.sampling_params.seed,
                token_start=s.token_start, token_count=s.token_count, depth=s.depth,
                phase=s.request.phase.value, generated_tokens=len(s.request.generated_token_ids))
                for s in batch.spans])
        # Native submit() already selected the core or boundary stream.
        stream = torch.cuda.current_stream(runner.device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        result = execute(runner, scheduled_batch, prepared)
        end.record(stream)
        samples.append((begin, end, metadata))
        return result

    def closed(engine):
        engine.model_runner.synchronize()
        rows = [dict(metadata, stream_elapsed_ms=begin.elapsed_time(end))
                for begin, end, metadata in samples]
        write(ROOT / 'raw' / cfg['name'] / f'device_execution_{os.getpid()}.json', dict(
            role=engine.stage_config.role, rows=rows,
            evidence='MEASURED_CUDA_EVENT_STREAM_ENCLOSURES',
            scope='Includes waits and launch gaps. Overlapping durations are not additive E2E shares. Instrumented calibration, not clean latency.'))
        return close(engine)
    ModelRunner._execute, StageEngine.close = observed, closed
