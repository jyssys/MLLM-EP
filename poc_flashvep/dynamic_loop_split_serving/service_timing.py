"""Record actual model-stream entry without synchronizing timed submissions.

The event marks the start of the native model execution enclosure.
It is not a CUPTI first-kernel timestamp or an exclusive GEMM duration.
"""
import os
import time
from .common import ROOT, write


def device_time(anchor, elapsed_ms):
    offset = elapsed_ms / 1000
    return dict(time=anchor['midpoint'] + offset,
                lower=anchor['lower'] + offset, upper=anchor['upper'] + offset)


def calibrated_device_time(anchor, end_anchor, total_ms, elapsed_ms):
    if not total_ms > 0 or not 0 <= elapsed_ms <= total_ms:
        raise ValueError('Invalid CUDA clock calibration interval')
    weight = elapsed_ms / total_ms
    return dict(time=anchor['midpoint'] + weight * (end_anchor['midpoint'] - anchor['midpoint']),
        lower=anchor['lower'] + weight * (end_anchor['lower'] - anchor['lower']),
        upper=anchor['upper'] + weight * (end_anchor['upper'] - anchor['upper']))


def prompt_span(batch, span):
    return (span.chunk_start is not None
            or (batch.operation.value == 'prelude' and span.token_count > 1
                and not span.request.generated_token_ids))


def install(cfg):
    if not cfg.get('observe_service_timing', False):
        return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    initialize, close, execute = StageEngine.__init__, StageEngine.close, ModelRunner._execute

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        runner = engine.model_runner
        torch.cuda.synchronize(runner.device)
        event = torch.cuda.Event(enable_timing=True)
        stream = runner.core_stream or torch.cuda.current_stream(runner.device)
        low = time.monotonic()
        event.record(stream)
        event.synchronize()
        high = time.monotonic()
        runner._serving_timing = dict(anchor_event=event,
            anchor=dict(lower=low, upper=high, midpoint=(low + high) / 2),
            requests={}, event_record_host_seconds=0., event_records=0)
        return result

    def observed(runner, scheduled_batch, prepared):
        state = runner._serving_timing
        batch = prepared.batch
        spans = [span for span in batch.spans if prompt_span(batch, span)]
        stream = torch.cuda.current_stream(runner.device)
        for span in spans:
            request = span.request
            if request.request_id not in state['requests']:
                begin = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                before = time.perf_counter()
                begin.record(stream)
                state['event_record_host_seconds'] += time.perf_counter() - before
                state['event_records'] += 1
                state['requests'][request.request_id] = dict(begin=begin, finish=finish,
                    seed=request.sampling_params.seed, prompt_tokens=len(request.prompt_token_ids),
                    first_host_submit=time.monotonic(), last_host_submit=None,
                    prompt_submissions=0, assigned_recurrent_token_rows=0)
        result = execute(runner, scheduled_batch, prepared)
        for span in spans:
            row = state['requests'][span.request.request_id]
            before = time.perf_counter()
            row['finish'].record(stream)
            state['event_record_host_seconds'] += time.perf_counter() - before
            state['event_records'] += 1
            row['last_host_submit'] = time.monotonic()
            row['prompt_submissions'] += 1
            row['assigned_recurrent_token_rows'] += span.token_count * batch.loop_count
        return result

    def closed(engine):
        runner = engine.model_runner
        runner.synchronize()
        state = runner._serving_timing
        end_event = torch.cuda.Event(enable_timing=True)
        low = time.monotonic()
        end_event.record(runner.core_stream or torch.cuda.current_stream(runner.device))
        end_event.synchronize()
        high = time.monotonic()
        end_anchor = dict(lower=low, upper=high, midpoint=(low + high) / 2)
        total_ms = state['anchor_event'].elapsed_time(end_event)
        drift_ms = 1000 * (end_anchor['midpoint'] - state['anchor']['midpoint']) - total_ms
        records = []
        for rid, row in state['requests'].items():
            begin = calibrated_device_time(state['anchor'], end_anchor, total_ms,
                                            state['anchor_event'].elapsed_time(row['begin']))
            finish = calibrated_device_time(state['anchor'], end_anchor, total_ms,
                                             state['anchor_event'].elapsed_time(row['finish']))
            records.append(dict(request_id=rid, **{k: v for k, v in row.items() if k not in ('begin', 'finish')},
                first_model_stream_entry=begin, last_prompt_stream_end=finish,
                prompt_device_enclosure_ms=1000 * (finish['time'] - begin['time'])))
        write(ROOT / 'raw' / cfg['name'] / f'service_device_times_{os.getpid()}.json', dict(
            role=engine.stage_config.role, anchor=state['anchor'], requests=records,
            end_anchor=end_anchor, CUDA_anchor_interval_ms=total_ms, endpoint_clock_drift_ms=drift_ms,
            clock_mapping='Linear interpolation between measured startup and close brackets. Endpoint uncertainty is retained. Nonlinear clock drift is not bounded by these brackets.',
            event_records=state['event_records'], event_record_host_seconds=state['event_record_host_seconds'],
            evidence='MEASURED_CUDA_EVENT_MODEL_STREAM_ENCLOSURES',
            scope='First native model execution stream entry after input preparation. Clock mapping retains host-anchor uncertainty. No synchronization occurs in timed submissions. An enclosure can include scheduling gaps and other requests. It is not an exclusive kernel sum.'))
        return close(engine)

    StageEngine.__init__, StageEngine.close, ModelRunner._execute = initialized, closed, observed
