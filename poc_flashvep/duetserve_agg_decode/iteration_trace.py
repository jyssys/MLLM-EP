"""Optional host scheduling and CUDA-event enclosures, not kernel intervals."""
import atexit
import collections
import json
import time

from .common import ROOT, write


def install(config, role):
    import torch
    import vllm
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    folder = ROOT / 'raw' / config['name']
    scheduled, executed, pending_events = [], [], []
    current = {}
    anchor_state = {}
    if config.get('iteration_trace_events', True):
        original_llm = vllm.LLM
        def initialized(*args, **kwargs):
            value = original_llm(*args, **kwargs)
            # Clear initialization work before the observation clock. All
            # synchronization occurs before request warm-up and measurement.
            torch.cuda.synchronize()
            anchor = torch.cuda.Event(enable_timing=True)
            before = time.monotonic()
            anchor.record()
            anchor.synchronize()
            after = time.monotonic()
            anchor_state.update(event=anchor, host_lower=before, host_upper=after)
            return value
        vllm.LLM = initialized
    schedule = Scheduler.schedule
    def observed_schedule(self, *args, **kwargs):
        before = {rid: r.num_computed_tokens for rid, r in self.requests.items()}
        begin = time.monotonic()
        result = schedule(self, *args, **kwargs)
        end = time.monotonic()
        requests = []
        for rid, count in result.num_scheduled_tokens.items():
            request = self.requests.get(rid)
            if request is None:
                continue
            length = len(request.prompt_token_ids)
            start = before.get(rid, 0)
            requests.append(dict(request_id=rid, computed_before=start, query_tokens=count,
                prompt_tokens=length, prefill_tokens=max(0, min(count, length - start)),
                context_tokens=start))
        if requests:
            scheduled.append(dict(host_start=begin, host_end=end, requests=requests,
                waiting=len(self.waiting), running=len(self.running),
                waiting_statuses=dict(collections.Counter(str(r.status) for r in self.waiting)),
                KV_usage=self.kv_cache_manager.usage, free_KV_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                estimated_iteration_cost_ms=None, decode_deadline=None,
                realtime_admission='NOT_IMPLEMENTED_BY_STOCK_SCHEDULER',
                scope='HOST_SCHEDULER. Async execution may be one or more steps later. Token budget is not a calibrated time budget.'))
        return result
    Scheduler.schedule = observed_schedule
    prepare = GPUModelRunner.prepare_inputs
    def prepared(self, *args, **kwargs):
        batch = prepare(self, *args, **kwargs)
        current['batch'] = batch
        current['metadata'] = dict(request_ids=list(batch.req_ids), query_tokens=batch.num_tokens,
            padded_query_tokens=batch.num_tokens_after_padding,
            query_lengths=batch.num_scheduled_tokens.tolist(), computed_tokens=batch.num_computed_tokens_np.tolist(),
            prefill_lens=batch.prefill_len_np.tolist(),
            prefilling=batch.is_prefilling_np.tolist())
        return batch
    GPUModelRunner.prepare_inputs = prepared
    execute = GPUModelRunner.execute_model
    def observed_execute(self, *args, **kwargs):
        if kwargs.get('dummy_run') or torch.cuda.is_current_stream_capturing():
            return execute(self, *args, **kwargs)
        # Metadata belongs to prepare_inputs inside execute_model. Do not attach
        # the previous prepared batch to the new call.
        current.pop('batch', None)
        current.pop('metadata', None)
        host_start = time.monotonic()
        stream = torch.cuda.current_stream()
        begin = end = None
        if config.get('iteration_trace_events', True):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
        value = execute(self, *args, **kwargs)
        if end is not None:
            end.record(stream)
        batch = current.get('batch')
        if batch is not None:
            metadata = dict(kind='MODEL_FORWARD', host_start=host_start, host_end=time.monotonic(),
                stream=int(stream.cuda_stream), **current['metadata'],
                scope='CUDA_EVENT_ENCLOSURE. Includes stream waiting and host submission gaps. Not kernel union, physical idle, or exclusive compute.')
            if begin is None:
                metadata['scope'] = 'HOST_ENCLOSURE_AND_ACTUAL_INPUT_METADATA. No GPU execution time measured.'
            pending_events.append((begin, end, metadata))
        return value
    GPUModelRunner.execute_model = observed_execute
    sample = GPUModelRunner.sample_tokens
    def observed_sample(self, *args, **kwargs):
        frame = self.execute_model_state
        if frame is None:
            return sample(self, *args, **kwargs)
        batch = frame.input_batch
        if not all(rid.split('-')[0].startswith('q') for rid in batch.req_ids):
            return sample(self, *args, **kwargs)
        host_start = time.monotonic()
        stream = torch.cuda.current_stream()
        metadata = dict(kind='SAMPLER', host_start=host_start,
            stream=int(stream.cuda_stream), request_ids=list(batch.req_ids),
            query_tokens=batch.num_tokens, query_lengths=batch.num_scheduled_tokens.tolist(),
            computed_tokens=batch.num_computed_tokens_np.tolist(), prefill_lens=batch.prefill_len_np.tolist(),
            prefilling=batch.is_prefilling_np.tolist())
        begin = end = None
        if config.get('iteration_trace_events', True):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
        value = sample(self, *args, **kwargs)
        if end is not None:
            end.record(stream)
        metadata.update(host_end=time.monotonic(),
            scope='CUDA_EVENT_ENCLOSURE. Sampling, host submission, and stream waits are not separated.' if begin is not None else 'HOST_SAMPLER_ENCLOSURE. Not GPU time.')
        pending_events.append((begin, end, metadata))
        return value
    GPUModelRunner.sample_tokens = observed_sample
    def close():
        if any(begin is not None for begin, _, _ in pending_events):
            torch.cuda.synchronize()
        for begin, end, metadata in pending_events:
            row = dict(metadata, CUDA_event_enclosure_ms=begin.elapsed_time(end) if begin is not None else None)
            if begin is not None:
                anchor = anchor_state['event']
                row.update(GPU_event_start_relative_anchor_ms=anchor.elapsed_time(begin),
                    GPU_event_end_relative_anchor_ms=anchor.elapsed_time(end))
            executed.append(row)
        write(folder / f'iteration_trace_{role}.json', dict(scheduler=scheduled, execution=executed,
            GPU_event_clock_anchor={key: value for key, value in anchor_state.items() if key != 'event'},
            evidence='TRACE_ANALYSIS',
            scope='Optional diagnostic. No per-iteration CUDA synchronization. Performance interference requires a separate control. No admission cost or deadline exists in this stock aggregated policy.'))
    atexit.register(close)


def bootstrap():
    import os
    import sys
    if '--role' not in sys.argv:
        return
    config = json.loads(open(os.environ['DENSE_CONFIG']).read())
    if config.get('iteration_trace'):
        install(config, sys.argv[sys.argv.index('--role') + 1])
