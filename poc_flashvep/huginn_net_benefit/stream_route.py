"""Optional independent prompt stream. Admission safety remains a separate gate."""
import os
from .common import ROOT, write


def prompt_route(flags):
    if not flags or any(flags) and not all(flags):
        raise ValueError('Independent prompt streams require a nonempty, single-phase batch')
    return all(flags)


def install(cfg):
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    initialize, close, submit = StageEngine.__init__, StageEngine.close, ModelRunner.submit
    synchronize = ModelRunner.synchronize
    counters = dict(prompt_submissions=0, decode_submissions=0, coda_lane_submissions=0)

    def buffers(engine, prompt):
        from vllm_rlt.worker.submission_queue import SubmissionQueue, MAX_CONCURRENT_SUBMISSIONS
        from vllm_rlt.worker.model_runner import BatchInputBuffers
        from vllm_rlt.worker.buffers import PackedMetadataBuffer
        from vllm_rlt.worker.tensors import StagedInputTensor
        runner, scheduler = engine.model_runner, engine.scheduler.config
        dtype = next(engine.model.parameters()).dtype
        rows = max(scheduler.max_num_batched_tokens, scheduler.prefill_chunk_size) if prompt else scheduler.max_num_seqs
        return dict(submissions=SubmissionQueue(),
            input_buffers={kind: StagedInputTensor(kind, runner.device,
                use_uva=runner.execution_config.use_uva) for kind in (torch.int64, torch.int32)},
            loop_hidden=[None]*MAX_CONCURRENT_SUBMISSIONS,
            packed_metadata_buffers=[PackedMetadataBuffer(runner.device,
                use_uva=runner.execution_config.use_uva) for _ in range(MAX_CONCURRENT_SUBMISSIONS)],
            batch_input_buffers=[BatchInputBuffers(scheduler.max_num_seqs, rows,
                runner.device, dtype, use_uva=runner.execution_config.use_uva)
                for _ in range(MAX_CONCURRENT_SUBMISSIONS)])

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        runner = engine.model_runner
        if engine.stage_config.role == 'decode':
            packed = getattr(engine, '_autonomous_prefill_graphs', None)
            assert packed is not None and packed is not runner.graphs, 'Prompt and Decode graph storage must be distinct'
            assert packed.pool != runner.graphs.pool, 'Concurrent graph caches must not share a capture pool'
            runner.net_prompt_stream = torch.cuda.Stream(device=runner.device, priority=0)
            if cfg.get('disjoint_green_prompt_sms'):
                from .green_pair import create_pair
                assert not packed.entries and not runner.graphs.entries
                device_id = runner.device.index if runner.device.index is not None else torch.cuda.current_device()
                decode, prompt, receipt = create_pair(cfg['disjoint_green_prompt_sms'], device_id)
                runner.net_decode_green, runner.net_prompt_green = decode, prompt
                runner.net_prompt_stream = prompt.Stream()
                runner.core_stream = decode.Stream()
                runner.boundary_stream = decode.new_stream(-1, device_id)
                if runner.readback_stream is not None:
                    runner.readback_stream = decode.new_stream(-1, device_id)
                runner.graphs.capture_stream = decode.Stream()
                if runner.coda_graphs is not None:
                    assert not runner.coda_graphs.entries
                    runner.coda_graphs.stream = runner.boundary_stream
                packed.capture_stream = prompt.Stream()
                runner.net_green_receipt = receipt
                assert not cfg.get('prompt_green_sms'), 'Select one resource-allocation method'
            if cfg.get('prompt_green_sms'):
                from torch.cuda.green_contexts import GreenContext
                from cuda.bindings import driver
                assert not packed.entries, 'A resource-specific graph cache must be configured before any capture'
                device_id = runner.device.index if runner.device.index is not None else torch.cuda.current_device()
                runner.net_prompt_green = GreenContext.create(cfg['prompt_green_sms'], device_id)
                runner.net_prompt_stream = runner.net_prompt_green.Stream()
                if not isinstance(runner.net_prompt_stream, torch.cuda.Stream):
                    value = runner.net_prompt_stream
                    runner.net_prompt_stream = torch.cuda.Stream(stream_id=value.stream_id,
                        device_index=value.device_index, device_type=value.device_type)
                # A graph launch stream does not change its captured node
                # resources. Capture the prompt graphs on the same Green stream.
                packed.capture_stream = runner.net_prompt_stream
                status, context = driver.cuStreamGetGreenCtx(driver.CUstream(runner.net_prompt_stream.cuda_stream))
                assert int(status) == 0 and int(context), 'Prompt stream has no valid Green context'
                status, resource = driver.cuGreenCtxGetDevResource(context,
                    driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM)
                assert int(status) == 0
                runner.net_green_receipt = dict(requested_SM_count=cfg['prompt_green_sms'],
                    allocated_SM_count=resource.sm.smCount,
                    stream=runner.net_prompt_stream.cuda_stream,
                    capture_stream=packed.capture_stream.cuda_stream,
                    mechanism='Documented CUDA Green context. Normal Decode streams retain full-device access. This is not a disjoint reservation or a concurrency guarantee.')
            runner.net_original_core_stream = runner.core_stream
            if cfg.get('independent_prompt_buffers'):
                assert not scheduler_wavefront(engine), 'Independent lanes require the qualified non-wavefront scheduler'
                runner.net_prompt_fields = buffers(engine, True)
            if cfg.get('independent_coda_buffers'):
                runner.net_coda_fields = buffers(engine, False)
            if cfg.get('priority_decode_stream') and not getattr(runner, 'net_decode_green', None):
                runner.core_stream = torch.cuda.Stream(device=runner.device, priority=-1)
            if cfg.get('priority_decode_boundaries') and not getattr(runner, 'net_decode_green', None):
                runner.boundary_stream = torch.cuda.Stream(device=runner.device, priority=-1)
                if runner.coda_graphs is not None:
                    assert not runner.coda_graphs.entries, 'Boundary priority must precede graph capture'
                    runner.coda_graphs.stream = runner.boundary_stream
                if runner.readback_stream is not None:
                    runner.readback_stream = torch.cuda.Stream(device=runner.device, priority=-1)
            runner.net_independent_streams = True
        return result

    def submitted(runner, scheduled=None, *, batch):
        if not getattr(runner, 'net_independent_streams', False):
            return submit(runner, scheduled, batch=batch)
        prompt = prompt_route([span.chunk_start is not None for span in batch.spans])
        coda = not prompt and batch.operation.value == 'coda' and bool(getattr(runner, 'net_coda_fields', None))
        if not prompt and not coda:
            counters['decode_submissions'] += 1
            green = getattr(runner, 'net_decode_green', None)
            if green is not None:
                green.set_context()
            try:
                return submit(runner, scheduled, batch=batch)
            finally:
                if green is not None:
                    green.pop_context()
        counters['prompt_submissions' if prompt else 'coda_lane_submissions'] += 1
        core, boundary = runner.core_stream, runner.boundary_stream
        fields = getattr(runner, 'net_prompt_fields' if prompt else 'net_coda_fields', {})
        native = {key: getattr(runner, key) for key in fields}
        for key, value in fields.items():
            setattr(runner, key, value)
        if prompt:
            runner.core_stream = runner.boundary_stream = runner.net_prompt_stream
        green = getattr(runner, 'net_prompt_green' if prompt else 'net_decode_green', None)
        green_current = False
        try:
            if green is not None:
                green.set_context()
                green_current = True
            # Native per-request waits and the global bounded submission queue
            # remain responsible for metadata and hidden-buffer lifetimes.
            return submit(runner, scheduled, batch=batch)
        finally:
            if green_current:
                green.pop_context()
            for key in fields:
                fields[key] = getattr(runner, key)
                setattr(runner, key, native[key])
            runner.core_stream, runner.boundary_stream = core, boundary

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'independent_streams_{os.getpid()}.json', dict(
                role=engine.stage_config.role, priority_decode=cfg.get('priority_decode_stream', False),
                independent_prompt_buffers=cfg.get('independent_prompt_buffers', False),
                independent_coda_buffers=cfg.get('independent_coda_buffers', False),
                priority_decode_boundaries=cfg.get('priority_decode_boundaries', False),
                green_context=getattr(engine.model_runner, 'net_green_receipt', None),
                **counters,
                scope='Separate prompt and Decode graph storage and streams. Native request events, allocation guards, and bounded input buffers remain. Separate streams alone do not prove GPU overlap or Decode non-inferiority.'))
    def synchronized(runner):
        for stream in (getattr(runner, 'net_prompt_stream', None),
                getattr(runner, 'net_original_core_stream', None)):
            if stream is not None:
                stream.synchronize()
        return synchronize(runner)
    StageEngine.__init__, StageEngine.close, ModelRunner.submit = initialized, closed, submitted
    ModelRunner.synchronize = synchronized


def scheduler_wavefront(engine):
    return engine.scheduler.config.wavefront_chunked_prefill
