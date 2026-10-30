"""Diagnostic operand accounting. No policy or tensor value changes occur here."""
import os
import time
from collections import Counter
from contextlib import nullcontext


def install():
    if os.environ.get('RCC_PROBE') != '1':
        return
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from multiprocessing.util import Finalize
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.worker.executor import ModelRuntime
    from .common import ROOT, write
    if getattr(ModelRunner, '_rcc_probe_installed', False):
        return
    ModelRunner._rcc_probe_installed = True

    batches = []
    calls = []
    target_traces = []
    context = [None]
    execute = ModelRunner._execute_batch
    forward = ModelRuntime.forward

    def traced_execute(self, batch, prepared):
        offset = 0
        spans = []
        for span in batch.spans:
            spans.append(dict(request_id=span.request.request_id,
                              seed=span.request.sampling_params.seed,
                              depth=span.depth, position=span.token_start,
                              rows=span.token_count, row_offset=offset,
                              phase='decode' if span.emit_gate else 'prefill' if span.chunk_start is not None else 'boundary'))
            offset += span.token_count
        item = dict(id=len(batches), operation=batch.operation.value,
                    loops=batch.loop_count, rows=batch.num_rows, spans=spans,
                    host_begin=time.monotonic())
        batches.append(item)
        context[0] = item
        try:
            return execute(self, batch, prepared)
        finally:
            item['host_end'] = time.monotonic()
            context[0] = None

    class Shapes(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.linear.default, torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                        torch.ops.aten.bmm.default):
                shapes = tuple(tuple(x.shape) for x in args if isinstance(x, torch.Tensor))
                self.counts[(str(func), shapes)] += 1
            return func(*args, **(kwargs or {}))

    def traced_forward(self, batch):
        index = context[0]['id'] if context[0] is not None else None
        phases = {s['phase'] for s in context[0]['spans']} if context[0] else set()
        trace = None
        if (os.environ.get('RCC_TARGET_TRACE') == '1' and not target_traces
                and phases >= {'decode', 'prefill'}):
            trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        shapes = Shapes()
        with (trace if trace is not None else nullcontext()), shapes, torch.profiler.record_function(f'RCC_core_{index}'):
            result = forward(self, batch)
        end.record()
        if trace is not None:
            target_traces.append((index, trace))
        calls.append(dict(batch_id=index, operation=batch.operation.value,
                          rows=len(batch.hidden) if batch.hidden is not None else None,
                          graph_requested=batch.graphs is not None,
                          mm_shapes=[dict(op=op, shapes=shape, count=count)
                                     for (op, shape), count in shapes.counts.items()],
                          events=(start, end)))
        return result

    ModelRunner._execute_batch = traced_execute
    ModelRuntime.forward = traced_forward

    def finish():
        if calls:
            torch.cuda.synchronize()
        for call in calls:
            start, end = call.pop('events')
            call['cuda_event_envelope_ms'] = start.elapsed_time(end)
        for index, trace in target_traces:
            trace.export_chrome_trace(str(ROOT/'raw'/os.environ['RCC_RUN']/f'mixed_core_trace_{os.getpid()}_{index}.json'))
        write(ROOT / 'raw' / os.environ['RCC_RUN'] / f'fusion_probe_{os.getpid()}.json',
              dict(batches=batches, calls=calls,
                   evidence='MEASURED_DIAGNOSTIC',
                   limitation='CUDA events measure each runtime envelope. They do not isolate kernel time. Graph replay hides internal aten calls.'))

    Finalize(None, finish, exitpriority=18)
