"""Bounded ready-queue and model-stream enclosures, not exclusive kernel time."""
import json
import os
import time
from .common import ROOT, write


def install(cfg):
    if not cfg.get('dlocal_trace'):
        return
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.worker.model_runner import ModelRunner
    initialize, closed = StageEngine.__init__, StageEngine.close
    scheduled, executed = Scheduler.schedule, ModelRunner._execute
    folder = ROOT / 'raw' / cfg['name']

    def initialized(self, *args, **kwargs):
        result = initialize(self, *args, **kwargs)
        if self.stage_config.role != 'decode':
            return result
        runner = self.model_runner
        stream = runner.core_stream or torch.cuda.current_stream(runner.device)
        event = torch.cuda.Event(enable_timing=True)
        low = time.monotonic(); event.record(stream); event.synchronize(); high = time.monotonic()
        state = dict(anchor=event, low=low, high=high, queues=[], operations=[], origin=None)
        runner._local_trace = self.scheduler._local_trace = self._local_trace = state
        return result

    def active(state):
        if state['origin'] is None and (folder / 'measurement_begin.json').exists():
            state['origin'] = json.loads((folder / 'measurement_begin.json').read_text())['time']
        if state['origin'] is None:
            return False
        start = state['origin'] + .35 * cfg['arrival_offsets'][-1]
        return start <= time.monotonic() <= start + cfg.get('dlocal_trace_seconds', 3.)

    def schedule(self, *args, **kwargs):
        state = getattr(self, '_local_trace', None)
        before = None
        if state is not None and active(state):
            before = dict(time=time.monotonic(), queues={str(p): len(q) for p, q in self.queues.items()},
                free_blocks=self.cache_manager.num_free_blocks if hasattr(self, 'cache_manager') else None)
        result = scheduled(self, *args, **kwargs)
        if before is not None:
            before.update(end=time.monotonic(), selected=[] if result is None else
                [dict(request_id=s.request.request_id, phase=str(result.phase_for(s)), depth=s.depth,
                    token_start=s.token_start, token_count=s.token_count) for s in result.requests])
            state['queues'].append(before)
        return result

    def execute(self, scheduled_batch, prepared):
        state = getattr(self, '_local_trace', None)
        if state is None or not active(state):
            return executed(self, scheduled_batch, prepared)
        left, right = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device)
        host = time.monotonic(); left.record(stream)
        result = executed(self, scheduled_batch, prepared)
        right.record(stream)
        state['operations'].append(dict(begin=left, end=right, host_submit=host,
            operation=prepared.batch.operation.value, loop_count=prepared.batch.loop_count,
            spans=[dict(request_id=s.request.request_id, seed=s.request.sampling_params.seed,
                depth=s.depth, start=s.token_start, count=s.token_count,
                phase=str(s.request.phase)) for s in prepared.batch.spans]))
        return result

    def close(self):
        state = getattr(self, '_local_trace', None)
        if state is not None:
            self.model_runner.synchronize()
            ops = []
            for row in state['operations']:
                ops.append(dict(**{k:v for k,v in row.items() if k not in ('begin','end')},
                    begin_device_ms=state['anchor'].elapsed_time(row['begin']),
                    end_device_ms=state['anchor'].elapsed_time(row['end']),
                    enclosure_ms=row['begin'].elapsed_time(row['end'])))
            write(folder / f'dlocal_trace_{os.getpid()}.json', dict(operations=ops, queues=state['queues'],
                anchor_host_lower=state['low'], anchor_host_upper=state['high'],
                evidence='MEASURED_BOUNDED_HOST_READY_QUEUES_AND_CUDA_EVENT_ENCLOSURES',
                scope='Queue membership is host scheduling readiness. Device intervals enclose model work and gaps. They are not exclusive kernel time or proof of safe idle windows.'))
        return closed(self)
    StageEngine.__init__, StageEngine.close = initialized, close
    Scheduler.schedule, ModelRunner._execute = schedule, execute
