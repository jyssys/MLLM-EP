"""Inclusive host timings. These counters do not measure CUDA kernel time."""
import os
import time
from collections import defaultdict
from .common import ROOT, write


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner, Submission
    folder = ROOT / 'raw' / cfg['name']
    marker = folder / 'measurement_begin.json'
    active = [False]
    totals = defaultdict(lambda: dict(calls=0, host_seconds=0., owner_thread_CPU_seconds=0., max_host_ms=0.))
    prompt_records = []
    wire_records = []
    python_profile = None
    if cfg.get('profile_python'):
        import cProfile
        python_profile = cProfile.Profile()
    from ..adaptive_recurrence_cut.connector import CutConnector
    from vllm_rlt.pd.transport import NixlConnector
    snapshot = CutConnector.queue_snapshot
    def queued(connector, transfer, n, event):
        start, cpu = time.monotonic(), time.thread_time()
        before = transfer.next_sequence
        try:
            return snapshot(connector, transfer, n, event)
        finally:
            if marker.exists():
                wire_records.append(dict(event='DESCRIPTOR_AND_QUEUE', tid=transfer.tid,
                    seed=transfer.request.sampling_params.seed, start=start, end=time.monotonic(),
                    owner_thread_CPU_ms=1000*(time.thread_time()-cpu), fragments=transfer.next_sequence-before))
    CutConnector.queue_snapshot = queued
    submit_wire, poll_wire = NixlConnector.submit, NixlConnector.poll
    def submitted_wire(owner, tid, sequence, peer, local, remote, size):
        begin, cpu = time.monotonic(), time.thread_time()
        accepted = submit_wire(owner, tid, sequence, peer, local, remote, size)
        if accepted and marker.exists():
            wire_records.append(dict(event='NIXL_SUBMIT_ACCEPTED', tid=tid, sequence=sequence,
                bytes=size, descriptors=len(local), start=begin, end=time.monotonic(),
                owner_thread_CPU_ms=1000*(time.thread_time()-cpu)))
        return accepted
    def polled_wire(owner):
        finished, notifications = poll_wire(owner)
        if finished and marker.exists():
            now = time.monotonic()
            wire_records.extend(dict(event='NIXL_DONE_OBSERVED', tid=t.transfer_id,
                sequence=t.sequence, bytes=t.size, time=now, submitted=t.submitted) for t in finished)
        return finished, notifications
    NixlConnector.submit, NixlConnector.poll = submitted_wire, polled_wire

    def timed(cls, method):
        original = getattr(cls, method)
        def observed(owner, *args, **kwargs):
            if not active[0]:
                active[0] = marker.exists()
                if active[0] and python_profile is not None:
                    python_profile.enable()
            if not active[0]:
                return original(owner, *args, **kwargs)
            batch = kwargs.get('batch')
            if batch is None and args:
                batch = getattr(args[0], 'batch', None)
                if batch is None and hasattr(args[0], 'spans'):
                    batch = args[0]
                if batch is None and len(args) > 1:
                    batch = getattr(args[1], 'batch', None)
            prompt = bool(batch is not None and any(s.token_count > 1 for s in batch.spans))
            label = method + ('_prompt' if prompt else '_decode_or_control')
            if method == 'collect':
                label = 'collect_' + str(getattr(getattr(owner, 'operation', None), 'value', 'unknown'))
            start = time.perf_counter()
            cpu_start = time.thread_time()
            try:
                return original(owner, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                counter = totals[label]
                counter['calls'] += 1
                counter['host_seconds'] += elapsed
                counter['owner_thread_CPU_seconds'] += time.thread_time() - cpu_start
                counter['max_host_ms'] = max(counter['max_host_ms'], 1000*elapsed)
                if method == 'submit' and prompt:
                    prompt_records.append(dict(time=time.monotonic(), host_submit_ms=1000*elapsed,
                        operation=batch.operation.value, rows=batch.num_rows, loops=batch.loop_count,
                        spans=[dict(tid=s.request.request_id, depth=s.depth, start=s.token_start,
                            tokens=s.token_count, seed=s.request.sampling_params.seed) for s in batch.spans]))
        setattr(cls, method, observed)

    for method in ('prepare_on_cpu', 'prepare_on_gpu', '_execute', 'submit'):
        timed(ModelRunner, method)
    timed(StageEngine, '_schedule')
    timed(Submission, 'collect')
    close = StageEngine.close
    def closed(engine):
        try:
            return close(engine)
        finally:
            if python_profile is not None:
                python_profile.disable()
                python_profile.dump_stats(str(folder / f'python_cost_{os.getpid()}.pstats'))
            write(folder / f'host_cost_{os.getpid()}.json', dict(role=engine.stage_config.role,
                totals=dict(totals), prompt_records=prompt_records, wire_records=wire_records,
                scope='Inclusive host call durations. Nested counters overlap. Do not sum them. CUDA launch waits can occur inside these calls. Not GPU kernel duration.'))
    StageEngine.close = closed
