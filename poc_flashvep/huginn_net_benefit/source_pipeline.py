"""Progress completed, disjoint transfers behind already submitted source work."""
from collections import deque
import os
import time
from .common import ROOT, write


class OrderedControl:
    """Stop at the first unrelated command. Preserve the original FIFO order."""
    def __init__(self, channel):
        self.channel = channel
        self.buffer = deque()

    def poll(self, *args):
        return bool(self.buffer) or self.channel.poll(*args)

    def recv(self):
        return self.buffer.popleft() if self.buffer else self.channel.recv()

    def __getattr__(self, key):
        return getattr(self.channel, key)

    def consume_safe(self, current, consume, limit):
        accepted = 0
        for _ in range(limit):
            if self.buffer or not self.channel.poll():
                break
            message = self.channel.recv()
            kind = message.get('kind')
            safe = kind == 'idle_capacity' or (
                kind in ('idle_prepared', 'idle_rejected', 'idle_committed')
                and message.get('tid') not in current
                and message.get('tid') is not None)
            if not safe:
                self.buffer.append(message)
                break
            consume(message)
            accepted += 1
        return accepted


def install(cfg):
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.worker.model_runner import ModelRunner
    init, execute = StageWorker.__init__, ModelRunner._execute
    close = StageWorker.run
    folder = ROOT/'raw'/cfg['name']
    records = []

    def initialized(worker, *args, **kwargs):
        result = init(worker, *args, **kwargs)
        if worker.engine.stage_config.role == 'prefill':
            worker.channel = OrderedControl(worker.channel)
            runner = worker.engine.model_runner
            runner.net_pipeline_worker = worker
            runner.net_pipeline_current = frozenset()
            runner.net_pipeline_events = deque(maxlen=64)
            runner.net_pipeline_last = 0.
            graphs = getattr(worker.engine, '_autonomous_prefill_graphs', None)
            if graphs is not None:
                graphs.net_pipeline_runner = runner
                original = graphs.execute
                def wrapped(call, hidden):
                    return replayed(graphs, call, hidden, original)
                graphs.execute = wrapped
        return result

    def executed(runner, scheduled, prepared):
        worker = getattr(runner, 'net_pipeline_worker', None)
        active = bool(worker is not None and scheduled is not None
            and (folder/'measurement_begin.json').exists()
            and not getattr(runner, 'net_pipeline_current', ()))
        if not active:
            return execute(runner, scheduled, prepared)
        current = frozenset(item.request.request_id for item in scheduled.requests)
        if cfg.get('source_pipeline_candidate_only') and not any(
                transfer.tid not in current and not transfer.cancelled
                and getattr(transfer.request, 'idle_armed', False)
                for transfer in worker.connector.transfers.values()):
            # One host check per Native submission replaces per-loop control
            # polling when no other request can need boundary negotiation.
            return execute(runner, scheduled, prepared)
        runner.net_pipeline_current = current
        runner.net_pipeline_events.clear()
        try:
            return execute(runner, scheduled, prepared)
        finally:
            runner.net_pipeline_current = frozenset()
            runner.net_pipeline_events.clear()

    def replayed(graphs, call, hidden, original):
        result = original(call, hidden)
        runner = getattr(graphs, 'net_pipeline_runner', None)
        if runner is None or not runner.net_pipeline_current:
            return result
        # Reuse the Native event. Do not create, replace, or synchronize it.
        event = graphs.last_event
        if event is None:
            return result
        events = runner.net_pipeline_events
        events.append(event)
        current = runner.net_pipeline_current
        worker = runner.net_pipeline_worker
        count = worker.channel.consume_safe(current, worker.command,
            cfg.get('source_pipeline_control_limit', 4))
        now = time.monotonic()
        if now-runner.net_pipeline_last < cfg.get('source_pipeline_interval_ms', 5.)/1000:
            return result
        pending = sum(not item.query() for item in events)
        if pending < cfg.get('source_pipeline_minimum_queued_loops', 8):
            return result
        connector = worker.connector
        if connector is None or not any(t.outgoing and not t.cancelled
                and t.tid not in current for t in connector.transfers.values()):
            return result
        begin = time.monotonic()
        # The same owner thread calls the same Native transport. There is no
        # background writer, model submission, state replay, or current-request
        # mutation. Native producer events still guard every data publication.
        connector.progress()
        runner.net_pipeline_last = time.monotonic()
        records.append(dict(time=begin, event='SOURCE_PIPELINED_PROGRESS',
            current_request_ids=sorted(current), queued_native_loop_events=pending,
            control_messages=count, host_elapsed_ms=(runner.net_pipeline_last-begin)*1000))
        return result

    def finished(worker):
        try:
            return close(worker)
        finally:
            if worker.engine.stage_config.role == 'prefill':
                write(folder/f'source_pipeline_{os.getpid()}.json', dict(records=records,
                    scope='Owner-thread control and Native transfer progress during another immutable source submission. Pending CUDA events show submitted work, not kernel busy time or a guaranteed overlap interval.'))

    StageWorker.__init__, StageWorker.run = initialized, finished
    ModelRunner._execute = executed
