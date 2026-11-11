"""Bounded Native progress bursts. Native wire and ownership checks stay active."""
import os
import time
from .common import ROOT, write


def bounded_progress(step, pending, limit, seconds, clock=time.monotonic):
    if type(limit) is not int or limit < 1 or seconds < 0:
        raise ValueError('A progress burst requires a positive step limit and a nonnegative time budget')
    start, calls = clock(), 0
    result = None
    while calls < limit:
        before = pending()
        value = step()
        if calls == 0:
            result = value
        calls += 1
        after = pending()
        if not after or after >= before or clock()-start >= seconds:
            break
    return result, calls


def install(cfg):
    from ..native_loop_2d_oracle import native_path
    connector = native_path.ACTIVE_CONNECTOR
    original = connector.progress
    counts = dict(calls=0, native_progress_calls=0, maximum_burst_steps=0)

    def progress(owner):
        if owner.engine.stage_config.role != 'prefill':
            return original(owner)
        def queued():
            return sum(len(t.chunks) for t in owner.transfers.values()
                if t.outgoing and not t.cancelled)
        result, calls = bounded_progress(lambda: original(owner), queued,
            cfg.get('wire_progress_steps', 32), cfg.get('wire_progress_budget_ms', 2.)/1000)
        counts['calls'] += 1
        counts['native_progress_calls'] += calls
        counts['maximum_burst_steps'] = max(counts['maximum_burst_steps'], calls)
        return result
    connector.progress = progress
    from vllm_rlt.engine.stage_engine import StageEngine
    close = StageEngine.close
    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT/'raw'/cfg['name']/f'wire_progress_burst_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                scope='Bounded repeated Native progress. Native producer events, in-flight byte credits, fragment counts, ACK, and source retention remain unchanged. The time budget is checked between Native calls. It cannot interrupt a blocking Native API.'))
    StageEngine.close = closed
