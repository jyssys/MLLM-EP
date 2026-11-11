"""Count actual pending source producer events in the synchronous Runtime."""
import os
from .common import ROOT, write


def pending_source_ids(rows, native_ids=()):
    result = set(native_ids)
    for row in rows:
        if row['outgoing'] and not row['cancelled'] and row['live'] and row['event'] is not None:
            if not row['event'].query():
                result.add(row['request_id'])
    return result


def install(cfg):
    from ..native_loop_2d_oracle import native_path
    connector = native_path.ACTIVE_CONNECTOR
    original = connector.pending_P_device_ids
    counts = dict(evaluations=0, evaluations_with_additional_pending=0, maximum_additional=0)

    def pending(owner):
        native = original(owner)
        if owner.engine.stage_config.role != 'prefill':
            return native
        scheduler, runner = owner.engine.scheduler, owner.engine.model_runner
        rows = [dict(outgoing=t.outgoing, cancelled=t.cancelled,
            live=scheduler.requests.get(t.request.request_id) is t.request,
            event=runner.events.get(t.request.request_id), request_id=t.request.request_id)
            for t in owner.transfers.values()]
        result = pending_source_ids(rows, native)
        # The optional owner-thread pipeline can progress another transfer
        # before ModelRunner.submit publishes this batch's final event.
        result.update(getattr(runner, 'net_pipeline_current', ()))
        extra = len(result-set(native))
        counts['evaluations'] += 1
        counts['evaluations_with_additional_pending'] += bool(extra)
        counts['maximum_additional'] = max(counts['maximum_additional'], extra)
        return result
    connector.pending_P_device_ids = pending
    from vllm_rlt.engine.stage_engine import StageEngine
    close = StageEngine.close
    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'source_producer_events_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                scope='Native per-request producer events complement the Async inflight list. Pending means queued device work, not measured kernel saturation. The distinct-request threshold and D admission guards remain unchanged.'))
    StageEngine.close = closed
