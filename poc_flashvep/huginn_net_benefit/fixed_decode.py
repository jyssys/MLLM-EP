"""Group exact fixed-depth Decode work through existing multi-loop execution."""
import os
from .common import ROOT, write


def group_limit(requests, total, cap):
    if not requests or cap < 1:
        raise ValueError('A positive group requires at least one request')
    for request in requests:
        params = request.sampling_params
        if params.min_loops != total or params.max_loops != total or params.exit_threshold != 1.:
            raise ValueError('Grouped Decode requires the exact fixed-depth contract')
        if not 0 <= request.loops_done < total:
            raise ValueError('The Decode frontier is outside the recurrent range')
    return min(cap, min(total-r.loops_done for r in requests))


def advance_before_native_update(request, loops, total):
    group_limit([request], total, loops)
    if loops < 1 or request.loops_done + loops > total:
        raise ValueError('Grouped Decode would duplicate or exceed recurrent work')
    # The unchanged Native transition records the last Loop and selects CODA.
    # Earlier Loops already executed in the same explicit multi-loop batch.
    request.loops_done += loops - 1


def install(cfg):
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import RequestPhase as P
    from vllm_rlt.engine.stage_engine import StageEngine
    original_plan = Scheduler._planned_loop_count
    total, cap = cfg['depth'], cfg['decode_loop_grouping']
    counts = dict(grouped_submissions=0, explicitly_grouped_recurrent_rows=0)

    def planned(scheduler, items):
        if items and all(item.phase == P.RECURRENT for item in items):
            return group_limit([item.request for item in items], total, cap)
        return original_plan(scheduler, items)
    Scheduler._planned_loop_count = planned

    for name in ('update_from_output', 'update_after_submission'):
        original = getattr(Scheduler, name)
        def updated(scheduler, batch, *args, _original=original, **kwargs):
            loops = batch.loop_count or 0
            if loops > 1 and all(batch.phase_for(i) == P.RECURRENT for i in batch.requests):
                seen = set()
                for item in batch.requests:
                    request = item.request
                    if scheduler.requests.get(request.request_id) is not request or request.phase == P.FINISHED:
                        continue
                    if request.request_id in seen:
                        raise RuntimeError('Duplicate request in a grouped Decode batch')
                    seen.add(request.request_id)
                    if item.depth is None or item.depth != request.loops_done:
                        raise RuntimeError('Stale Decode depth in an explicit grouped transition')
                    advance_before_native_update(request, loops, total)
                    counts['explicitly_grouped_recurrent_rows'] += loops
                counts['grouped_submissions'] += 1
            return _original(scheduler, batch, *args, **kwargs)
        setattr(Scheduler, name, updated)
    close = StageEngine.close
    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'fixed_decode_grouping_{os.getpid()}.json', dict(
                role=engine.stage_config.role, cap=cap, **counts,
                scope='Explicit fixed-depth multi-loop batches. Every Loop uses the existing model, Attention, graph replay, and KV plane. Gate-based early exit is prohibited.'))
    StageEngine.close = closed
