"""Defer fixed-depth output collection without speculative next-token execution."""
import os
from .common import ROOT, write
from .fixed_submit import fixed_contract


def pending_request_ids(entries):
    return {item.request.request_id for ticket in entries for item in ticket.batch.requests}


def collect_completed(entries, native_update):
    outputs, remaining, collected = [], [], 0
    for ticket in entries:
        if ticket.ready():
            outputs.extend(native_update(ticket.batch, ticket.collect()))
            collected += 1
        else:
            remaining.append(ticket)
    return outputs, remaining, collected


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    initialize, ready, step, close = (StageEngine.__init__, StageEngine._execution_ready,
        StageEngine._step, StageEngine.close)
    counts = dict(deferred_batches=0, collected_batches=0,
        pending_request_deferrals=0, continuation_coda_dependency_deferrals=0,
        maximum_pending_batches=0, decode_retirement_deferrals=0)

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        if engine.stage_config.role == 'decode':
            assert not engine.execution_config.async_scheduling, 'This extension is not the unsupported global Huginn Async path'
            assert cfg.get('fixed_depth_submit') and cfg.get('decode_loop_grouping') == cfg['depth']
            assert engine.speculative_config is None
            engine.net_pending_coda = []
        return result

    def execution_ready(engine, request):
        entries = getattr(engine, 'net_pending_coda', None)
        if entries is not None:
            if (cfg.get('preserve_decode_cohort') and entries
                    and request.phase == P.RECURRENT):
                # The synchronous reference does not enqueue another Decode
                # core while its preceding cohort awaits token delivery.
                # Keep that coalescing rule, but allow independent prompt work.
                counts['decode_retirement_deferrals'] += 1
                engine._connector_blocked.add(request.request_id)
                return False
            if request.request_id in pending_request_ids(entries):
                counts['pending_request_deferrals'] += 1
                engine._connector_blocked.add(request.request_id)
                return False
            if request.phase == P.CODA:
                if len(entries) >= cfg.get('deferred_coda_batches', 2):
                    engine._connector_blocked.add(request.request_id)
                    return False
                connector = engine.connector
                transfer = connector.transfers.get(request.request_id) if connector else None
                continuation = (transfer is not None and transfer.request is request
                    and not transfer.outgoing and not request.generated_token_ids
                    and (transfer.cut < cfg['depth']
                        or getattr(transfer, 'prefix', len(request.prompt_token_ids)) < len(request.prompt_token_ids)))
                if continuation:
                    event = engine.model_runner.events.get(request.request_id)
                    if event is None or not event.query():
                        # Do not put a wait for low-priority prompt work at
                        # the head of the ordinary Decode boundary stream.
                        counts['continuation_coda_dependency_deferrals'] += 1
                        engine._connector_blocked.add(request.request_id)
                        return False
        return ready(engine, request)

    def stepped(engine):
        pending = getattr(engine, 'net_pending_coda', None)
        if pending is None:
            return step(engine)
        # Native update owns token delivery, stop decisions, and resource
        # release. No next token was scheduled speculatively.
        outputs, remaining, collected = collect_completed(pending, engine._update)
        counts['collected_batches'] += collected
        engine.net_pending_coda = remaining
        batch = engine._schedule()
        engine.last_schedule = batch
        if batch is None:
            if remaining or engine._waiting_for_connector():
                return outputs
            raise RuntimeError('scheduler made no progress without a retained GPU output or connector wait')
        ticket = engine.executor.submit(batch)
        if batch.phase == P.CODA:
            assert ticket.has_result and ticket.event is not None
            assert fixed_contract([item.request for item in batch.requests], cfg['depth'])
            assert not (pending_request_ids(remaining) & {item.request.request_id for item in batch.requests})
            remaining.append(ticket)
            counts['deferred_batches'] += 1
            counts['maximum_pending_batches'] = max(counts['maximum_pending_batches'], len(remaining))
            return outputs
        result = ticket.collect() if ticket.has_result else None
        outputs.extend(engine._update(batch, result))
        return outputs

    def closed(engine):
        try:
            return close(engine)
        finally:
            pending = getattr(engine, 'net_pending_coda', [])
            write(ROOT/'raw'/cfg['name']/f'deferred_coda_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                remaining_pending_batches=len(pending),
                scope='Exact fixed-depth delayed output collection. Global Async, speculative next-token work, exit decisions, and Native KV guards are unchanged. A request remains blocked in CODA until its real token result is collected.'))
    StageEngine.__init__, StageEngine._execution_ready = initialized, execution_ready
    StageEngine._step, StageEngine.close = stepped, closed
