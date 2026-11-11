"""Avoid gate readback for exact fixed-depth Decode without enabling global Async."""
import os
from .common import ROOT, write


def fixed_contract(requests, total):
    return bool(requests) and all(r.sampling_params.min_loops == total
        and r.sampling_params.max_loops == total
        and r.sampling_params.exit_threshold == 1. for r in requests)


def install(cfg):
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    execute, submit = ModelRunner._execute, ModelRunner.submit
    update, close = StageEngine._update, StageEngine.close
    total = cfg['depth']
    counts = dict(discarded_unused_gate_results=0, logical_fixed_updates=0)

    def eligible(batch):
        return bool(batch is not None and batch.requests
            and all(batch.phase_for(item) == P.RECURRENT for item in batch.requests)
            and fixed_contract([item.request for item in batch.requests], total))

    def executed(runner, scheduled, prepared):
        result = execute(runner, scheduled, prepared)
        if eligible(scheduled):
            # Preserve every graph and model operation. Only the unused host
            # gate result is removed. Fixed R32 cannot consume an exit signal.
            counts['discarded_unused_gate_results'] += 1
            return None
        return result

    def submitted(runner, scheduled=None, *, batch):
        pending = getattr(runner, 'net_fixed_tickets', [])
        runner.net_fixed_tickets = [ticket for ticket in pending if not ticket.ready()]
        ticket = submit(runner, scheduled, batch=batch)
        if eligible(scheduled):
            assert not ticket.has_result, 'The fixed-depth path must not require a host gate value'
            assert ticket.event is not None, 'Logical progress requires a retained GPU completion event'
            runner.net_fixed_tickets.append(ticket)
        return ticket

    def updated(engine, batch, result, *, delivered=False):
        if eligible(batch):
            assert not delivered and result is None
            counts['logical_fixed_updates'] += 1
            return engine._apply_update(engine.scheduler.update_after_submission(batch,
                request_events=engine.model_runner.events))
        return update(engine, batch, result, delivered=delivered)

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'fixed_submit_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                scope='Exact fixed-depth logical submission. All model and graph operations remain. Native stream dependencies, bounded submission buffers, Attention guards, CODA collection, and KV lifetime remain enabled. Global Async is not enabled.'))
    ModelRunner._execute, ModelRunner.submit = executed, submitted
    StageEngine._update, StageEngine.close = updated, closed
