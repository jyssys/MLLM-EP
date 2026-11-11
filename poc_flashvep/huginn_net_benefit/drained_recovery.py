"""Allow one smaller retry after a reservation miss and a drained Decode epoch."""
from dataclasses import dataclass, field, replace
import os
import time
from .common import ROOT, write


@dataclass
class Retry:
    ceiling: int | None = None
    last_miss: float | None = None
    attempts: int = 0
    seen_misses: set = field(default_factory=set)

    def miss(self, work, now):
        cap = max(0, work//2)
        self.ceiling = cap if self.ceiling is None else min(cap, self.ceiling)
        self.last_miss = now

    def ready(self, now, cooldown_ms, reason, active, owned, leases, pending):
        return bool(self.attempts == 0 and self.ceiling and self.last_miss is not None
            and reason == 'COMPLETION_RESERVATION_MISSED'
            and 1000*(now-self.last_miss) >= cooldown_ms
            and not (active or owned or leases or pending))


def install(cfg):
    """Retain the Native offer, commit, and final release state machines."""
    from ..native_loop_2d_oracle import native_path
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.engine.stage_engine import StageEngine
    connector_class = native_path.ACTIVE_CONNECTOR
    method = cfg['intervention_method']
    assert method in ('dynamic_loop', 'dynamic_tile', 'dynamic_token')
    assert cfg.get('completion_reservations') and cfg.get('predictive_admission')
    token = method == 'dynamic_token'
    cooldown = 2*cfg['completion_deadline_ms']
    folder = ROOT/'raw'/cfg['name']
    records = []
    original_progress = connector_class.progress
    original_plan = connector_class.completion_plan

    def plan(owner, remaining):
        result = original_plan(owner, remaining)
        retry = getattr(owner, 'net_reservation_retry', None)
        if retry is not None and retry.ceiling is not None and remaining > retry.ceiling:
            return replace(result, feasible=False, reason='RECOVERY_SUFFIX_CAP')
        return result

    def progress(owner):
        retry = getattr(owner, 'net_reservation_retry', None)
        if retry is None:
            return original_progress(owner)
        known_misses = len(owner.completion_missed)
        measured = owner.measurement_started
        result = original_progress(owner)
        now = time.monotonic()
        if measured and len(owner.completion_missed) > known_misses:
            fresh = owner.completion_missed - retry.seen_misses
            for tid in sorted(fresh):
                transfer = owner.transfers.get(tid)
                work = (len(transfer.request.prompt_token_ids)-transfer.prefix if token
                        else cfg['depth']-transfer.cut) if transfer is not None else 0
                retry.miss(work, now)
                records.append(dict(time=now, event='MEASURED_COMPLETION_MISS',
                    tid=tid, failed_work=work, future_ceiling=retry.ceiling,
                    previous_retries=retry.attempts))
        # Do not allow startup hashing failures to spend the measured retry.
        if len(owner.completion_missed) > known_misses:
            retry.seen_misses.update(owner.completion_missed)
        if measured and owner.suspended and retry.ready(now, cooldown, owner.suspension_reason,
                owner.active_decode(), owner.committed_loans, owner.leases,
                owner.pending_quantum_record or (owner.last_quantum_event is not None
                    and not owner.last_quantum_event.query())):
            retry.attempts += 1
            owner.suspended = False
            owner.suspension_reason = None
            owner.credit = True
            owner.last_advertised = 0.
            records.append(dict(time=now, event='DRAINED_RESERVATION_RETRY',
                future_ceiling=retry.ceiling, cooldown_ms=cooldown,
                live_decode=0, live_committed=0, live_offer_leases=0))
        return result

    connector_class.progress = progress
    connector_class.completion_plan = plan
    initialize, close = StageWorker.__init__, StageEngine.close

    def initialized(worker, *args, **kwargs):
        result = initialize(worker, *args, **kwargs)
        if worker.engine.stage_config.role != 'decode':
            return result
        owner = worker.connector
        owner.net_reservation_retry = retry = Retry()
        notify = owner.notify

        def notifying(kind, **fields):
            if kind == 'idle_capacity' and retry.ceiling is not None:
                fields['recovery_remaining_ceiling'] = retry.ceiling
            return notify(kind, **fields)
        owner.notify = notifying
        return result

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(folder/f'drained_completion_recovery_{os.getpid()}.json', dict(
                role=engine.stage_config.role, records=records,
                scope=('At most one measured retry after a completion miss. No live '
                    'Decode, accepted continuation, offer lease, or pending quantum may '
                    'remain. Subsequent offers and commits retain all Native checks '
                    'and a smaller suffix ceiling. No SLO or reservation deadline changes.')))

    StageWorker.__init__, StageEngine.close = initialized, closed
