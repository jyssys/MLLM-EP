"""Extend the existing reserve when logical Decode progress leads the GPU."""
from collections import defaultdict


def additional_pending_loops(rows, already_reserved=1):
    pending = defaultdict(int)
    for request_id, loops in rows:
        if not isinstance(loops, int) or loops < 1:
            raise ValueError('A pending recurrent group must contain explicit positive Loops')
        pending[request_id] += loops
    # The qualified single-loop formula already retains one pending Loop.
    return max(0, max(pending.values(), default=0) - already_reserved)


def install(cfg):
    from ..native_loop_2d_oracle import native_path
    from vllm_rlt.request import RequestPhase as P
    connector = native_path.ACTIVE_CONNECTOR
    previous = connector.deadline_margin
    assert cfg.get('device_decode_costs'), 'Grouped pending work requires a measured per-loop reserve'

    def margin(owner):
        value = previous(owner)
        pending = []
        seen = set()
        native_pending = False
        for ticket in owner.engine._inflight:
            if ticket.ready():
                continue
            batch = ticket.batch
            if batch.phase != P.RECURRENT:
                continue
            seen.add(id(ticket))
            native_pending = True
            loops = batch.loop_count or 1
            pending.extend((item.request.request_id, loops) for item in batch.requests)
        for ticket in getattr(owner.engine.model_runner, 'net_fixed_tickets', []):
            if id(ticket) in seen or ticket.ready():
                continue
            batch = ticket.batch
            pending.extend((item.request.request_id, batch.loop_count or 1) for item in batch.requests)
        if pending and not owner.active_decode():
            return 0.
        extra = additional_pending_loops(pending, int(native_pending))
        return max(0., value-extra*cfg['device_decode_costs']['loop_p99_ms'])

    connector.deadline_margin = margin
