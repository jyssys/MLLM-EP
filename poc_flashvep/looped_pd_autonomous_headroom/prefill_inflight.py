"""Observe original submission phases and device-event readiness on P."""
import os
import time
from .common import ROOT, append


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    previous = StageEngine._step
    clocks = {}
    def step(self, *args, **kwargs):
        now = time.monotonic()
        if self.stage_config.role == 'prefill' and now-clocks.get(id(self), 0.) >= .05:
            clocks[id(self)] = now
            tickets = []
            for ticket in self._inflight:
                tickets.append(dict(submission_phase=str(ticket.batch.phase),
                    ready=bool(ticket.ready()),
                    requests=[dict(request_id=item.request.request_id,
                                   current_phase=str(item.request.phase)) for item in ticket.batch.requests]))
            append(ROOT/'raw'/cfg['name']/f'P_submission_readiness_{os.getpid()}.jsonl',
                   dict(time=now, tickets=tickets,
                        scope='Native event readiness. A pending event alone does not prove GPU compute saturation.'))
        return previous(self, *args, **kwargs)
    StageEngine._step = step
