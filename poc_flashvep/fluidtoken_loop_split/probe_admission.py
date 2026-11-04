"""Optionally defer independent probes at an observed decode-load limit.

The inherited probe still owns its request, state, memory guard, and full work.
This hook does not defer background serving requests or change their outputs.
"""
import time


class AdmissionConfig(dict):
    def __init__(self, cfg, emit):
        super().__init__(cfg)
        self.engine = None
        self.emit = emit
        self.last_log = 0.

    def get(self, key, default=None):
        value = super().get(key, default)
        if key != 'D_probe_count' or self.engine is None:
            return value
        if self.get('D_local_probe_output_pacing') and any(
                r.request_id.startswith('autonomous_probe_')
                for r in self.engine.scheduler.requests.values()):
            return 0
        scope = self.get('D_probe_admission_scope', 'recurrent_phase')
        if scope not in ('recurrent_phase', 'all_decode_phases'):
            raise ValueError('Unknown probe admission scope')
        phases = ('recurrent',) if scope == 'recurrent_phase' else ('prelude', 'recurrent', 'coda', 'speculative')
        active = sum(not r.request_id.startswith('autonomous_probe_')
            and str(getattr(r.phase, 'value', r.phase)).lower() in phases
            for r in self.engine.scheduler.requests.values())
        allowed = active <= self['D_probe_max_active_decode']
        now = time.monotonic()
        if now - self.last_log >= .25:
            self.last_log = now
            self.emit(dict(time=now, active_decode=active, allowed=allowed,
                           maximum=self['D_probe_max_active_decode'], scope=scope))
        return value if allowed else 0


def prepare(cfg):
    if not cfg.get('D_probe_max_active_decode'):
        return
    from ..looped_pd_autonomous_headroom import decode_probe
    original = decode_probe.install

    def install(config):
        from .common import ROOT, prior
        from vllm_rlt.engine.stage_engine import StageEngine
        controlled = AdmissionConfig(config, lambda row: prior.append(
            ROOT / 'raw' / config['name'] / 'probe_admission.jsonl', row))
        original(controlled)
        step = StageEngine.step

        def measured(engine):
            controlled.engine = engine
            try:
                return step(engine)
            finally:
                controlled.engine = None
        StageEngine.step = measured

    decode_probe.install = install
