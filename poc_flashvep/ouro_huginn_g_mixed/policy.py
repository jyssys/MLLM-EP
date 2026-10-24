"""Choose a pure prefill group or an unchanged one-loop Mixed submission."""
from dataclasses import replace
import time

class FixedHybrid:
    def __init__(self, native, group):
        self.native, self.group = native, group
        self.decisions = []
    def __getattr__(self, key): return getattr(self.native, key)
    def record_batch(self, phase): self.native.record_batch(phase)
    def schedule(self, scheduler, *, prefer_recurrent=False):
        from vllm_rlt.request import RequestPhase as P
        selector_start=time.perf_counter_ns()
        decode = scheduler.has_ready(P.RECURRENT)
        prompt = scheduler.has_ready(P.PREFILLING_RECURRENT)
        old = scheduler.config
        # The immutable scheduled span records its actual group before restore.
        group = 1 if decode and prompt else self.group
        scheduler.config = replace(old, prefill_loop_group_size=group)
        selector_before_native_ns=time.perf_counter_ns()-selector_start
        start = time.perf_counter_ns()
        try:
            result = self.native.schedule(scheduler, prefer_recurrent=prefer_recurrent)
        finally:
            scheduler.config = old
        self.decisions.append(dict(host_s=time.perf_counter(), decode_ready=decode,
            prompt_ready=prompt, requested_G=group,
            selector_before_native_ns=selector_before_native_ns,
            native_policy_interval_ns=time.perf_counter_ns()-start))
        return result

def attach(engine, arm, group):
    policy = engine.scheduler.policy
    if arm in ('hybrid', 'event'):
        policy = FixedHybrid(policy, group if arm == 'hybrid' else 1)
        engine.scheduler.policy = policy
    if arm == 'fair':
        from ..ouro_budget_joint_cg.fairness import attach as fair
        policy = fair(engine)
    return policy
