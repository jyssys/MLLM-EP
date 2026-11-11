"""Submit ready Decode before independent prompt work. Do not preempt kernels."""
import os
from .common import ROOT, write


def choose_decode(owner, phases):
    for phase in phases:
        if owner.has_ready(phase):
            batch = owner._take(phase)
            if batch is not None:
                return batch
    return None


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    initialize, close = StageEngine.__init__, StageEngine.close
    counts = dict(decode_first_batches=0, native_policy_fallbacks=0)

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        if engine.stage_config.role != 'decode':
            return result
        assert cfg.get('deferred_coda_collection') and cfg.get('preserve_decode_cohort')
        assert cfg.get('independent_prompt_buffers')
        policy = engine.scheduler.policy
        previous = policy.schedule

        def scheduled(owner, *, prefer_recurrent=False):
            batch = choose_decode(owner, (P.PRELUDE, P.CODA, P.RECURRENT))
            if batch is not None:
                counts['decode_first_batches'] += 1
                return batch
            counts['native_policy_fallbacks'] += 1
            return previous(owner, prefer_recurrent=prefer_recurrent)
        policy.schedule = scheduled
        return result

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT/'raw'/cfg['name']/f'explicit_decode_priority_{os.getpid()}.json', dict(
                role=engine.stage_config.role, **counts,
                scope='Ready Decode batches precede prompt submission. The delayed collector still permits independent prompt work while a Decode cohort waits for real GPU output. No running kernel is preempted.'))
    StageEngine.__init__, StageEngine.close = initialized, closed
