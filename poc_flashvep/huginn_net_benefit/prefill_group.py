"""Bound prompt groups without changing fixed-depth Decode grouping."""

def bounded_count(value, phases, allowed, limit):
    return min(value, limit) if phases and phases <= allowed else value


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    initialize = StageEngine.__init__
    limit = cfg.get('D_prefill_loop_cap', cfg['depth'])
    assert type(limit) is int and 1 <= limit <= cfg['depth']
    allowed = {P.PREFILLING_RECURRENT}
    if cfg.get('D_prefill_cap_includes_prelude'):
        allowed.add(P.PREFILLING_PRELUDE)
    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        if engine.stage_config.role != 'decode':
            return result
        original = engine.scheduler._planned_loop_count
        def planned(items):
            value = original(items)
            return bounded_count(value, {item.phase for item in items}, allowed, limit)
        engine.scheduler._planned_loop_count = planned
        token_cap = cfg.get('D_prefill_token_cap')
        if token_cap is not None:
            assert type(token_cap) is int and 1 <= token_cap <= cfg['length']
            make = engine.scheduler._make_scheduled_request
            def bounded(request, phase, token_budget):
                if phase in (P.PREFILLING_PRELUDE, P.PREFILLING_RECURRENT):
                    token_budget = min(token_budget, token_cap)
                return make(request, phase, token_budget)
            engine.scheduler._make_scheduled_request = bounded
        return result
    StageEngine.__init__ = initialized
