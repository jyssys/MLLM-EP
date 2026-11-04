"""Scale only explicitly identified startup requests, not measured prompts."""
from contextvars import ContextVar
import json
import time
from .common import ROOT, prior


def startup_plan(cfg, prompt_tokens, request_id, seed, phase):
    if not (cfg.get('startup_relative_token_boundary') and cfg['axis'] == 'token'
            and not cfg.get('dynamic') and cfg.get('prime_all_graph_shapes')
            and phase.get('state') == 'RUNNING' and phase.get('kind') == 'short_D_geometry'
            and request_id.startswith('recovery_warm_short_D_geometry_')
            and seed in phase.get('seeds', []) and prompt_tokens == min(512, cfg['length'])):
        return None
    prefix = cfg.get('prefix_tokens')
    if prefix is None:
        prefix = int(cfg['length'] * cfg.get('prefix_fraction', 1.))
    if not 0 <= prefix <= cfg['length']:
        raise ValueError('Invalid primary token boundary')
    mapped = prompt_tokens * prefix // cfg['length']
    return (cfg['depth'] if mapped else 0), mapped


def install(cfg):
    if not cfg.get('startup_relative_token_boundary'):
        return
    from ..fluidtoken_loop_split import split_adapter
    from vllm_rlt.pd.engine import PDEngine
    original_plan, original_add = split_adapter.static_plan, PDEngine.add_request
    context = ContextVar('startup_request', default=None)
    marker = ROOT / 'raw' / cfg['name'] / 'native_graph_warmup_phase.json'

    def plan(contract, prompt_tokens):
        current = context.get()
        if current is not None and marker.exists():
            request_id, seed = current
            phase = json.loads(marker.read_text())
            mapped = startup_plan(contract, prompt_tokens, request_id, seed, phase)
            if mapped is not None:
                prior.append(ROOT / 'raw' / cfg['name'] / 'startup_token_boundaries.jsonl', dict(
                    request_id=request_id, seed=seed, time=time.monotonic(),
                    prompt_tokens=prompt_tokens, primary_tokens=contract['length'],
                    primary_prefix=contract.get('prefix_tokens'), startup_prefix=mapped[1],
                    scope='Explicit short graph warm-up only. Measured boundaries are unchanged.'))
                return mapped
        return original_plan(contract, prompt_tokens)

    def add(engine, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        token = context.set((request_id, getattr(sampling_params, 'seed', None)))
        try:
            return original_add(engine, request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
        finally:
            context.reset(token)

    split_adapter.static_plan, PDEngine.add_request = plan, add
