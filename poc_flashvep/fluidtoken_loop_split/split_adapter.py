"""Exact static plans over existing connectors, independent of token budget."""
import time


def static_plan(cfg, prompt_tokens):
    depth = cfg['depth']
    if cfg.get('dynamic'):
        raise NotImplementedError('Dynamic runtime requires a measured calibration and a separate gate')
    if cfg['axis'] == 'loop':
        cut, prefix = cfg['cut'], prompt_tokens
    elif cfg['axis'] == 'token':
        prefix = cfg.get('prefix_tokens')
        if prefix is None:
            fraction = cfg.get('prefix_fraction', 1.)
            if not 0 <= fraction <= 1:
                raise ValueError('Illegal prefix fraction')
            prefix = int(prompt_tokens * fraction)
        cut = depth if prefix else 0
    else:
        raise ValueError('Unknown partition axis')
    if not isinstance(prefix, int) or not 0 <= prefix <= prompt_tokens:
        raise ValueError('Illegal exact token prefix')
    if not isinstance(cut, int) or not 0 <= cut <= depth:
        raise ValueError('Illegal recurrent cut')
    return cut, prefix


def install(cfg):
    if not cfg.get('split_evaluation'):
        return
    if not cfg.get('wavefront') or cfg.get('scheduler_mode', 'refill') != 'refill':
        raise ValueError('The qualified connectors require staged separate-phase scheduling')
    if cfg.get('dynamic'):
        from .dynamic_adapter import install as dynamic_install
        dynamic_install(cfg)
        return
    from .common import ROOT, prior
    from ..adaptive_loop_token_pd_2gpu import policy
    from vllm_rlt.pd.engine import PDEngine
    inherited = policy.install
    def initialize(engine, contract):
        inherited(engine, contract)
        def choose(request_id, tokens):
            cut, prefix = static_plan(contract, len(tokens))
            engine._arc_cuts[request_id] = cut
            engine._ltp_prefixes[request_id] = prefix
            prior.append(ROOT / 'raw' / contract['name'] / 'online_decisions.jsonl', dict(
                event='STATIC_EXACT_BOUNDARY', request_id=request_id,
                time=time.monotonic(), axis=contract['axis'], cut=cut,
                prefix_k=prefix, prompt_tokens=len(tokens),
                execution_chunk=contract['chunk'], online=False))
        engine.ltp_choose = choose
    policy.install = initialize
    add_request = PDEngine.add_request
    def add(self, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        self.ltp_choose(request_id, prompt_token_ids)
        return add_request(self, request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
    PDEngine.add_request = add
