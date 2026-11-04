"""Retain the calibrated policy and isolate startup and branch diagnostics."""
import time
from .common import ROOT, prior

EXTRA_SIGNATURE = ('observe_service_timing', 'startup_relative_token_boundary', 'prime_dynamic_split_geometries',
                   'contiguous_prompt_loop_burst')


def validate_branch_scope(cfg):
    if (cfg.get('phase') != 'EXACT_DYNAMIC_BRANCH_DIAGNOSTIC'
            or not cfg.get('state_probe') or not cfg.get('verify_transfer')):
        raise ValueError('Forced branches require explicit heavy correctness diagnostics')
    if not cfg.get('calibration_serial') and cfg.get('count', 0) < 32:
        raise ValueError('The initial forced branch uses serial N8 before concurrent N32')


def signature():
    from ..fluidtoken_loop_split import dynamic_adapter
    dynamic_adapter.SIGNATURE = tuple(dict.fromkeys(dynamic_adapter.SIGNATURE + EXTRA_SIGNATURE))
    return dynamic_adapter.SIGNATURE


def prepare(cfg):
    if not cfg.get('dynamic'):
        return
    signature()
    from ..fluidtoken_loop_split import dynamic_adapter
    from .online_adapter import install as service_cost_policy
    dynamic_adapter.install = service_cost_policy
    from ..adaptive_loop_token_pd_2gpu import policy
    inherited = policy.install

    def initialized(engine, contract):
        inherited(engine, contract)
        # This sender already retains native transport and lifecycle observers.
        engine._serving_base_sender = engine._send

    policy.install = initialized


def finalize(cfg):
    if not cfg.get('dynamic'):
        return
    from ..adaptive_loop_token_pd_2gpu import policy
    from ..fluidtoken_loop_split.dynamic_adapter import load_table
    table, costs, full = load_table(cfg)
    inherited = policy.install
    marker = ROOT / 'raw' / cfg['name'] / 'measurement_begin.json'
    diagnostic = cfg.get('qualification_force_partial_key')
    reject = cfg.get('qualification_reject_frontier', False)
    if diagnostic is not None or reject:
        validate_branch_scope(cfg)
        assert diagnostic in {row.key for row in costs if row.key != full.key}

    def initialized(engine, contract):
        inherited(engine, contract)
        send, receive = engine._send, engine._message
        startup_index, startup_tids = [0], {}

        def boundary(w, key):
            engine._arc_cuts[w.request.request_id] = key if cfg['axis'] == 'loop' else cfg['depth']
            engine._ltp_prefixes[w.request.request_id] = (len(w.request.prompt_token_ids)
                if cfg['axis'] == 'loop' else key)

        def sending(destination, kind, **fields):
            w = engine.transfers.get(fields.get('tid'))
            if kind == 'reserve' and w is not None:
                if not marker.exists():
                    rid = w.request.request_id
                    assert rid.startswith(('r-1_q', 'recovery_warm_')), 'Unexpected unmeasured dynamic request'
                    length = len(w.request.prompt_token_ids)
                    key = cfg['depth'] if cfg['axis'] == 'loop' else length
                    if cfg.get('prime_dynamic_split_geometries'):
                        source_key = costs[startup_index[0] % len(costs)].key
                        startup_index[0] += 1
                        key = source_key if cfg['axis'] == 'loop' else length * source_key // cfg['length']
                        startup_tids[w.tid] = key
                        prior.append(ROOT / 'raw' / cfg['name'] / 'online_decisions.jsonl', dict(
                            event='WARMUP_CALIBRATED_GEOMETRY', request_id=rid,
                            calibrated_key=source_key, startup_key=key, prompt_tokens=length,
                            online=False, time=time.monotonic(),
                            scope='Valid startup requests only. No measured cut or SLO threshold is changed.'))
                    boundary(w, key)
                    return engine._serving_base_sender(destination, kind, **fields)
                if diagnostic is not None:
                    boundary(w, diagnostic)
                    prior.append(ROOT / 'raw' / cfg['name'] / 'online_decisions.jsonl', dict(
                        event='FORCED_EXACT_BRANCH_DIAGNOSTIC', request_id=w.request.request_id,
                        key=diagnostic, online=False, time=time.monotonic(),
                        scope='Correctness-only serial branch exercise. Not online-policy performance.'))
                    return engine._serving_base_sender(destination, kind, **fields)
            return send(destination, kind, **fields)

        def received(peer, message):
            if not marker.exists() and message['kind'] == 'frontier_ready':
                w = engine.transfers.get(message.get('tid') or message.get('request_id'))
                assert w is not None and w.tid in startup_tids
                engine._send(w.p, 'frontier_decision', tid=w.tid, accept=True,
                             reason='QUALIFIED_STARTUP_GEOMETRY_ONLY')
                return
            if diagnostic is not None and message['kind'] == 'frontier_ready':
                w = engine.transfers.get(message.get('tid') or message.get('request_id'))
                assert w is not None
                if reject:
                    boundary(w, full.key)
                prior.append(ROOT / 'raw' / cfg['name'] / 'online_decisions.jsonl', dict(
                    event='FORCED_EXACT_FRONTIER_REJECTION' if reject else 'FORCED_EXACT_FRONTIER_ACCEPTANCE',
                    request_id=w.request.request_id,
                    online=False, time=time.monotonic(),
                    scope='Forced correctness branch only. Concurrent N32 can violate latency SLOs and is not online-policy performance. Rejection must finish all remaining work on P.'))
                engine._send(w.p, 'frontier_decision', tid=w.tid, accept=not reject,
                             reason='EXACT_BRANCH_DIAGNOSTIC_REJECTION' if reject else 'EXACT_BRANCH_DIAGNOSTIC_ACCEPTANCE')
                return
            return receive(peer, message)

        engine._send, engine._message = sending, received

    policy.install = initialized
