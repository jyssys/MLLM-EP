"""Admission-time FluidToken-style pressure policy over calibrated exact cuts.

This is not FluidPD's full per-P-iteration queue-budget algorithm.
Both axes use the same observed-state cost selector and frontier safety check.
"""
import json
import time
from math import isfinite
from pathlib import Path
from .common import ROOT, digest, prior
from .decision_model import QueuedRequest, SplitCost, prefill_pressure, decode_pressure, queue_relief_choice

SIGNATURE = ('model', 'model_revision', 'depth', 'length', 'chunk', 'token_budget',
             'D_token_budget', 'P_slots', 'D_slots', 'graph_rows', 'P_prefill_graph_rows',
             'D_prefill_graph_rows', 'D_kv_blocks', 'kv_blocks', 'wavefront', 'handoff_mode',
             'route_campaign_evidence_async', 'D_local_probe_output_pacing',
             'token_suffix_origin_candidate')


def load_table(cfg):
    path = Path(cfg['dynamic_cost_table'])
    assert path.is_relative_to(ROOT), 'Calibration must belong to this campaign'
    assert digest(path) == cfg['dynamic_cost_table_SHA256']
    table = json.loads(path.read_text())
    assert table['qualified'] and table['axis'] == cfg['axis']
    assert table['execution_signature'] == {key: cfg.get(key) for key in SIGNATURE}
    frozen = json.loads((ROOT / 'configs/frozen_SLO.json').read_text())[cfg['model'] + '_L' + str(cfg['length'])]
    assert table['TTFT_limit_s'] == frozen['TTFT_ms'] / 1000
    assert table['ITL_limit_s'] == frozen['ITL_ms'] / 1000
    assert table['TPOT_margin_s'] == .9 * frozen['TPOT_ms'] / 1000
    assert table['calibration_evidence']
    for evidence, expected in table['calibration_evidence'].items():
        artifact = Path(evidence)
        assert artifact.is_relative_to(ROOT) and digest(artifact) == expected
    assert isinstance(table['safe_decode_batch'], int) and table['safe_decode_batch'] > 0
    assert isfinite(table['baseline_token_interval_s']) and table['baseline_token_interval_s'] > 0
    costs = [SplitCost(**row) for row in table['costs']]
    assert all(all(isfinite(v) and v >= 0 for v in
        (row.p_service_s, row.d_continuation_s, row.transfer_exposed_s))
        and isinstance(row.d_extra_blocks, int) and row.d_extra_blocks >= 0 for row in costs)
    full_key = cfg['depth'] if cfg['axis'] == 'loop' else cfg['length']
    assert all(isinstance(row.key, int) and 0 < row.key <= full_key for row in costs), 'Local-Prefill endpoint needs a separate routing gate'
    assert len({row.key for row in costs}) == len(costs)
    full = next(row for row in costs if row.key == full_key)
    assert full.d_continuation_s == 0
    return table, costs, full


def install(cfg):
    table, costs, full = load_table(cfg)
    from ..adaptive_loop_token_pd_2gpu import policy
    from vllm_rlt.pd.engine import PDEngine
    inherited = policy.install

    def initialize(engine, contract):
        inherited(engine, contract)
        send, receive = engine._send, engine._message
        last_token, recent_intervals = {}, []
        measurement_started = [False]
        marker = ROOT / 'raw' / contract['name'] / 'measurement_begin.json'
        log = ROOT / 'raw' / contract['name'] / 'online_decisions.jsonl'

        def set_key(rid, key):
            engine._arc_cuts[rid] = key if contract['axis'] == 'loop' else contract['depth']
            engine._ltp_prefixes[rid] = len(engine.requests[rid].prompt_token_ids) if contract['axis'] == 'loop' else key

        def initial(rid, tokens):
            # The real decision occurs after native KV reservation, not at future arrival time.
            engine._arc_cuts[rid] = contract['depth']
            engine._ltp_prefixes[rid] = len(tokens)
        engine.ltp_choose = initial

        def observed(current):
            if not measurement_started[0] and marker.exists():
                measurement_started[0] = True
                last_token.clear()
                recent_intervals.clear()
            now = time.monotonic()
            transfers = [w for w in engine.transfers.values() if not w.finished and not w.cancelled]
            # An activated request can consume D resources before its first output.
            active = [w for w in transfers if w.phase == 'decode']
            d = engine.peers[current.d]
            active_blocks = sum(w.d_reserved_blocks for w in active if w.d == current.d)
            committed = d.reserved_blocks
            pressure = decode_pressure(active_blocks, committed - active_blocks,
                d.client.info['num_blocks'], len(active), table['safe_decode_batch'])
            queued = [w for w in transfers if w.phase in ('waiting', 'reserving', 'prefill')]
            predictions = []
            p_remaining = 0.
            cumulative = 0.
            later_deficit = 0.
            for w in sorted(queued, key=lambda item: item.created):
                key = (engine._arc_cuts.get(w.request.request_id, full.key) if contract['axis'] == 'loop'
                       else engine._ltp_prefixes.get(w.request.request_id, full.key))
                cost = next((c for c in costs if c.key == key), full).p_service_s
                if w.phase == 'prefill' and 'prefill_started' in w.timings:
                    cost = max(0., cost - (now - w.created - w.timings['prefill_started']))
                predictions.append(QueuedRequest(w.created, cost, table['TTFT_limit_s']))
                cumulative += cost
                if w.created > current.created:
                    later_deficit = max(later_deficit,
                        cumulative - (w.created + table['TTFT_limit_s'] - now))
                if w is not current and w.created <= current.created:
                    p_remaining += cost
            prior_values = sorted(recent_intervals[-64:])
            recent = prior_values[int(.95 * (len(prior_values) - 1))] if prior_values else 0.
            interval = max(table['baseline_token_interval_s'], recent)
            pending = 0
            for w in transfers:
                if w is current:
                    continue
                key = (engine._arc_cuts.get(w.request.request_id, full.key) if contract['axis'] == 'loop'
                       else engine._ltp_prefixes.get(w.request.request_id, full.key))
                pending += key != full.key and not w.request.generated_token_ids
            return dict(time=now, PPI=prefill_pressure(predictions, now), DPI=pressure,
                active_decode=len(active), p_queue_service_s=p_remaining,
                later_queue_deficit_s=later_deficit,
                observed_token_interval_s=interval,
                max_decode_age_s=max((now - last_token.get(w.tid, now) for w in active), default=0.),
                pending_continuations=pending, current_reserved_blocks=current.d_reserved_blocks)

        def logged(event, w, state, key, reason, elapsed):
            prior.append(log, dict(event=event, time=time.monotonic(), request_id=w.request.request_id,
                axis=contract['axis'], selected_key=key, state=state, reason=reason,
                policy_host_seconds=elapsed, calibration_SHA256=contract['dynamic_cost_table_SHA256'],
                online=True, scope='Observed admission/frontier state only. No future arrivals or actual future output lengths.'))

        def sending(destination, kind, **fields):
            w = engine.transfers.get(fields.get('tid'))
            if kind == 'reserve' and w is not None:
                begin = time.perf_counter()
                state = observed(w)
                selected, reason = full, 'FULL_PD_DECODE_LOAD_OR_CREDIT'
                if state['DPI']['load'] < 1 and state['current_reserved_blocks'] > 0:
                    selected, reason = queue_relief_choice(costs, full, state['later_queue_deficit_s'],
                        state['observed_token_interval_s'] if state['active_decode'] else 0.,
                        state['max_decode_age_s'], table['ITL_limit_s'], table['TPOT_margin_s'],
                        state['DPI']['free_blocks'], state['PPI'], bool(state['pending_continuations']))
                set_key(w.request.request_id, selected.key)
                logged('ADMISSION_CALIBRATED_PLAN', w, state, selected.key, reason, time.perf_counter() - begin)
            return send(destination, kind, **fields)

        def received(peer, message):
            w = engine.transfers.get(message.get('tid') or message.get('request_id'))
            if message['kind'] == 'frontier_ready':
                if w is None:
                    raise RuntimeError('Unknown recurrent/token frontier')
                begin = time.perf_counter()
                state = observed(w)
                key = message['arc_cut'] if contract['axis'] == 'loop' else message['prefix_k']
                cost = next(c for c in costs if c.key == key)
                slack = (min(table['ITL_limit_s'] - state['max_decode_age_s'], table['TPOT_margin_s'])
                    - state['observed_token_interval_s']) if state['active_decode'] else float('inf')
                accept = (not state['pending_continuations'] and state['DPI']['load'] < 1 and
                    state['current_reserved_blocks'] > 0 and cost.d_extra_blocks <= state['DPI']['free_blocks'] and
                    cost.d_continuation_s <= slack)
                reason = 'CALIBRATED_FRONTIER_ACCEPT' if accept else 'SAFE_FULL_PD_FALLBACK'
                if not accept:
                    set_key(w.request.request_id, full.key)
                logged('FRONTIER_CALIBRATED_RECHECK', w, state, key, reason, time.perf_counter() - begin)
                engine._send(w.p, 'frontier_decision', tid=w.tid, accept=accept, reason=reason)
                return
            if w is not None and message['kind'] == 'output':
                now = time.monotonic()
                if w.tid in last_token:
                    recent_intervals.append(now - last_token[w.tid])
                last_token[w.tid] = now
            return receive(peer, message)
        engine._send, engine._message = sending, received
    policy.install = initialize

    add = PDEngine.add_request
    def adding(engine, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        engine.ltp_choose(request_id, prompt_token_ids)
        return add(engine, request_id, prompt_token_ids, sampling_params, trace_id=trace_id)
    PDEngine.add_request = adding
