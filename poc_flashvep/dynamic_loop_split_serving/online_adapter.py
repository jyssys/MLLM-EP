"""Use measured service costs without treating reservation as execution.

This discrete admission/frontier policy is a FluidToken-style reproduction.
It is not the paper's complete continuous queue-budget optimizer.
"""
import time
from .common import ROOT, prior
from ..fluidtoken_loop_split.decision_model import (
    QueuedRequest, prefill_pressure, decode_pressure, queue_relief_choice)


def next_output_mean_budget(first_output_s, now_s, emitted_tokens, baseline_interval_s, mean_limit_s):
    """Bound added delay if the next observed output terminates the request."""
    if emitted_tokens < 1 or first_output_s is None:
        return mean_limit_s - baseline_interval_s
    if first_output_s > now_s:
        raise ValueError('An observed first output cannot be in the future')
    # After the next token there are emitted_tokens completed intervals.
    # No remaining output length, output cap, or future EOS is assumed.
    return mean_limit_s * emitted_tokens - (now_s - first_output_s) - baseline_interval_s


def select(costs, full, state, table):
    if state['DPI']['load'] >= 1 or state['current_reserved_blocks'] <= 0:
        return full, 'FULL_PD_DECODE_LOAD_OR_CREDIT'
    active = state['active_decode'] > 0
    # No existing Decode request has an ITL to protect when the batch is empty.
    # Frozen evaluation limits do not change. Recheck the batch at handoff.
    return queue_relief_choice(costs, full, state['later_queue_deficit_s'],
        state['observed_token_interval_s'] if active else 0.,
        state['max_decode_age_s'], table['ITL_limit_s'] if active else float('inf'),
        (state['observed_token_interval_s'] + state.get('request_mean_TPOT_extra_budget_s',
            table['TPOT_margin_s'] - state['observed_token_interval_s'])) if active else float('inf'),
        state['DPI']['free_blocks'], state['PPI'], bool(state['pending_continuations']))


def install(cfg):
    from ..fluidtoken_loop_split.dynamic_adapter import load_table
    table, costs, full = load_table(cfg)
    from ..adaptive_loop_token_pd_2gpu import policy
    from vllm_rlt.pd.engine import PDEngine
    inherited = policy.install

    def initialized(engine, contract):
        inherited(engine, contract)
        send, receive = engine._send, engine._message
        last_token, intervals = {}, []
        marker = ROOT / 'raw' / contract['name'] / 'measurement_begin.json'
        log = ROOT / 'raw' / contract['name'] / 'online_decisions.jsonl'
        started = [False]

        def key_for(w):
            values = engine._arc_cuts if cfg['axis'] == 'loop' else engine._ltp_prefixes
            return values.get(w.request.request_id, full.key)

        def set_key(w, key):
            engine._arc_cuts[w.request.request_id] = key if cfg['axis'] == 'loop' else cfg['depth']
            engine._ltp_prefixes[w.request.request_id] = (len(w.request.prompt_token_ids)
                if cfg['axis'] == 'loop' else key)

        def initial(rid, tokens):
            engine._arc_cuts[rid] = cfg['depth']
            engine._ltp_prefixes[rid] = len(tokens)
        engine.ltp_choose = initial

        def observe(current):
            if not started[0] and marker.exists():
                started[0] = True
                last_token.clear()
                intervals.clear()
            now = time.monotonic()
            transfers = [w for w in engine.transfers.values() if not w.finished and not w.cancelled]
            active = [w for w in transfers if w.phase == 'decode' and w.d == current.d]
            d = engine.peers[current.d]
            active_blocks = sum(w.d_reserved_blocks for w in active)
            dpi = decode_pressure(active_blocks, d.reserved_blocks - active_blocks,
                                  d.client.info['num_blocks'], len(active), table['safe_decode_batch'])
            queued = sorted((w for w in transfers if w.phase in ('waiting', 'reserving', 'prefill')),
                            key=lambda w: w.created)
            cumulative, later_deficit = 0., 0.
            predictions = []
            for w in queued:
                cost = next((c for c in costs if c.key == key_for(w)), full).p_service_s
                # Completion is an observed event. Reservation time is not a
                # GPU start, so elapsed reservation time is never subtracted.
                if 'prefill_compute_completed' in w.timings:
                    cost = 0.
                cumulative += cost
                predictions.append(QueuedRequest(w.created, cost, table['TTFT_limit_s']))
                if w.created > current.created:
                    later_deficit = max(later_deficit,
                        cumulative - (w.created + table['TTFT_limit_s'] - now))
            recent = sorted(intervals[-64:])
            interval = max(table['baseline_token_interval_s'],
                           recent[int(.95 * (len(recent) - 1))] if recent else 0.)
            mean_budgets = [next_output_mean_budget(
                w.created + w.timings['first_token'] if 'first_token' in w.timings else None,
                now, len(w.request.generated_token_ids), interval, table['TPOT_margin_s'])
                for w in active]
            pending = sum(w is not current and w.d == current.d and key_for(w) != full.key
                          and not w.request.generated_token_ids for w in transfers)
            return dict(time=now, PPI=prefill_pressure(predictions, now), DPI=dpi,
                active_decode=len(active), later_queue_deficit_s=max(0., later_deficit),
                observed_token_interval_s=interval,
                request_mean_TPOT_extra_budget_s=min(mean_budgets) if mean_budgets else None,
                mean_budget_scope='Observed first output, emitted count, and elapsed time only. Worst case: the next output terminates. No future output length is assumed.',
                max_decode_age_s=max((now - last_token[w.tid] for w in active if w.tid in last_token), default=0.),
                pending_continuations=pending, current_reserved_blocks=current.d_reserved_blocks,
                known_queue_requests=len(queued),
                unreserved_waiting_requests=sum(w.phase == 'waiting' for w in queued),
                pressure_scope='Known coordinator queue, not a claim of D-credit-clear P-ready compute pressure.')

        def record(event, w, state, cost, reason, begin):
            prior.append(log, dict(event=event, time=time.monotonic(), request_id=w.request.request_id,
                axis=cfg['axis'], selected_key=cost.key, state=state, reason=reason,
                predicted_P_service_s=cost.p_service_s, predicted_D_continuation_s=cost.d_continuation_s,
                predicted_exposed_transfer_s=cost.transfer_exposed_s,
                policy_host_seconds=time.perf_counter() - begin, online=True,
                calibration_SHA256=cfg['dynamic_cost_table_SHA256'],
                scope='Observed queue, reservation, output, and completion state only. No future arrivals or output lengths.'))

        def sending(destination, kind, **fields):
            w = engine.transfers.get(fields.get('tid'))
            if kind == 'reserve' and w is not None:
                begin = time.perf_counter()
                state = observe(w)
                cost, reason = select(costs, full, state, table)
                set_key(w, cost.key)
                record('ADMISSION_CALIBRATED_PLAN', w, state, cost, reason, begin)
            return send(destination, kind, **fields)

        def received(peer, message):
            w = engine.transfers.get(message.get('tid') or message.get('request_id'))
            if message['kind'] == 'frontier_ready':
                if w is None:
                    raise RuntimeError('Unknown exact frontier')
                begin = time.perf_counter()
                state = observe(w)
                key = message['arc_cut'] if cfg['axis'] == 'loop' else message['prefix_k']
                cost = next(c for c in costs if c.key == key)
                slack = min(table['ITL_limit_s'] - state['max_decode_age_s'] - state['observed_token_interval_s'],
                    state['request_mean_TPOT_extra_budget_s']) if state['active_decode'] else float('inf')
                accept = (not state['pending_continuations'] and state['DPI']['load'] < 1
                    and state['current_reserved_blocks'] > 0 and cost.d_extra_blocks <= state['DPI']['free_blocks']
                    and cost.d_continuation_s <= slack)
                if not accept:
                    set_key(w, full.key)
                reason = 'CALIBRATED_FRONTIER_ACCEPT' if accept else 'SAFE_FULL_PD_FALLBACK'
                record('FRONTIER_CALIBRATED_RECHECK', w, state, cost, reason, begin)
                engine._send(w.p, 'frontier_decision', tid=w.tid, accept=accept, reason=reason)
                return
            if w is not None and message['kind'] == 'output':
                now = time.monotonic()
                if w.tid in last_token:
                    intervals.append(now - last_token[w.tid])
                last_token[w.tid] = now
            return receive(peer, message)

        engine._send, engine._message = sending, received

    policy.install = initialized
    add = PDEngine.add_request

    def adding(engine, request_id, prompt_token_ids, sampling_params=None, *, trace_id=None):
        engine.ltp_choose(request_id, prompt_token_ids)
        return add(engine, request_id, prompt_token_ids, sampling_params, trace_id=trace_id)

    PDEngine.add_request = adding
