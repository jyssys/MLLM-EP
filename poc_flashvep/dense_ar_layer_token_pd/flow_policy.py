"""Use only arrived FIFO work in a bounded, empirical pipeline lookahead."""
import itertools
import json
import math
import time
from dataclasses import dataclass
from .common import ROOT
from .replay import Model


def plan_score(groups, cuts, now, available, model, regime):
    P, D = now, available
    qualified, output, waiting = 0, 0, 0.
    for group, cut in zip(groups, cuts):
        caps, arrivals = group['caps'], group['arrivals']
        batch = len(caps)
        P += model.cost('negotiation_ms', batch, cut) + model.cost('P_source_ms', batch, cut)
        first = max(P, D) + model.cost('D_prepare_ms', batch, cut)
        # Cached service curve includes decreasing active batch for mixed caps.
        signature = tuple(caps)
        cache = getattr(model, '_decode_curve_cache', None)
        if cache is None:
            cache = model._decode_curve_cache = {}
        if signature not in cache:
            cumulative, maximum = [0.], [0.]
            for k in range(1, max(caps)):
                elapsed = model.step(sum(cap > k for cap in caps))
                cumulative.append(cumulative[-1] + elapsed)
                maximum.append(max(maximum[-1], elapsed))
            cache[signature] = cumulative, maximum
        cumulative, maximum = cache[signature]
        for arrival, cap in zip(arrivals, caps):
            limits = model.thresholds[str(cap)][regime]
            ttft = 1000 * (first - arrival)
            tpot = 1000 * cumulative[cap-1] / (cap-1)
            passed = ttft <= limits['TTFT_ms'] and tpot <= limits['TPOT_ms'] and 1000*maximum[cap-1] <= limits['ITL_ms']
            qualified += cap if passed else 0
            output += cap
            waiting += first-arrival
        D = first + cumulative[-1]
    span = max(1e-6, D - min(groups[0]['arrivals']))
    return (qualified/span, output/span, -waiting), dict(P_finish=P, D_finish=D,
        qualified_output=qualified, output=output, waiting_sum=waiting)


def install(cfg, module):
    if not cfg.get('flow_balance_policy'):
        return
    assert cfg['policy'] == 'DIRECT' and (cfg.get('preplanned_source_groups') or cfg.get('flow_adaptive_source'))
    assert cfg['P_group_size'] == cfg['D_admission_batch']
    original_observation, original_policy = module.Observation, module.LayerPolicy

    @dataclass
    class VisibleObservation(original_observation):
        visible_fifo_groups: tuple = ()

    class FlowPolicy(original_policy):
        def __init__(self, config, costs):
            super().__init__(config, costs)
            empirical = json.loads((ROOT / 'configs' / config['flow_cost_model']).read_text())
            thresholds = json.loads((ROOT / 'configs/frozen_SLO_regimes.json').read_text())
            self.model = Model(empirical, costs, thresholds, group_size=config['P_group_size'])

        def _cost(self, kind, layers, batch):
            # Existing availability bookkeeping now consumes the measured host
            # preparation enclosure. This is not a zero wire-time assertion.
            if kind == 'P_prefix_ms':
                return 1000*self.model.cost('P_source_ms', batch, layers)
            if kind == 'D_suffix_ms':
                return 1000*self.model.cost('D_prepare_ms', batch, layers)
            if kind in ('handoff_ms', 'D_commit_ms'):
                return 0.
            return super()._cost(kind, layers, batch)

        def decode_bound(self, batch):
            if batch <= 0:
                return super().decode_bound(batch)
            cell = self.model.tables['decode_step_ms'].get(str(batch))
            return cell['p95'] + self.cfg.get('flow_decode_margin_ms', 2.) if cell else super().decode_bound(batch)

        def choose(self, state):
            self.counter += 1
            if state.state_age_ms > self.cfg.get('maximum_state_age_ms', 250):
                return module.Choice(reason='FLOW_STALE_OBSERVATION_NATIVE')
            if set(state.input_token_lengths) != {8000}:
                return module.Choice(reason='FLOW_UNCALIBRATED_INPUT_NATIVE')
            batch = state.group_size
            blocks = sum(math.ceil((8000 + cap)/16) for cap in state.group_output_caps)
            regime = self.cfg.get('objective_SLO', 'Decode_priority')
            arrivals = tuple(deadline-self.model.thresholds[str(cap)][regime]['TTFT_ms']/1000
                for cap, deadline in zip(state.group_output_caps, state.own_TTFT_deadlines))
            current = dict(caps=tuple(state.group_output_caps), arrivals=arrivals)
            future = list(state.visible_fifo_groups[:self.cfg.get('flow_horizon_groups', 3)-1])
            assert all(max(g['arrivals']) <= state.now for g in future), 'Future arrival leaked into online policy'
            groups = [current] + future
            available = state.D_available_nominal if state.D_available_nominal is not None else state.D_available
            retirement_bounds = [item['retire_upper'] for item in state.affected_deadlines
                if item['kind']=='ITL' and item.get('ordering')=='BEFORE_CANDIDATE'
                and item.get('retire_upper') is not None]
            available_upper = max([state.D_available]+retirement_bounds)
            candidates = []
            all_cuts = (0, 2, 4, 8, 12)
            for layers in all_cuts:
                P_ready = state.now + self.model.cost('negotiation_ms', batch, layers) + self.model.cost('P_source_ms', batch, layers)
                source_cell = self.model.tables['P_source_ms'].get(f'{batch}:{layers}')
                prepare_cell = self.model.tables['D_prepare_ms'].get(f'{batch}:{layers}')
                uncertainty = self.cfg.get('flow_completion_margin_ms', 110.)
                source_tail = max(0., source_cell['p95']-source_cell['p50']) if source_cell else uncertainty
                prep_tail = max(0., prepare_cell['p95']-prepare_cell['p50']) if prepare_cell else uncertainty
                start = max(P_ready, available_upper)
                finish = start + self.model.cost('D_prepare_ms', batch, layers)
                finish_upper = finish + (source_tail+prep_tail+uncertainty)/1000
                failures = []
                if layers and blocks > state.KV_free_blocks:
                    failures.append('KV_CREDIT')
                if layers and state.D_preparation_overdue:
                    failures.append('KNOWN_D_PREPARATION_OVERRUN')
                if layers and any(deadline > state.now and finish_upper > deadline for deadline in state.own_TTFT_deadlines):
                    failures.append('OWN_UNEXPIRED_TTFT_DEADLINE')
                if layers and self.decode_bound(batch) > state.own_ITL_limit_ms:
                    failures.append('OWN_DECODE_STEP_BOUND')
                if layers and 1000*max(0., available_upper-P_ready) > self.cfg.get('direct_wait_limit_ms', 1000.):
                    failures.append('BOUNDED_ADMISSION_RESIDENCE')
                # The actual worker admits no suffix before the old Decode
                # cohort retires. Only known, non-retired work can conflict.
                from ..dense_ar_slo_layer_pd.deadline import affected_deadline_conflicts
                affected = [item for item in state.affected_deadlines if item['kind']=='ITL' and item.get('ordering')=='BEFORE_CANDIDATE']
                conflicts, retired = affected_deadline_conflicts(start, finish_upper, affected)
                if layers and conflicts:
                    failures.append('AFFECTED_DECODE_DEADLINE')
                paths = []
                if not failures:
                    for suffix in itertools.product(all_cuts, repeat=len(future)):
                        cuts = (layers,) + suffix
                        score, detail = plan_score(groups, cuts, state.now, available, self.model, regime)
                        paths.append((score, cuts, detail))
                best = max(paths, key=lambda v:v[0]) if paths else None
                candidates.append(dict(layers=layers, accepted=not failures, rejection_reasons=failures,
                    predicted_P_cut_ready=P_ready, predicted_D_start=start,
                    predicted_D_finish=finish_upper, predicted_D_finish_mean=finish,
                    predicted_P_compute_saved_ms=1000*(self.model.cost('P_source_ms',batch,0)-self.model.cost('P_source_ms',batch,layers)),
                    score=list(best[0]) if best else None, lookahead_cuts=list(best[1]) if best else None,
                    predicted_chain=best[2] if best else None, known_horizon=len(groups),
                    safety_margin_ms=source_tail+prep_tail+uncertainty,
                    affected_deadline_conflicts=conflicts, retired_before_suffix=retired,
                    cost_scope='Empirical host service enclosures. No exclusive GPU-time, zero transport-cost, physical idle or WCET claim. Future groups are already arrived but uncommitted. Future credits are a planning estimate; only the current prefix receives a real reservation.'))
            feasible = [value for value in candidates if value['accepted']]
            best = max(feasible, key=lambda v:(tuple(v['score']),-v['layers']))
            if self.cfg.get('flow_controller_zero'):
                return module.Choice(0, 'FLOW_CONTROLLER_ZERO', candidates)
            return module.Choice(best['layers'], 'FLOW_VISIBLE_FIFO_HORIZON', candidates)

    module.Observation, module.LayerPolicy = VisibleObservation, FlowPolicy
