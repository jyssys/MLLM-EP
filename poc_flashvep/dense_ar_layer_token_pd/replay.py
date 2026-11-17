"""State-updating fixed-geometry replay, not a measured serving speedup."""
import copy
import itertools
import json
import math
from dataclasses import dataclass, field
from .common import ROOT
from ..dense_ar_slo_layer_pd.analyze import percentile


@dataclass
class Group:
    indices: tuple
    arrivals: tuple
    caps: tuple
    length: int = 8000


@dataclass
class State:
    P: float = 0.
    D: float = 0.
    source: list = field(default_factory=list)
    destination: list = field(default_factory=list)
    hidden_source: list = field(default_factory=list)
    hidden_destination: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    decisions: list = field(default_factory=list)


class Model:
    def __init__(self, empirical, legacy, thresholds, group_size=4, regime='Decode_priority'):
        self.tables = empirical['tables']
        self.legacy = legacy
        self.thresholds = thresholds['output_caps']
        self.group_size = group_size
        self.regime = regime
        self.borrowed = set()
        self.KV_blocks = 19114
        self.hidden_slots = 32

    def cost(self, kind, batch, layers):
        key = str(batch) + ':' + str(layers)
        value = self.tables[kind].get(key)
        if value:
            return value['p50'] / 1000
        self.borrowed.add(kind + ':' + key)
        if kind == 'P_source_ms':
            return self.legacy['P_prefix_ms'][str(batch)][str(layers)] / 1000
        if kind == 'negotiation_ms':
            return 0. if layers == 0 else .010
        assert kind == 'D_prepare_ms'
        estimate = sum(self.legacy[name][str(batch)][str(layers)]
                       for name in ('handoff_ms', 'D_suffix_ms', 'D_commit_ms'))
        anchor = self.tables[kind].get('4:' + str(layers))
        if anchor:
            anchor_old = sum(self.legacy[name]['4'][str(layers)]
                             for name in ('handoff_ms', 'D_suffix_ms', 'D_commit_ms'))
            estimate += anchor['p50'] - anchor_old
        return max(0., estimate) / 1000

    def step(self, batch):
        value = self.tables['decode_step_ms'].get(str(batch))
        if value:
            return value['p50'] / 1000
        self.borrowed.add('decode_step_ms:' + str(batch))
        table = self.legacy['decode_step_by_batch_p95_ms']
        key = min(int(k) for k in table if int(k) >= batch)
        old = table[str(key)]
        anchor = self.tables['decode_step_ms']['4']['p50']
        bias = self.legacy['decode_step_by_batch_p95_ms']['4'] - anchor
        return max(1., old - bias) / 1000

    def advance(self, state, group, requested_layers):
        s = copy.copy(state)
        s.rows = list(state.rows)
        s.decisions = list(state.decisions)
        for name in ('source', 'destination', 'hidden_source', 'hidden_destination'):
            setattr(s, name, list(getattr(state, name)))
        batch = len(group.indices)
        start = max(state.P, max(group.arrivals))
        blocks = batch * math.ceil(group.length / 16)
        while sum(n for end, n in s.source if end > start) + blocks > self.KV_blocks:
            start = min(end for end, _ in s.source if end > start)
        layers = requested_layers
        reason = 'REQUESTED_CUT'
        d_blocks = sum(math.ceil((group.length + cap) / 16) for cap in group.caps)
        if layers and (sum(n for end, n in s.destination if end > start) + d_blocks > self.KV_blocks or
                       sum(n for end, n in s.hidden_source if end > start) + batch > self.hidden_slots or
                       sum(n for end, n in s.hidden_destination if end > start) + batch > self.hidden_slots):
            layers, reason = 0, 'CONSERVATIVE_CREDIT_NATIVE_FALLBACK'
        negotiation = self.cost('negotiation_ms', batch, layers)
        ready = start + negotiation + self.cost('P_source_ms', batch, layers)
        d_start = max(ready, state.D)
        prep = self.cost('D_prepare_ms', batch, layers)
        first = d_start + prep
        cache=getattr(self,'_decode_curves',None)
        if cache is None:
            cache=self._decode_curves={}
        signature=tuple(group.caps)
        if signature not in cache:
            cumulative,step_costs=[0.],[]
            for k in range(1,max(group.caps)):
                active=sum(cap>k for cap in group.caps)
                elapsed=self.step(active)
                step_costs.append(elapsed)
                cumulative.append(cumulative[-1]+elapsed)
            cache[signature]=(cumulative,step_costs)
        cumulative,step_costs=cache[signature]
        done = first + cumulative[-1]
        for index, arrival, cap in zip(group.indices, group.arrivals, group.caps):
            values = step_costs[:cap - 1]
            limits = self.thresholds[str(cap)][self.regime]
            ttft = 1000 * (first - arrival)
            tpot = 1000 * sum(values) / (cap - 1)
            maximum = 1000 * max(values)
            passed = ttft <= limits['TTFT_ms'] and tpot <= limits['TPOT_ms'] and maximum <= limits['ITL_ms']
            s.rows.append(dict(sample_id=index, cap=cap, arrival=arrival, first=first,
                finish=first + cumulative[cap - 1], TTFT_ms=ttft, TPOT_ms=tpot,
                max_ITL_ms=maximum, joint=passed, layers=layers))
        # ACK is modeled after import, before suffix compute. The empirical
        # preparation enclosure is not an exclusive wire duration.
        import_estimate = self.legacy['handoff_ms'][str(batch)][str(layers)] / 1000
        ack = d_start + min(prep, import_estimate)
        s.source.append((ack, blocks))
        s.destination.append((done, d_blocks))
        if layers:
            s.hidden_source.append((ack, batch))
            s.hidden_destination.append((first, batch))
        s.P, s.D = ready, done
        deadline = min(arrival + self.thresholds[str(cap)][self.regime]['TTFT_ms'] / 1000
                       for arrival, cap in zip(group.arrivals, group.caps))
        s.decisions.append(dict(indices=list(group.indices), requested_layers=requested_layers,
            layers=layers, reason=reason, P_start=start, P_ready=ready,
            D_start=d_start, first=first, D_done=done,
            D_wait_ms=1000 * (d_start - ready),
            own_first_output_budget_ms=1000 * (deadline - d_start),
            P_source_saved_ms=1000 * (self.cost('P_source_ms', batch, 0) - self.cost('P_source_ms', batch, layers)),
            budget_scope='Modeled first-token deadline budget after the whole existing D cohort retires. Not physical GPU idle.'))
        return s


def groups(offsets, caps, size):
    return [Group(tuple(range(i, min(i + size, len(caps)))), tuple(offsets[i:i + size]), tuple(caps[i:i + size]))
            for i in range(0, len(caps), size)]


def evaluate(state):
    span = max(r['finish'] for r in state.rows) - min(r['arrival'] for r in state.rows)
    qualified = sum(r['cap'] for r in state.rows if r['joint'])
    return dict(SLO_output_tokens_s=qualified / span,
        raw_output_tokens_s=sum(r['cap'] for r in state.rows) / span,
        Joint_attainment=sum(r['joint'] for r in state.rows) / len(state.rows),
        TTFT_p95_ms=percentile([r['TTFT_ms'] for r in state.rows], .95),
        mean_TTFT_ms=sum(r['TTFT_ms'] for r in state.rows) / len(state.rows),
        span=span, migrated_requests=sum(r['layers'] > 0 for r in state.rows),
        layer_work=sum(r['layers'] for r in state.rows))


def run(model, workload, policy):
    state = State()
    for index, group in enumerate(workload):
        state = model.advance(state, group, policy(index, group))
    return state


def static_frontier(model, workload, fractions=(.25, .5, .75, 1.)):
    candidates = []
    for layers in (0, 2, 4, 8, 12):
        for fraction in ((1.,) if layers == 0 else fractions):
            state = run(model, workload, lambda i, _g, m=layers, f=fraction:
                m if math.floor((i + 1) * f) > math.floor(i * f) else 0)
            candidates.append((evaluate(state), dict(kind='FIXED_FRACTION', layers=layers, fraction=fraction), state))
    cap_types = sorted({max(g.caps) for g in workload})
    if len(cap_types) <= 3:
        for values in itertools.product((0, 2, 4, 8, 12), repeat=len(cap_types)):
            mapping = dict(zip(cap_types, values))
            state = run(model, workload, lambda _i, g: mapping[max(g.caps)])
            candidates.append((evaluate(state), dict(kind='KNOWN_CAP_MAPPING', mapping=mapping), state))
    return max(candidates, key=lambda value: (value[0]['SLO_output_tokens_s'], -value[0]['TTFT_p95_ms'])), candidates


def oracle(model, workload, width=256):
    beam = [State()]
    for group in workload:
        children = [model.advance(state, group, layers) for state in beam for layers in (0, 2, 4, 8, 12)]
        children.sort(key=lambda state: (evaluate(state)['SLO_output_tokens_s'],
                                        -evaluate(state)['TTFT_p95_ms']), reverse=True)
        # Preserve distinct resource states; do not count identical Native
        # fallbacks from four rejected layer candidates as four beam slots.
        seen, beam = set(), []
        for state in children:
            key = (round(state.P, 6), round(state.D, 6),
                   sum(r['cap'] for r in state.rows if r['joint']),
                   tuple(r['layers'] for r in state.rows[-model.group_size:]))
            if key in seen:
                continue
            seen.add(key)
            beam.append(state)
            if len(beam) >= width:
                break
    return max(beam, key=lambda state: (evaluate(state)['SLO_output_tokens_s'], -evaluate(state)['TTFT_p95_ms']))


def load_model(size=4, version='v2'):
    return Model(json.loads((ROOT / 'configs' / ('empirical_host_cost_model_discovery_' + version + '.json')).read_text()),
        json.loads((ROOT / 'configs/canonical_cost_model_v5.json').read_text()),
        json.loads((ROOT / 'configs/frozen_SLO_regimes.json').read_text()), group_size=size)
