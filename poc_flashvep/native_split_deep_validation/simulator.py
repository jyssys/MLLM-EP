"""Event-driven offline model. Its outputs require independent GPU validation."""
import heapq
import statistics
from collections import deque


def value(record, default=0.):
    return record['median'] if record else default


def lookup(curve, batch):
    if not curve:
        return 0.
    sizes = sorted(map(int, curve))
    if str(batch) in curve:
        return value(curve[str(batch)])
    upper = next((n for n in sizes if n >= batch), None)
    if upper is not None:
        # The nearest larger measured geometry is a stated model approximation.
        return value(curve[str(upper)])
    return value(curve[str(sizes[-1])]) * batch / sizes[-1]


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


class Simulation:
    def __init__(self, model, arrivals_ms, choices, slo):
        if len(arrivals_ms) != len(choices) or not choices:
            raise ValueError('An action is required for every offered request')
        if any(choice not in model['actions'] for choice in choices):
            raise ValueError('An unmeasured action cannot enter this model')
        self.model, self.slo = model, slo
        self.now = 0.
        self.serial = 0
        self.events = []
        self.requests = [dict(arrival=arrival, method=method, P_step=0, start=None,
                              tokens=[], depth=0, phase=None, finished=False)
                         for arrival, method in zip(arrivals_ms, choices)]
        self.waiting = deque()
        self.P_ready = []
        self.D_imported = deque()
        self.D_active = set()
        self.P_leases = set()
        self.D_credits = set()
        self.P_busy = self.D_busy = False
        self.link_available = 0.
        self.prefills_since_core = 0
        self.graph_keys = {'prefill': set(), 'decode': set()}
        self.graph_geometry_known = True
        self.P_steps = {method: [phase for phase in ('prelude', 'recurrent') if phase in cost['P_steps']]
                        for method, cost in model['actions'].items()}
        native = model['actions']['native']
        target = value(native['serial_TPOT_ms'])
        device = (lookup(model['decode']['prelude'], 1) + lookup(model['decode']['coda'], 1) +
                  model['depth'] * lookup(model['decode']['recurrent'], 1))
        self.D_host_gap = max(0., (target - device) / (model['depth'] + 2))
        self.delivery = {}
        for method, action in model['actions'].items():
            predicted = (sum(lookup(action['P_steps'][phase], 1) for phase in self.P_steps[method]) +
                value(action['P_window_gap_ms']) + value(action['handoff_ms']) +
                lookup(action['D_prompt'], 1) + lookup(action.get('D_prepare', {}), 1) + value(action['D_window_gap_ms']) +
                lookup(model['decode']['coda'], 1) + self.D_host_gap)
            self.delivery[method] = max(0., value(action['serial_service_ms']) - predicted)
        for index, arrival in enumerate(arrivals_ms):
            self.event(arrival, 'arrival', [index])

    def event(self, at, kind, indices):
        self.serial += 1
        heapq.heappush(self.events, (at, self.serial, kind, indices))

    def action(self, index):
        return self.model['actions'][self.requests[index]['method']]

    def graph(self, role, batch, query):
        if query is None:
            self.graph_geometry_known = False
            return
        self.graph_keys[role].add((batch * query, batch, query, True, False))

    def complete(self, kind, indices):
        if kind == 'arrival':
            self.waiting.extend(indices)
        elif kind == 'P_done':
            self.P_busy = False
            for index in indices:
                request = self.requests[index]
                request['P_step'] += 1
                if request['P_step'] < len(self.P_steps[request['method']]):
                    self.P_ready.append(index)
                else:
                    request['P_completed'] = self.now
                    cost = self.action(index)
                    end = max(self.now, self.link_available) + value(cost['handoff_ms'])
                    self.link_available = end
                    self.event(end, 'D_received', [index])
                    self.event(end + value(cost['ACK_lag_ms']), 'source_ACK', [index])
                    release = value(cost.get('P_credit_release_lag_ms'), value(cost['ACK_lag_ms']))
                    self.event(end + max(release, value(cost['ACK_lag_ms'])), 'P_credit_released', [index])
        elif kind == 'source_ACK':
            for index in indices:
                self.requests[index]['source_ACK'] = self.now
        elif kind == 'P_credit_released':
            self.P_leases.difference_update(indices)
            for index in indices:
                self.requests[index]['P_credit_released'] = self.now
        elif kind == 'D_received':
            self.D_imported.extend(indices)
            for index in indices:
                self.requests[index]['D_received'] = self.now
        elif kind == 'D_prompt_done':
            self.D_busy = False
            for index in indices:
                self.requests[index]['phase'] = 'coda'
        elif kind == 'D_prelude_done':
            self.D_busy = False
            for index in indices:
                self.requests[index].update(phase='recurrent', depth=0)
        elif kind == 'D_recurrent_done':
            self.D_busy = False
            for index in indices:
                request = self.requests[index]
                request['depth'] += 1
                if request['depth'] == self.model['depth']:
                    request['phase'] = 'coda'
        elif kind == 'D_coda_done':
            self.D_busy = False
            for index in indices:
                request = self.requests[index]
                request['tokens'].append(self.now + self.delivery[request['method']])
                if len(request['tokens']) == self.model['cap']:
                    request.update(finished=True, phase=None)
                    self.D_active.remove(index)
                    self.D_credits.remove(index)
                else:
                    request['phase'] = 'prelude'
        else:
            raise ValueError('Unknown model event: ' + kind)

    def dispatch(self):
        while self.waiting and len(self.P_leases) < self.model.get('P_credit_limit', self.model['P_slots']) and len(self.D_credits) < self.model['D_credit_limit']:
            index = self.waiting.popleft()
            self.P_leases.add(index)
            self.D_credits.add(index)
            self.P_ready.append(index)
        while self.D_imported and len(self.D_active) < self.model['D_slots']:
            index = self.D_imported.popleft()
            self.D_active.add(index)
            self.requests[index]['phase'] = 'prompt' if self.action(index)['D_prompt'] else 'coda'
        if not self.P_busy and self.P_ready:
            index = min(self.P_ready, key=lambda i: (self.requests[i]['P_step'] != 0, i))
            request = self.requests[index]
            method, step = request['method'], request['P_step']
            # Different actions do not receive free mixed-cut batching.
            batch = [i for i in self.P_ready if self.requests[i]['method'] == method
                     and self.requests[i]['P_step'] == step][:self.model['P_slots']]
            action = self.action(index)
            preparation = lookup(action.get('P_prepare', {}), len(batch)) if step == 0 else 0.
            for i in batch:
                self.P_ready.remove(i)
                if self.requests[i]['start'] is None:
                    self.requests[i]['start'] = self.now + preparation
            phase = self.P_steps[method][step]
            self.graph('prefill', len(batch), action.get('P_query_rows', {}).get(phase))
            cost = preparation + lookup(action['P_steps'][phase], len(batch))
            if step == len(self.P_steps[method]) - 1:
                cost += value(action['P_window_gap_ms']) * len(batch)
            self.P_busy = True
            self.event(self.now + max(.001, cost), 'P_done', batch)
        if self.D_busy or not self.D_active:
            return
        phases = {phase: sorted(i for i in self.D_active if self.requests[i]['phase'] == phase)
                  for phase in ('prelude', 'coda', 'recurrent', 'prompt')}
        if phases['prelude']:
            phase = 'prelude'
        elif phases['coda']:
            phase = 'coda'
        elif self.prefills_since_core >= 1 and phases['recurrent']:
            phase = 'recurrent'
        elif phases['prompt']:
            phase = 'prompt'
        else:
            phase = 'recurrent'
        batch = phases[phase]
        if not batch:
            return
        if phase == 'prompt':
            method = self.requests[batch[0]]['method']
            action = self.action(batch[0])
            maximum = action.get('D_max_batch', 1)
            batch = [i for i in batch if self.requests[i]['method'] == method][:maximum]
            self.graph('decode', len(batch), action.get('D_query_rows'))
            cost = (lookup(action.get('D_prepare', {}), len(batch)) + lookup(action['D_prompt'], len(batch))
                    + value(action['D_window_gap_ms']) * len(batch))
            self.prefills_since_core += 1
        else:
            cost = lookup(self.model['decode'][phase], len(batch)) + self.D_host_gap
            if phase == 'recurrent':
                self.prefills_since_core = 0
        self.D_busy = True
        self.event(self.now + max(.001, cost), 'D_' + phase + '_done', batch)

    def run(self):
        iterations = 0
        while self.events:
            self.now = self.events[0][0]
            while self.events and self.events[0][0] <= self.now + 1e-8:
                _, _, kind, indices = heapq.heappop(self.events)
                self.complete(kind, indices)
            self.dispatch()
            iterations += 1
            if iterations > 2000000:
                raise RuntimeError('Offline model event bound exceeded')
        if not all(request['finished'] for request in self.requests):
            raise RuntimeError('Offline model deadlocked with incomplete requests')
        assert not self.P_leases and not self.D_credits and not self.D_active
        metrics = {}
        vectors = {field: [] for field in ('TTFT', 'service_TTFT', 'queue', 'TPOT', 'ITL', 'E2E')}
        passed = 0
        for request in self.requests:
            tokens = request['tokens']
            first, last = tokens[0], tokens[-1]
            ttft = first - request['arrival']
            tpot = (last - first) / (len(tokens) - 1)
            vectors['TTFT'].append(ttft)
            vectors['service_TTFT'].append(first - request['start'])
            vectors['queue'].append(request['start'] - request['arrival'])
            vectors['TPOT'].append(tpot)
            vectors['ITL'] += [right - left for left, right in zip(tokens, tokens[1:])]
            vectors['E2E'].append(last - request['arrival'])
            passed += ttft <= self.slo['TTFT_ms'] and tpot <= self.slo['TPOT_ms']
        for field, values in vectors.items():
            for q in (.5, .95, .99):
                metrics[f'{field}_p{int(100*q)}_ms'] = percentile(values, q)
        span = max(request['tokens'][-1] for request in self.requests) - min(request['arrival'] for request in self.requests)
        metrics.update(requests_per_second=1000 * len(self.requests) / span,
            joint_SLO_goodput=1000 * passed / span, joint_SLO_attainment=passed / len(self.requests),
            maximum_ITL_ms=max(vectors['ITL']),
            Decode_SLO_screen=metrics['TPOT_p95_ms'] <= .9 * self.slo['TPOT_ms']
                and metrics['ITL_p99_ms'] <= self.slo['ITL_ms'],
            evidence='MODELED_OFFLINE_EVENT_SIMULATION_NOT_GPU_MEASUREMENT')
        budget = self.model.get('graph_budget')
        metrics.update(graph_budget_screen=(all(len(keys) <= budget for keys in self.graph_keys.values())
                         if budget is not None and self.graph_geometry_known else None),
            graph_keys_per_role={role: len(keys) for role, keys in self.graph_keys.items()})
        return dict(metrics=metrics, requests=self.requests, event_iterations=iterations,
                    required_packed_graph_keys={role: [list(key) for key in sorted(keys)] for role, keys in self.graph_keys.items()},
                    D_host_gap_per_quantum_ms=self.D_host_gap,
                    scope='Serialized device-resource approximation with explicit KV reservations, source ACK leases, causal stages, Decode phases, and bounded transfer queue. Requires static GPU validation.')
