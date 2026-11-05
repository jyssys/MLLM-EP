"""Fit finite operation costs from calibration only. Do not read held-out runs."""
import statistics
from collections import defaultdict
from .common import *


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), median=statistics.median(values),
                minimum=values[0], maximum=values[-1], samples=values)


def curve(samples):
    return {str(batch): summary(values) for batch, values in sorted(samples.items()) if values}


def preparation_enclosures(frontiers, device_rows, expected, since):
    starts = {row['request_id']: row['time'] for row in frontiers
              if row['event'] == 'P_FIRST_SUBMISSION' and row.get('seed') in expected and row['time'] >= since}
    groups = defaultdict(list)
    covered = set()
    for row in device_rows:
        rid = row['request_id']
        if rid not in starts:
            continue
        # This field is the host entry into ModelRunner._execute, after input
        # preparation. Using its projected CUDA event would also count an
        # earlier batch's queued GPU execution a second time.
        groups[starts[rid]].append(max(0., 1000 * (row['first_host_submit'] - starts[rid])))
        covered.add(row['seed'])
    if covered != expected:
        raise ValueError('Calibration is missing a first-submit/model-entry preparation enclosure')
    return [(len(values), max(values)) for values in groups.values()]


def fit():
    ledger = json.loads((ROOT / 'state/cost_calibration.json').read_text())
    models = {}
    sources = []
    for model in ('ouro', 'huginn'):
        actions = {}
        decode = {phase: defaultdict(list) for phase in ('prelude', 'recurrent', 'coda')}
        for record in ledger['results']:
            if record['model'] != model or not record.get('passed'):
                continue
            name = record['name']
            cfg_path = ROOT / 'configs' / (name + '.json')
            cfg = json.loads(cfg_path.read_text())
            assert cfg['oracle_calibration'] and cfg['data_split'] == 'calibration'
            folder = ROOT / 'raw' / name
            action = actions.setdefault(cfg['method'], dict(P_steps={}, D_prompt=defaultdict(list),
                P_prepare=defaultdict(list), D_prepare=defaultdict(list),
                handoff=[], serial_handoff=[], ACK_lag=[], P_credit_release_lag=[], serial_service=[], serial_TPOT=[], P_host_residual=[],
                payload_bytes=[], graph_records=[], P_window_gap=[], D_window_gap=[]))
            event_sources = []
            serial_device = {'prefill': defaultdict(float), 'decode': defaultdict(float)}
            for path in folder.glob('device_execution_*.json'):
                value = json.loads(path.read_text())
                role = value['role']
                event_sources.append(dict(file=str(path.relative_to(ROOT)), SHA256=digest(path)))
                for row in value['rows']:
                    if not row['measured'] or not row['requests']:
                        continue
                    n = len(row['requests'])
                    prompt = any(request['token_count'] > 1 and request['generated_tokens'] == 0
                                 for request in row['requests'])
                    phase = row['operation']
                    if role == 'prefill' and prompt:
                        action['P_steps'].setdefault(phase, defaultdict(list))[n].append(row['stream_elapsed_ms'])
                    elif role == 'decode' and prompt:
                        action['D_prompt'][n].append(row['stream_elapsed_ms'])
                    elif role == 'decode' and phase in decode:
                        decode[phase][n].append(row['stream_elapsed_ms'])
                    if cfg.get('calibration_serial') and prompt:
                        for request in row['requests']:
                            serial_device[role][request['seed']] += row['stream_elapsed_ms']
            intervals = json.loads((folder / 'service_latency_requests.json').read_text())
            action['handoff'] += [max(0., row['observed_handoff_ms']) for row in intervals
                                  if row['observed_handoff_ms'] is not None]
            if cfg.get('calibration_serial'):
                action['serial_handoff'] += [max(0., row['observed_handoff_ms']) for row in intervals
                                             if row['observed_handoff_ms'] is not None]
                action['serial_service'] += [row['service_TTFT_ms'] for row in intervals]
                action['P_window_gap'] += [max(0., row['P_prompt_device_enclosure_ms'] -
                    serial_device['prefill'][row['seed']]) for row in intervals]
                action['D_window_gap'] += [max(0., (row['D_prompt_device_enclosure_ms'] or 0.) -
                    serial_device['decode'][row['seed']]) for row in intervals]
                action['P_host_residual'] += [max(0., row['service_TTFT_ms'] -
                    row['P_prompt_device_enclosure_ms'] - (row['D_prompt_device_enclosure_ms'] or 0.) -
                    max(0., row['observed_handoff_ms'] or 0.)) for row in intervals]
                outputs = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()
                           if not json.loads(line).get('warmup')]
                action['serial_TPOT'] += [row['tpot_ms'] for row in outputs]
            expected_ids = {row['request_id'] for row in intervals}
            measured_begin = json.loads((folder / 'measurement_begin.json').read_text())['time']
            hosts = {json.loads(path.read_text())['role']: json.loads(path.read_text())
                     for path in folder.glob('host_execution_*.json')}
            for path in folder.glob('service_device_times_*.json'):
                value = json.loads(path.read_text())
                role = value['role']
                if role == 'decode' and not action['D_prompt']:
                    continue
                samples = preparation_enclosures(hosts[role]['frontiers'], value['requests'],
                    {row['seed'] for row in intervals}, measured_begin)
                target = action['P_prepare' if role == 'prefill' else 'D_prepare']
                for batch, duration in samples:
                    target[batch].append(duration)
            received = {row['tid']: row['time'] for line in (folder / 'lifetime_events.jsonl').read_text().splitlines()
                        if (row := json.loads(line))['event'] == 'COORDINATOR_received'
                        and row.get('request_id') in expected_ids and row['time'] >= measured_begin}
            for path in folder.glob('lifetime_owner_*.jsonl'):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row['event'] == 'NATIVE_ACK_RECEIVED' and row['tid'] in received:
                        action['ACK_lag'].append(max(0., 1000 * (row['time'] - received[row['tid']])))
            for line in (folder / 'lifetime_events.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row['event'] == 'COORDINATOR_released' and row['role'] == 'prefill' and row['tid'] in received:
                    action['P_credit_release_lag'].append(max(0., 1000 * (row['time'] - received[row['tid']])))
            actual = record['actual_work']['counters']['prefill']['transport_bytes_sent'] / cfg['count']
            action['payload_bytes'].append(actual)
            action['graph_records'].append(record.get('graphs'))
            sources.append(dict(name=name, configuration_SHA256=digest(cfg_path), events=event_sources))
        if 'native' not in actions:
            continue
        model_cost = dict(actions={}, decode={phase: curve(value) for phase, value in decode.items()})
        for method, action in actions.items():
            model_cost['actions'][method] = dict(
                P_prepare=curve(action['P_prepare']), D_prepare=curve(action['D_prepare']),
                P_steps={phase: curve(value) for phase, value in action['P_steps'].items()},
                D_prompt=curve(action['D_prompt']), handoff_ms=summary(action['serial_handoff']),
                observed_handoff_including_burst_queue_ms=summary(action['handoff']),
                ACK_lag_ms=summary(action['ACK_lag']), serial_service_ms=summary(action['serial_service']),
                P_credit_release_lag_ms=summary(action['P_credit_release_lag']),
                serial_TPOT_ms=summary(action['serial_TPOT']), P_host_residual_ms=summary(action['P_host_residual']),
                P_window_gap_ms=summary(action['P_window_gap']), D_window_gap_ms=summary(action['D_window_gap']),
                payload_bytes=summary(action['payload_bytes']), graph_records=action['graph_records'])
        cfg = json.loads((ROOT / 'configs' / next(row['name'] + '.json' for row in ledger['results']
            if row['model'] == model)).read_text())
        # Read the actual native contract, not an assumed pool size.
        contracts = [json.loads(path.read_text()) for source in sources if source['name'].startswith('f_' + model + '_native')
                     for path in (ROOT / 'raw' / source['name']).glob('native_contract_initial_*.json')]
        actual_D = next(row for row in contracts if row['role'] == 'decode')
        actual_P = next(row for row in contracts if row['role'] == 'prefill')
        blocks_per_request = ((cfg['length'] + cfg['cap'] - 1 + 15) // 16) * cfg['depth']
        P_blocks = ((cfg['length'] + 15) // 16) * cfg['depth']
        model_cost.update(length=cfg['length'], depth=cfg['depth'], cap=cfg['cap'],
            P_slots=cfg['P_slots'], D_slots=cfg['D_slots'],
            D_KV_blocks=actual_D['KV_blocks'], request_KV_blocks=blocks_per_request,
            P_KV_blocks=actual_P['KV_blocks'], P_request_KV_blocks=P_blocks,
            P_credit_limit=min(cfg['P_slots'] + cfg.get('native_transport_limits', {}).get('max_draining_requests', 8),
                               actual_P['KV_blocks'] // P_blocks),
            D_credit_limit=min(cfg['D_slots'] + cfg.get('native_transport_limits', {}).get('max_receiving_requests', 32),
                               actual_D['KV_blocks'] // blocks_per_request),
            admission_scope='KV reservations include receiving requests. Active Decode compute slots remain a separate limit.',
            max_D_prefill_rows=cfg['D_token_budget'])
        models[model] = model_cost
    return dict(time=time.time(), models=models, sources=sources,
        evidence='MEASURED_CALIBRATION_CUDA_EVENT_ENCLOSURES_AND_HOST_INTERVALS',
        scope='Finite measured costs. Device and host-submit-to-execute enclosures include waits and are not exclusive CPU/kernel times. Initial P preparation precedes Service TTFT. The preparation envelope excludes queued GPU execution. No held-out results enter fitting. Missing shapes require explicit interpolation. Not a validated oracle until static prediction tests pass.')
