"""Keep correctness, graph coverage, marker timing, and clean latency separate."""
from .common import *
from .run import execute


def graph_contract(name):
    rows = []
    for path in (ROOT / 'raw' / name).glob('deep_graph_cache_*.json'):
        value = json.loads(path.read_text())
        count = value['counts']['measured']
        total = count['selects']
        rows.append(dict(role=value['role'], budget=value['budget'],
            entries=value['entries'], measured=count,
            measured_existing_hit_ratio=count['existing_hits'] / total if total else None,
            passed=count['cold_entries'] == 0 and count['fallbacks'] == 0,
            scope='Measured native packed-prefill selections only. Startup counters remain separate.'))
    return dict(roles=rows, passed=bool(rows) and all(row['passed'] for row in rows))


def evaluate(cfg, reference=None):
    initialize()
    save(cfg)
    folder = ROOT / 'raw' / cfg['name']
    cached = folder / 'deep_serving_gate.json'
    if cached.exists():
        return json.loads(cached.read_text())
    from ..native_loop_2d_oracle import serving as inherited
    inherited.initialize, inherited.execute = initialize, execute
    row = inherited.evaluate(cfg, reference)
    if row.get('complete'):
        from ..dynamic_loop_split_serving.metrics import individual_slo_attainment
        outputs = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()
                   if not json.loads(line).get('warmup')]
        offered = [json.loads(line) for line in (folder / 'offered_requests.jsonl').read_text().splitlines()]
        metrics = row['metrics']
        metrics.update(individual_slo_attainment(outputs, cfg['count'], metrics['frozen_SLO']))
        metrics['input_tokens_per_second'] = metrics['total_tokens_per_second'] - metrics['output_tokens_per_second']
        metrics.update(nominal_QPS=cfg.get('arrival_rate'),
            realized_arrival_QPS=(len(outputs) - 1) / (max(v['arrival'] for v in outputs) - min(v['arrival'] for v in outputs)),
            actual_submit_QPS=(len(offered) - 1) / (max(v['actual_send_start'] for v in offered) - min(v['actual_send_start'] for v in offered)),
            markers=cfg['observe_service_timing'], data_split=cfg['data_split'],
            service_scope='Native model-stream entry event to first output, including handoff and continuation.'
                if cfg['observe_service_timing'] else 'NOT_MEASURED_IN_MARKER_FREE_RUN')
        if cfg.get('oracle_calibration'):
            metrics['scope'] = 'Instrumented calibration only. Serial or bounded burst, not clean open-loop serving or capacity.'
        row['graphs'] = graph_contract(cfg['name'])
        row['performance_eligible_graphs'] = row['graphs']['passed']
        wire = [json.loads(path.read_text()) for path in folder.glob('wire_payload_*.json')]
        source = next((item for item in wire if item['role'] == 'prefill'), None)
        if source:
            measured = source['counts']['measured']
            total = measured['KV_bytes'] + measured['hidden_bytes']
            stats = row['actual_work']['counters']['prefill']
            row['wire_accounting'] = dict(roles=wire,
                agrees_with_native_measured_bytes=total == stats['transport_bytes_sent'],
                agrees_with_native_measured_submissions=measured['accepted_submissions'] == stats['transport_completed_count'],
                scope='Accepted-region accounting is compared with native completed-transfer counters. Startup remains separate. Control/ACK messages and wire headers remain unmeasured.')
        else:
            row['wire_accounting'] = dict(status='NOT_MEASURED')
        write(folder / 'deep_serving_metrics.json', metrics)
    write(cached, row)
    if row.get('complete') and not row['passed']:
        raise RuntimeError('Stop the affected serving path after a correctness failure: ' + cfg['name'])
    return row
