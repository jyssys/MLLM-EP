"""Compute load ratios without conflating worker time, delivery, or kernels."""
import argparse
import collections
import json
import statistics

from .common import PRIOR, ROOT, bind_previous, write


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def output_gate(name):
    from ..dense_ar_slo_layer_pd.common import PREVIOUS
    config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    reference_path = (ROOT / 'raw' / config['reference_name'] / 'requests.jsonl' if config.get('reference_name') else
        PREVIOUS / 'raw/baseline_Native_O200_N64_q4p8_s21_v1/requests.jsonl')
    reference = {row['prompt_id']: row for row in records(reference_path)}
    rows = records(ROOT / 'raw' / name / 'requests.jsonl')
    failures = []
    for row in rows:
        old = reference.get(row['prompt_id'])
        if old is None or row['input_ids'] != old['input_ids']:
            failures.append(dict(sample_id=row['sample_id'], reason='INCOMPARABLE_PROMPT'))
        elif row['output_ids'] != old['output_ids'][:row['output_tokens']]:
            first = next((i for i, (a, b) in enumerate(zip(row['output_ids'], old['output_ids'])) if a != b), None)
            failures.append(dict(sample_id=row['sample_id'], reason='GREEDY_OUTPUT_MISMATCH', first_token_index=first))
    gate = dict(name=name, reference=str(reference_path), compared=len(rows), expected=config['count'],
        failures=failures, passed=len(rows) == config['count'] and not failures,
        evidence='CORRECTNESS_ONLY',
        scope='Original reference supplies 200 greedy tokens. Prompt IDs and complete input IDs must match. Output parity does not imply all-state parity.')
    write(ROOT / 'analysis' / (name + '_output_gate.json'), gate)
    return gate


def metrics(name):
    bind_previous()
    from ..dense_ar_slo_layer_pd.analyze import metrics as inherited_metrics, percentile
    out = inherited_metrics(name)
    if not out['complete']:
        return out
    folder = ROOT / 'raw' / name
    config = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    rows = records(folder / 'requests.jsonl')
    parent, worker = [], []
    for row in rows:
        assert row['token_events'][0]['count'] == 1, 'First individual token delivery is required'
        assert len(row['token_events']) == row['output_tokens'], 'Missing individual token delivery timestamps'
        for a, b in zip(row['token_events'], row['token_events'][1:]):
            assert b['count'] - a['count'] == 1, 'Do not infer individual ITL from coalesced delivery'
            parent.append(1000 * (b['time'] - a['time']))
            worker.append(1000 * (b['worker_time'] - a['worker_time']))
    out.update(mean_TBT_ms=statistics.mean(parent), mean_worker_emission_TBT_ms=statistics.mean(worker),
        mean_TTFT_ms=statistics.mean(row['ttft_ms'] for row in rows),
        mean_request_TPOT_ms=statistics.mean(row['tpot_ms'] for row in rows),
        token_budget=config['token_budget'], max_num_seqs=config['D_slots'],
        graph_mode=config['ar_cudagraph_mode'], graph_capture_sizes=config['explicit_cudagraph_capture_sizes'],
        total_input_plus_output_tokens_s=out['input_tokens_s'] + out['output_tokens_s'],
        evidence='CORRECTNESS_ONLY' if config.get('correctness_only') else 'MEASURED',
        measurement_class=('CORRECTNESS_INSTRUMENTED' if config.get('correctness_only') else
            'INSTRUMENTED_DIAGNOSTIC' if config.get('iteration_trace') or config.get('nsys_bounded_capture')
                or config.get('api28_bounded_activity') else 'FINITE_COHORT'),
        source_publication='db300024e29e6adec845bec96f90d6efcdbd0f72')
    out['comparison_eligibility'] = config.get('comparison_eligibility', 'STRICT_ORIGINAL_TP1_OUTPUT_GATE_REQUIRED')
    out['output_reference_name'] = config.get('reference_name', 'FROZEN_ORIGINAL_TP1_REFERENCE')
    if config.get('comparison_eligibility') == 'DIAGNOSTIC_WITHIN_TP2_ONLY_TP1_PARITY_FAILED':
        out.update(measurement_class='TP2_DIAGNOSTIC_NOT_PRIMARY_COMPARISON',
                   original_TP1_parity='FAILED', primary_comparison_eligible=False)
    geometry = collections.Counter()
    for path in folder.glob('measured_geometry_*.json'):
        geometry.update(json.loads(path.read_text())['counts'])
    out['batch_geometry_counts'] = dict(geometry)
    out['prefilling_input_batches'] = sum(n for key, n in geometry.items() if json.loads(key)['prefilling'])
    decode_sizes = [json.loads(key)['requests'] for key, n in geometry.items()
                    if not json.loads(key)['prefilling'] for _ in range(n)]
    out['decode_batch_p50'] = percentile(decode_sizes, .5)
    out['decode_batch_p95'] = percentile(decode_sizes, .95)
    mixed = 0
    for path in folder.glob('GPU_phase_mask_*.json'):
        mixed += json.loads(path.read_text())['counters']['mixed_fullgraph_replays']
    out['mixed_fullgraph_replays'] = mixed
    chunks = []
    for path in folder.glob('native_*.json'):
        value = json.loads(path.read_text())
        chunks.extend(s['count'] for s in value.get('prompt_spans', []) if s['request_id'].split('-')[0].startswith('q'))
    out['actual_prefill_chunk_tokens'] = dict(count=len(chunks), mean=statistics.mean(chunks) if chunks else None,
        p50=percentile(chunks, .5), p95=percentile(chunks, .95), maximum=max(chunks, default=None))
    out['output_gate'] = output_gate(name)
    write(folder / 'metrics.json', out)
    write(ROOT / 'analysis' / (name + '_metrics.json'), out)
    return out


def table():
    rows = []
    for path in sorted((ROOT / 'raw').glob('*/dispatcher_result.json')):
        if json.loads(path.read_text()).get('complete'):
            value = metrics(path.parent.name)
            rows.append({k: value.get(k) for k in (
                'name', 'output_cap', 'offered_QPS', 'token_budget', 'max_num_seqs',
                'mean_TTFT_ms', 'TTFT_p95_ms', 'mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms',
                'output_tokens_s', 'completed_QPS', 'maximum_ITL_ms', 'decode_batch_p50', 'decode_batch_p95')})
    write(ROOT / 'analysis/current_metric_table.json', rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*')
    args = parser.parse_args()
    if args.names:
        for name in args.names:
            value = metrics(name)
            print(json.dumps({k: value.get(k) for k in ('name', 'complete', 'mean_TBT_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'output_gate')}))
    else:
        print(json.dumps(table()))
