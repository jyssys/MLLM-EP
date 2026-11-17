"""Use frozen metric definitions and retain accounting and geometry controls."""
import collections
import json
import statistics
from .common import BASE, ROOT, bind_previous, write


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def metrics(name):
    bind_previous()
    from ..dense_ar_slo_layer_pd import analyze as frozen
    from ..duetserve_agg_decode import analyze as inherited
    inherited.ROOT = frozen.ROOT = ROOT
    value = frozen.metrics(name)
    if not value.get('complete'):
        return value
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    folder = ROOT / 'raw' / name
    rows = records(folder / 'requests.jsonl')
    intervals, worker_intervals = [], []
    for row in rows:
        events = row['token_events']
        assert len(events) == row['output_tokens'] and events[0]['count'] == 1
        for a, b in zip(events, events[1:]):
            assert b['count'] - a['count'] == 1
            intervals.append(1000 * (b['time'] - a['time']))
            worker_intervals.append(1000 * (b['worker_time'] - a['worker_time']))
    value.update(mean_TTFT_ms=statistics.mean(row['ttft_ms'] for row in rows),
        mean_TBT_ms=statistics.mean(intervals), mean_worker_emission_TBT_ms=statistics.mean(worker_intervals),
        mean_request_TPOT_ms=statistics.mean(row['tpot_ms'] for row in rows),
        token_budget=cfg['token_budget'], max_num_seqs=cfg['D_slots'],
        graph_mode=cfg['ar_cudagraph_mode'], graph_capture_sizes=cfg.get('explicit_cudagraph_capture_sizes'),
        output_gate=inherited.output_gate(name),
        evidence='CORRECTNESS_ONLY' if cfg.get('correctness_only') else 'MEASURED',
        measurement_class='CORRECTNESS_INSTRUMENTED' if cfg.get('correctness_only') else 'FINITE_COHORT')
    geometry = collections.Counter()
    for path in folder.glob('measured_geometry_*.json'):
        geometry.update(json.loads(path.read_text())['counts'])
    value['batch_geometry_counts'] = dict(geometry)
    sizes = [json.loads(key)['requests'] for key, count in geometry.items()
             if not json.loads(key)['prefilling'] for _ in range(count)]
    value['decode_batch_p50'] = frozen.percentile(sizes, .5)
    value['decode_batch_p95'] = frozen.percentile(sizes, .95)
    commits = records(folder / 'actual_cut_commits.jsonl')
    cuts = {rid: row['offloaded_layers'] for row in commits for rid in row['requests']}
    source_groups = [row['requests'] for row in commits]
    admissions = records(folder / 'D_host_admission.jsonl')
    destination_groups = collections.defaultdict(list)
    for row in admissions:
        destination_groups[row['cohort']].append(row['request_id'])
    # The token-axis file is a JSON object, not JSONL.
    token_file = folder / 'token_axis_prefill.json'
    token_data = json.loads(token_file.read_text())['records'] if token_file.exists() else []
    token_source = {r['request_id']: r for r in token_data
                    if r['event'] == 'SOURCE_PREFIX_READY' and r['request_id'].startswith('q')}
    work = dict(P_GEMM_token_layer_rows=0, D_GEMM_token_layer_rows=0,
                valid_KV_payload_bytes=0, hidden_residual_payload_bytes=0,
                completed_migrated_requests=0, completed_layer_work=0,
                P_unused_prefix_output_tokens=0,
                D_native_prompt_replay_queries=0,
                Full36_generated_decode_query_rows=sum(r['output_tokens']-1 for r in rows))
    for row in rows:
        rid = 'q' + str(row['sample_id'])
        length = row['input_tokens']
        m = cuts.get(rid, 0)
        if cfg.get('token_axis'):
            transferred = token_source[rid]
            prefix = transferred['prefix_tokens']
            work['P_GEMM_token_layer_rows'] += prefix * 36
            work['D_GEMM_token_layer_rows'] += (length - prefix) * 36
            work['valid_KV_payload_bytes'] += prefix * 36 * 4096
            work['completed_migrated_requests'] += 1
            work['P_unused_prefix_output_tokens'] += 1
        else:
            work['P_GEMM_token_layer_rows'] += length * (36 - m)
            work['D_GEMM_token_layer_rows'] += length * m
            work['valid_KV_payload_bytes'] += length * (36 - m) * 4096
            work['hidden_residual_payload_bytes'] += length * 4096 * 2 * 2 if m else 0
            work['completed_migrated_requests'] += int(m > 0)
            work['completed_layer_work'] += m
            work['P_unused_prefix_output_tokens'] += int(m>0 or not cfg.get('reuse_prefill_token'))
            work['D_native_prompt_replay_queries'] += int(m==0 and not cfg.get('reuse_prefill_token'))
    assert work['P_GEMM_token_layer_rows'] + work['D_GEMM_token_layer_rows'] == sum(r['input_tokens'] * 36 for r in rows)
    handoffs=records(folder/'native_handoffs.jsonl')
    if len(handoffs)==len(rows):
        assert all(len(r['params']['remote_block_ids'])==1 for r in handoffs)
        work['source_exported_cache_blocks']=sum(len(r['params']['remote_block_ids'][0]) for r in handoffs)
    else:
        work['source_exported_cache_blocks']=None
    work['D_full_prompt_and_output_blocks_structural']=sum((r['input_tokens']+r['output_tokens']+15)//16 for r in rows)
    value.update(method=cfg.get('evaluation_arm'), publication_parent=BASE,
        actual_cut_histogram=dict(collections.Counter(cuts.values())), work_accounting=work,
        source_groups=source_groups, destination_groups=list(destination_groups.values()),
        accounting_scope='STRUCTURAL_ACCOUNTING over actual completed requests, actual commits and prefix exports. Rows are not FLOPs or measured GPU time. Payload is valid logical state, not protocol headers or padding. Exported source block counts are actual metadata, not exclusive physical-HBM traffic or a retention-time integral. Equal valid KV bytes do not imply equal source KV-credit occupancy: layer cuts still allocate the full36-layer cache group.',
        preplanned_source_groups=bool(cfg.get('preplanned_source_groups')),
        resource_status='FINITE_COHORT_NOT_SUSTAINABLE_CAPACITY')
    if cfg['arm']=='PD_G1':
        from .transport_audit import audit
        value['transport_audit']=audit(name,work)
    else:
        value['work_accounting']=None
        value['accounting_scope']='Aggregated replicas have no P/D prefill partition. Combined logical prompt work is36 times the total input tokens. No GPU0/GPU1 assignment is inferred from the PD fields.'
    value['policy_information']='OFFLINE_FUTURE_SELECTED_DIAGNOSTIC' if cfg.get('offline_selected_schedule') else 'ONLINE_OR_FIXED_BEFORE_RUN'
    value['primary_eligible']=not cfg.get('offline_selected_schedule') and not cfg.get('correctness_only')
    if cfg.get('execution_events') or cfg.get('boundary_events'):
        value['measurement_class'] = 'CUDA_EVENT_ENCLOSURE_DIAGNOSTIC_NOT_KERNEL_UNION'
    write(ROOT / 'analysis' / (name + '_metrics.json'), value)
    print('METRICS', name, value['output_gate']['passed'],
          round(value['TTFT_p95_ms'], 3), round(value['ITL_p99_ms'], 3),
          round(value['output_tokens_s'], 3), flush=True)
    return value


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        metrics(name)
