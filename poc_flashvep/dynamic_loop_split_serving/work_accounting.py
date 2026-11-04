"""Separate accepted work, completed work, and structural attention counts."""
from .common import *


def recurrent_geometry(length, depth, axis, cut, prefix):
    if not (length > 0 and depth > 0 and 0 <= cut <= depth and 0 <= prefix <= length):
        raise ValueError('Invalid recurrent geometry')
    pairs = length * (length + 1) // 2
    if axis == 'loop':
        rows = length * (depth - cut)
        transferred_pairs = pairs * (depth - cut)
    elif axis == 'token':
        rows = (length - prefix) * depth
        transferred_pairs = (pairs - prefix * (prefix + 1) // 2) * depth
    else:
        raise ValueError('Unknown split axis')
    return dict(assigned_D_recurrent_token_rows=rows,
        assigned_D_causal_attention_pairs_per_physical_attention_layer=transferred_pairs,
        full_recurrent_token_rows=length * depth,
        full_causal_attention_pairs_per_physical_attention_layer=pairs * depth,
        scope='Structural dense-causal pair counts. These are not measured FLOPs, GPU time, or speedup. Prelude and Coda are excluded.')


def huginn_boundary_geometry(length, depth, axis, cut, prefix):
    core = recurrent_geometry(length, depth, axis, cut, prefix)
    if axis == 'loop':
        prelude_tokens = length if cut == 0 else 0
        coda_KV_tokens = length - 1 if cut < depth else 0
    else:
        prelude_tokens = length - prefix
        coda_KV_tokens = max(0, length - 1 - prefix)
    return dict(D_recurrent_block_token_rows=4 * core['assigned_D_recurrent_token_rows'],
        D_additional_prelude_block_token_rows=2 * prelude_tokens,
        D_additional_prompt_Coda_KV_block_token_rows=2 * coda_KV_tokens,
        scope='Huginn checkpoint has Prelude/Core/Coda layer counts 2/4/2. These are structural block-token rows, not equal-FLOP blocks or measured GPU time. Common first-token Coda, injection, embedding, and other operations are separate.')


def measured_partitions(rows, expected, measurement_begin):
    accepted = {}
    for row in rows:
        if row['seed'] not in expected or row['time'] < measurement_begin:
            continue
        assert row['seed'] not in accepted, 'Count one actual measured boundary per request'
        accepted[row['seed']] = row
    return accepted


def summarize(name):
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
    completed = {json.loads(line)['seed'] for line in (folder / 'requests.jsonl').read_text().splitlines()
                 if not json.loads(line).get('warmup')}
    measurement_begin = json.loads((folder / 'measurement_begin.json').read_text())['time']
    accepted = measured_partitions(
        [json.loads(line) for path in folder.glob('accepted_partitions_*.jsonl')
         for line in path.read_text().splitlines()], expected, measurement_begin)
    payloads = [json.loads(line) for path in folder.glob('migration_*.jsonl')
                for line in path.read_text().splitlines()
                if json.loads(line)['seed'] in expected and json.loads(line)['time'] >= measurement_begin]
    rows = []
    for seed, row in accepted.items():
        geometry = recurrent_geometry(row['prompt_tokens'], row['full_depth'], row['axis'],
                                      row['accepted_cut'], row['accepted_prefix_tokens'])
        assert geometry['assigned_D_recurrent_token_rows'] == row['assigned_D_recurrent_token_rows']
        rows.append(dict(seed=seed, completed=seed in completed, partial=row['partial'],
                         cut=row['accepted_cut'], prefix=row['accepted_prefix_tokens'], **geometry))
        if cfg['model'] == 'huginn':
            rows[-1]['Huginn_boundary_geometry'] = huginn_boundary_geometry(
                row['prompt_tokens'], row['full_depth'], row['axis'], row['accepted_cut'], row['accepted_prefix_tokens'])
    done = [row for row in rows if row['completed']]
    before = json.loads((folder / 'before_measured.json').read_text())
    after = json.loads((folder / 'after_measured.json').read_text())
    counters = {}
    for role in ('prefill', 'decode'):
        workers = [key for key in after if '-' + role + '-' in key]
        assert len(workers) == 1, 'This accounting contract covers one P and one D worker'
        key = workers[0]
        counters[role] = {field: after[key][field] - before[key][field] for field in
            ('issued_loop_rows', 'issued_submissions', 'transport_bytes_sent',
             'transport_completed_count', 'transport_completion_observation_seconds')}
    outputs = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()
               if not json.loads(line).get('warmup')]
    decode_rows = sum((row['output_tokens'] - 1) * cfg['depth'] for row in outputs)
    actual_D_prompt_rows = counters['decode']['issued_loop_rows'] - decode_rows
    assigned_rows = sum(row['assigned_D_recurrent_token_rows'] for row in done)
    assert actual_D_prompt_rows == assigned_rows, 'Assigned offload rows differ from active execution counters'
    snapshot_bytes = sum(row['KV_bytes'] + row['hidden_bytes'] for row in payloads)
    result = dict(name=name, offered=cfg['count'], completed=len(completed & expected),
        accepted_boundaries=len(accepted),
        completed_offloaded_serving_requests=sum(row['partial'] for row in done),
        assigned_D_recurrent_token_rows_on_completed_requests=sum(row['assigned_D_recurrent_token_rows'] for row in done),
        assigned_D_attention_pairs_on_completed_requests=sum(
            row['assigned_D_causal_attention_pairs_per_physical_attention_layer'] for row in done),
        measured_snapshot_KV_bytes=sum(row['KV_bytes'] for row in payloads),
        measured_snapshot_hidden_bytes=sum(row['hidden_bytes'] for row in payloads),
        measured_snapshot_fragments=sum(row['fragments'] for row in payloads),
        measured_snapshot_descriptors=sum(row['descriptors'] for row in payloads),
        measured_worker_counters=counters,
        measured_D_prompt_rows_from_active_counter=actual_D_prompt_rows,
        expected_D_decode_rows=decode_rows,
        measured_transport_payload_bytes=counters['prefill']['transport_bytes_sent'],
        measured_completed_transport_fragments=counters['prefill']['transport_completed_count'],
        snapshot_payload_reconciles_with_transport=(snapshot_bytes == counters['prefill']['transport_bytes_sent']) if payloads else 'NOT_APPLICABLE_NATIVE_STREAMING',
        transport_time_scope='Cumulative host-observed completion age. Not pure GPU DMA time or an exclusive E2E component.',
        submission_scope='Native model-runner submissions, not CUDA kernel or GEMM invocation counts.',
        requests=rows,
        scope='Completion is paired with actual accepted boundaries. Full-depth execution still requires the separate work gate. Native-streaming payloads are not zero when snapshot records are absent.',
        FLOPs='NOT_MEASURED', kernel_invocations='NOT_MEASURED')
    write_derived(folder / 'split_work_and_payload.json', result)
    return result
