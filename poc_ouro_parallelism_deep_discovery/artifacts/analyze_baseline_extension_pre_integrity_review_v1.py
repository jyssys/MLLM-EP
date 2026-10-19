"""Score complete baseline extensions. Never admit partial arms as benchmarks."""
import argparse
import collections
import csv
import datetime as dt
import gzip
import json
import math
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import MODELS, PROJECT, ROOT, read, sha
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval
from poc_async_recurrent_pp.scoring import score


ARMS = [('1.4B', 'fixed', 4), ('1.4B', 'adaptive', 5), ('2.6B', 'fixed', 6), ('2.6B', 'adaptive', 7)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_metrics(frozen, value, size, mode, gpu, protocol_hash, evaluate=score):
    require((value['id'], value['dataset'], value['size'], value['mode'], value['physical_gpu']) ==
            (frozen['id'], frozen['dataset'], size, mode, gpu), 'Request identity differs')
    require(value['revision'] == MODELS[size]['revision'], 'Request model revision differs')
    require(value['source_protocol_sha256'] == protocol_hash, 'Request protocol hash differs')
    require(value['cap'] == 8192 and value['temperature'] == 0 and value['EOS'] == 0, 'Request sampling differs')
    require(value['evidence_class'] == 'QUALITY_ONLY', 'Request evidence class differs')
    tokens, depths = value['output_ids'], value['depth_events']
    require(value['generated_tokens'] == len(tokens) and 0 < len(tokens) <= 8192, 'Output count differs')
    require(all(type(token) is int and 0 <= token < 49152 for token in tokens), 'Invalid output token')
    require(len(depths) == len(tokens) - 1, 'Decode depth count differs')
    require(value['depth_count_matches'] and value['depth_range_matches'], 'Runner depth validation failed')
    require(all(type(item['position']) is int and type(item['depth']) is int and
                item['depth'] in ([4] if mode == 'fixed' else [2, 3, 4]) for item in depths), 'Depth identity differs')
    positions = [item['position'] for item in depths]
    require(len(positions) == len(set(positions)), 'Decode position repeats')
    require(not positions or positions == list(range(len(value['prompt_ids']), len(value['prompt_ids']) + len(positions))),
            'Decode positions do not follow the recorded prompt')
    require(value['finish_reason'] in ('eos', 'stop', 'length'), 'Unknown completion reason')
    require((value['finish_reason'] == 'length') == (len(tokens) == 8192 and tokens[-1] != 0),
            'Length finish metadata differs')
    require((value['finish_reason'] == 'eos') == (tokens[-1] == 0), 'EOS finish metadata differs')
    require(isinstance(value['wall_ms'], (int, float)) and math.isfinite(value['wall_ms']) and value['wall_ms'] > 0,
            'Request wall is invalid')
    evaluation = evaluate(frozen, value)
    evaluation['truncated'] = value['finish_reason'] == 'length'
    evaluation['correct_completed'] = bool(evaluation['answer_matches'] and not evaluation['truncated'])
    depth_values = [item['depth'] for item in depths]
    count = len(depth_values)
    passes = sum(depth_values)
    frequency = collections.Counter(depth_values)
    stats = value['stats']
    require(stats['generated_tokens'] == len(tokens) and stats['recurrent_steps'] == passes
            and stats['recurrent_batches'] == passes and stats['coda_tokens'] == count
            and stats['coda_batches'] == count and stats['fixed_depth_recurrent_steps'] == 4 * count,
            'CDB counters disagree with recorded decode work')
    require(stats['prefill_batches'] == stats['prefill_requests'] == stats['prefill_coda_tokens'] == 1
            and stats['prefill_recurrent_steps'] == 4 * len(value['prompt_ids']), 'Full R4 prefill counters differ')
    native_histogram = {int(key) + 1: amount for key, amount in stats['exit_depth_histogram'].items() if amount}
    require(native_histogram == dict(frequency), 'Native exit histogram differs')
    hits, captures = stats['recurrent_graph_hits'], stats['recurrent_graph_captures']
    require(hits + captures == passes and hits >= 0 and captures >= 0, 'Recurrent graph coverage differs')
    require(stats['scheduler_refills'] == stats['mixed_recurrent_batches'] == 0, 'Unexpected multi-request scheduling')
    return dict(id=frozen['id'], dataset=frozen['dataset'], size=size, mode=mode, physical_gpu=gpu,
        generated_tokens=len(tokens), finish_reason=value['finish_reason'], wall_ms=value['wall_ms'],
        recorded_decode_positions=count, recorded_decode_core_pass_work=passes,
        mean_decode_depth=statistics.mean(depth_values) if count else None,
        decode_depth_variance=statistics.pvariance(depth_values) if count else None,
        R2_positions=frequency[2], R3_positions=frequency[3], R4_positions=frequency[4],
        recurrent_graph_hits=hits, recurrent_graph_captures=captures, source_counters_match=True,
        physical_layer_execution_proxy=MODELS[size]['layers'] * passes,
        full_R4_decode_pass_reference=4 * count,
        normalized_recorded_decode_work=passes / (4 * count) if count else None,
        **evaluation)


def domain_summary(rows):
    require(bool(rows) and len({row['id'] for row in rows}) == len(rows), 'Summary needs unique requests')
    bits = [int(row['correct_completed']) for row in rows]
    with_depth = [row for row in rows if row['recorded_decode_positions']]
    positions = sum(row['recorded_decode_positions'] for row in rows)
    passes = sum(row['recorded_decode_core_pass_work'] for row in rows)
    request_means = [row['mean_decode_depth'] for row in with_depth]
    request_work = [row['normalized_recorded_decode_work'] for row in with_depth]
    quality_ci = interval(bits)
    depth_ci = interval(request_means)
    work_ci = interval(request_work)
    return dict(size=rows[0]['size'], mode=rows[0]['mode'], dataset=rows[0]['dataset'], N=len(rows), correct=sum(bits),
        accuracy_percent=100 * statistics.mean(bits), accuracy_CI_percent=[100 * value for value in quality_ci],
        parsing_failures=sum(not row['parsed'] for row in rows), truncated=sum(row['truncated'] for row in rows),
        EOS=sum(row['finish_reason'] == 'eos' for row in rows), frozen_stops=sum(row['finish_reason'] == 'stop' for row in rows),
        total_generated_tokens=sum(row['generated_tokens'] for row in rows),
        median_generated_tokens=statistics.median(row['generated_tokens'] for row in rows),
        total_recorded_decode_positions=positions, total_recorded_decode_core_pass_work=passes,
        token_weighted_mean_decode_depth=passes / positions if positions else None,
        request_weighted_mean_decode_depth=statistics.mean(request_means) if request_means else None,
        request_bootstrap_mean_depth_CI=depth_ci, N_requests_with_decode=len(with_depth),
        request_weighted_normalized_decode_work=statistics.mean(request_work) if request_work else None,
        request_bootstrap_normalized_work_CI=work_ci,
        R2_position_fraction=sum(row['R2_positions'] for row in rows) / positions if positions else None,
        R3_position_fraction=sum(row['R3_positions'] for row in rows) / positions if positions else None,
        R4_position_fraction=sum(row['R4_positions'] for row in rows) / positions if positions else None,
        median_request_wall_ms=statistics.median(row['wall_ms'] for row in rows),
        evidence_class='QUALITY_ONLY', candidate_quality='NOT_RUN', architecture_speedup='NOT_RUN')


def analyze(root):
    root = Path(root)
    protocol_path = root / 'artifacts/baseline_extension_protocol_v1.json'
    contract_path = root / 'artifacts/baseline_extension_analysis_contract_v1.json'
    protocol, contract = read(protocol_path), read(contract_path)
    hashes = {str(path.relative_to(root)): sha(path) for path in (protocol_path, contract_path)}
    for path, expected in contract['source_sha256'].items():
        require(sha(root.parent / path) == expected, 'Frozen scorer or runner changed')
        hashes[str(root.parent / path)] = expected
    protocol_hash = hashes[str(protocol_path.relative_to(root))]
    expected_rows = {row['id']: row for row in protocol['rows']}
    require(len(expected_rows) == protocol['N'] == 64, 'Frozen extension IDs differ')
    old = read(root / 'artifacts/probe_protocol.json')
    require(not set(expected_rows) & {row['id'] for row in old['rows']}, 'Original campaign ID entered extension')
    records, jobs, summaries, paired = [], [], [], []
    for size, mode, gpu in ARMS:
        short = size.replace('.', '').replace('B', '')
        directory = root / f'outputs/baseline_extension_base{short}_{mode}_gpu{gpu}_v1'
        sub = directory / 'baseline_extension'
        status_path, guard_path = sub / 'status.json', directory / 'guard_status.json'
        states, reasons = {}, []
        for key, path in [('runner', status_path), ('guard', guard_path)]:
            if not path.exists():
                reasons.append(key + ' not complete')
            else:
                try:
                    states[key] = read(path)
                    hashes[str(path.relative_to(root))] = sha(path)
                except json.JSONDecodeError:
                    reasons.append(key + ' partial JSON')
                    continue
                if states[key].get('success') is not True:
                    reasons.append(key + ' success is not true')
        if states.get('guard', {}).get('success') is True and states['guard'].get('exit_code') != 0:
            reasons.append('guard exit is not zero')
        present = len(list(sub.glob('*.json.gz'))) if sub.exists() else 0
        jobs.append(dict(size=size, mode=mode, physical_gpu=gpu, source=str(directory.relative_to(root)),
            status='NOT_AVAILABLE' if reasons else 'AVAILABLE', reasons=reasons, raw_request_files_present=present))
        if reasons:
            continue
        require(states['runner']['N_completed'] == states['runner']['N_expected'] == 64, 'Completed arm count differs')
        require(states['runner']['source_protocol_sha256'] == protocol_hash, 'Completed arm protocol hash differs')
        require(read(sub / 'frozen_protocol.json') == protocol, 'Saved execution protocol differs')
        paths = sorted(sub.glob('*.json.gz'))
        require(len(paths) == 64, 'Completed arm has missing or extra request files')
        arm, ids = [], set()
        for path in paths:
            hashes[str(path.relative_to(root))] = sha(path)
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                value = json.load(stream)
            require(value['id'] in expected_rows and value['id'] not in ids, 'Request is unexpected or repeated')
            ids.add(value['id'])
            row = request_metrics(expected_rows[value['id']], value, size, mode, gpu, protocol_hash)
            row['source'] = str(path.relative_to(root))
            arm.append(row)
        require(ids == set(expected_rows), 'Completed arm ID set differs')
        records.extend(arm)
        for domain in ('gsm8k', 'math500'):
            selected = [row for row in arm if row['dataset'] == domain]
            require(len(selected) == 32, 'Domain sample count differs')
            summaries.append(domain_summary(selected))
    for size in MODELS:
        for domain in ('gsm8k', 'math500'):
            arms = {mode: {row['id']: row for row in records if
                    (row['size'], row['mode'], row['dataset']) == (size, mode, domain)} for mode in ('fixed', 'adaptive')}
            if not all(len(values) == 32 for values in arms.values()):
                continue
            ids = sorted(arms['fixed'])
            require(set(ids) == set(arms['adaptive']), 'Paired domain ID sets differ')
            differences = [int(arms['adaptive'][key]['correct_completed']) - int(arms['fixed'][key]['correct_completed']) for key in ids]
            paired.append(dict(size=size, dataset=domain, N_pairs=32,
                adaptive_minus_fixed_accuracy_pp=100 * statistics.mean(differences),
                paired_bootstrap_95_CI_pp=[100 * value for value in interval(differences)],
                fixed_correct=sum(arms['fixed'][key]['correct_completed'] for key in ids),
                adaptive_correct=sum(arms['adaptive'][key]['correct_completed'] for key in ids),
                interpretation='Original operating-point comparison. Not a candidate quality change or noninferiority proof.'))
    return dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), jobs=jobs, records=records,
        summaries=summaries, paired_operating_points=paired, input_sha256=hashes, evidence_class='QUALITY_ONLY',
        original_held_out='UNTOUCHED in this exact ID audit', candidate_quality='NOT_RUN', architecture_speedup='NOT_RUN',
        uncertainty='Request-level resampling, separately per domain/arm. New32/domain baseline cohort is not original candidate confirmation.',
        work_scope='Decode core-pass work excludes prefill, first output token, attention-length effects, norms outside recorded passes, gate, sampling and other costs.',
        time_scope='Independent one-GPU jobs run concurrently. Request walls cannot enter a same-P architecture-speedup comparison.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new analysis directory')
    value = analyze(ROOT)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for name in ('records', 'summaries', 'paired_operating_points'):
        rows = value[name]
        if rows:
            with (args.output / f'{name}.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(dict(completed_arms=sum(job['status'] == 'AVAILABLE' for job in value['jobs']),
                          records=len(value['records']), summaries=len(value['summaries']))))


if __name__ == '__main__':
    main()
