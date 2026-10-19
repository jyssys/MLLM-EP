"""Validate completed native context controls. No GPU or model call occurs."""
import argparse
import csv
import math
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, read, sha, write
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval


def request_metrics(row):
    metrics = row['metrics']
    count = row['generated_tokens']
    assert count == 128 and metrics['num_generation_tokens'] == count and not metrics['is_corrupted']
    # first/last timestamps share a monotonic clock. arrival_time uses another clock.
    elapsed = metrics['last_token_ts'] - metrics['first_token_ts']
    assert elapsed > 0 and row['wall_ms'] > 0 and math.isfinite(row['wall_ms'])
    first = metrics['first_token_latency'] * 1000
    assert 0 <= first < row['wall_ms']
    return dict(native_first_token_latency_ms=first,
                native_decode_span_ms=elapsed * 1000,
                native_decode_TPOT_ms=elapsed * 1000 / (count - 1),
                wall_ms=row['wall_ms'], output_TPS=count * 1000 / row['wall_ms'])


def finite_capture(path):
    import numpy as np
    result = []
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {'logits', 'hidden', 'KV'}
        for name in sorted(data.files):
            value = data[name]
            assert value.size and np.issubdtype(value.dtype, np.floating)
            result.append(dict(tensor=name, shape=list(value.shape), values=int(value.size),
                finite=bool(np.isfinite(value).all()),
                max_absolute=float(np.max(np.abs(value)))))
    assert all(row['finite'] for row in result)
    return result


def plot_series(cells, size, metric):
    result = []
    for tp in [1, 2, 4]:
        points = [(r['context'], r[metric]) for r in cells if r['size'] == size and r['TP'] == tp]
        if points:
            result.append(dict(name=f'TP{tp}', points=points))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    rows, runs, cells, numeric, inputs = [], [], [], [], {}

    def load(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    for size in MODELS:
        for tp in [1, 2, 4]:
            short = size.replace('.', '').replace('B', '')
            directory = ROOT / f'outputs/native_longctx_base{short}_P{tp}_v{args.attempt}'
            required = [directory / name for name in ['guard_status.json', 'status.json', 'ownership.json',
                                                        'engine_config.json', 'numerical_capture.json']]
            if any(not path.exists() for path in required):
                runs.append(dict(size=size, TP=tp, status='INCOMPLETE_OR_NOT_STARTED',
                    clean_trial_files=len(list(directory.glob('case*_repeat*.json')))))
                continue
            guard, status, owners, config, capture = [load(path) for path in required]
            assert guard['success'] and guard['exit_code'] == 0 and status['success']
            assert config['kv_cache_memory_bytes'] * tp == 52 * 1024 ** 3
            assert config['max_model_len'] == config['max_num_batched_tokens'] == 32896
            assert config['hf_overrides']['total_ut_steps'] == 4 and config['async_scheduling']
            assert not config['enable_prefix_caching'] and not config['enable_chunked_prefill']
            assert len(owners) == tp and all(v['cache_slots'] == 4 * MODELS[size]['layers'] for v in owners)
            for rank in capture['capture']:
                path = Path(rank['path'])
                assert path.is_relative_to(directory) and sha(path) == rank['sha256']
                inputs[str(path.relative_to(ROOT))] = rank['sha256']
                assert rank['teacher_steps'] == 'NOT_APPLICABLE'
                numeric.append(dict(size=size, TP=tp, rank=rank['rank'], sampled_finiteness=finite_capture(path),
                                    scope='Four captured logits/hidden calls and selected written KV only. Not full KV or natural quality.'))
            files = sorted(directory.glob('case*_repeat*.json'))
            assert len(files) == status['clean_trials'] == 24
            local = []
            for file in files:
                raw = load(file)
                assert (raw['size'], raw['TP'], raw['R'], raw['C']) == (size, tp, 4, 1)
                value = dict(size=size, TP=tp, context=raw['context'], repeat=raw['repeat'],
                    source_request_id=raw['source_request_id'], source=str(file.relative_to(ROOT)), **request_metrics(raw))
                local.append(value)
                rows.append(value)
            memory = load(directory / 'memory_after.json')
            for context in sorted({r['context'] for r in local}):
                selected = [r for r in local if r['context'] == context]
                assert len(selected) == 6 and len({r['source_request_id'] for r in selected}) == 2
                cell = dict(size=size, TP=tp, context=context, timing_trials=6, frozen_prompt_patterns=2)
                for metric in ['wall_ms', 'native_first_token_latency_ms', 'native_decode_TPOT_ms', 'output_TPS']:
                    cell['median_' + metric] = statistics.median(r[metric] for r in selected)
                cell['job_peak_allocated_max_rank_GiB'] = max(r['peak_allocated'] for r in memory) / 1024 ** 3
                cells.append(cell)
            runs.append(dict(size=size, TP=tp, status='VALIDATED_COMPLETED_CONTROL', clean_trials=24,
                             natural_quality='NOT_RUN', candidate_speedup='NOT_RUN'))
    pairs = []
    for size in MODELS:
        for context in [1024, 8192, 16384, 32768]:
            for tp in [2, 4]:
                a = [r for r in rows if (r['size'], r['context'], r['TP']) == (size, context, 1)]
                b = [r for r in rows if (r['size'], r['context'], r['TP']) == (size, context, tp)]
                if not a or not b:
                    continue
                ratios = []
                for key in sorted({r['source_request_id'] for r in a}):
                    x = statistics.median(r['wall_ms'] for r in a if r['source_request_id'] == key)
                    y = statistics.median(r['wall_ms'] for r in b if r['source_request_id'] == key)
                    ratios.append(x / y)
                pairs.append(dict(size=size, context=context, compared_TP=tp, baseline_TP=1,
                    median_request_speedup_x=statistics.median(ratios),
                    mean_request_speedup_x=statistics.mean(ratios),
                    request_bootstrap_95_CI_x=interval(ratios), N_frozen_prompt_patterns=len(ratios),
                    caution='Two repeated prompt patterns provide a narrow hardware control, not broad workload inference.'))
    output.mkdir(parents=True)
    for name, data in [('trials', rows), ('cells', cells), ('paired_speedup', pairs)]:
        if data:
            with (output / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    for size in MODELS:
        for metric in ['median_wall_ms', 'median_native_first_token_latency_ms', 'median_native_decode_TPOT_ms']:
            series = plot_series(cells, size, metric)
            if series:
                svg(output / f'{size}_{metric}.svg', series,
                    f'{size}: native fixed R4, 128 controlled outputs', 'Prompt tokens', metric)
    write(output / 'summary.json', dict(runs=runs, cells=cells, pairs=pairs, sampled_numerics=numeric,
        input_sha256=inputs, evidence_class='LIVE_E2E', candidate='NONE', natural_quality='NOT_RUN',
        timing_boundary='Native generate API wall. Native timestamp submetrics retain their own explicit boundaries.',
        finiteness='Captured subset only. This does not establish full cache correctness or natural quality.',
        memory_scope='Peak allocation spans the whole job and all context cases. It is not measured separately per context.',
        bootstrap='4000 paired request resamples after within-pattern medians. Only two frozen prompt patterns.'))
    print('LONG_CONTEXT_ANALYSIS', len(rows), 'trials', len(cells), 'cells', len(numeric), 'rank captures')


if __name__ == '__main__':
    main()
