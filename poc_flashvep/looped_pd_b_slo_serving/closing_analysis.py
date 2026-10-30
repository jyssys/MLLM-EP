"""Rescore saved observations. This script never starts GPU inference."""
import csv
import json

from .common import ROOT, REPO, write
from .jobs import output_rows


def records(folder):
    return [row for line in (folder / 'requests.jsonl').read_text().splitlines()
            for row in [json.loads(line)] if not row.get('warmup')]


def flags(row, predicate, lock):
    normal = row.get('finish_reason') in ('eos', 'stop', 'length') and row.get('output_tokens', 0) > 0
    gaps = [1000 * (event['time'] - event['previous_time']) for event in row['token_events']
            if not event['first'] and event['new_tokens'] >= 1]
    if predicate == 'historical_absolute150':
        return normal and row['ttft_ms'] <= 5000 and max(gaps, default=0) <= 150 and row['e2e_ms'] <= 30000
    return normal and row['ttft_ms'] <= lock['TTFT_ms'] and row['tpot_ms'] <= lock['TPOT_ms']


def main():
    previous = REPO / 'poc_flashvep/reports/looped_pd_abc'
    old = list(csv.DictReader((previous / 'paired_SLO_comparisons.csv').open()))
    lock = json.loads((ROOT / 'configs/slo_lock.json').read_text())
    rows = []
    for model in ('ouro', 'huginn'):
        for baseline in ('DP', 'DP_Mixed', 'PD', 'PD_G1'):
            chosen = next(row for row in old if row['model'] == model and row['candidate'] == 'B'
                          and row['baseline'] == baseline and row['input_tokens'] == '512'
                          and float(row['ITL_limit_ms']) == 150 and float(row['TTFT_limit_ms']) == 5000)
            pairs = json.loads(chosen['raw_pairs'])
            for side, method in ((0, baseline), (1, 'B')):
                data = [records(previous / 'raw' / pair[side]) for pair in pairs]
                total = sum(len(group) for group in data)
                duration = sum(max(row['finish'] for row in group) - min(row['arrival'] for row in group) for group in data)
                for predicate in ('historical_absolute150', 'current_primary'):
                    good = sum(flags(row, predicate, lock[model]) for group in data for row in group)
                    rows.append(dict(trace_family='prior_closed_loop_P512_C4_cap32', model=model, method=method,
                                     comparison_baseline=baseline, predicate=predicate, starts=len(pairs),
                                     correct_latency_completions=good, total=total, attainment_percent=100*good/total,
                                     denominator_seconds=duration, goodput_requests_s=good/duration,
                                     evidence='TRACE_RESCORING_OF_MEASURED_OUTPUTS_NOT_NEW_GPU_EXECUTION',
                                     denominator_scope='Sum of original cohort spans. Not open-loop capacity.'))
        for method in ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'B'):
            rate = 1. if model == 'ouro' else .2
            name = f'{model}_W1_{method}_N2_rate{rate:g}_seed2026103001'
            folder = ROOT / 'raw' / name
            if not (folder / 'window.json').exists():
                continue
            data = output_rows(name)
            window = json.loads((folder / 'window.json').read_text())['window_seconds']
            for predicate in ('historical_absolute150', 'current_primary'):
                good = sum(flags(row, predicate, lock[model]) for row in data)
                rows.append(dict(trace_family='current_original_epoch_W1_cap128', model=model, method=method,
                                 comparison_baseline='shared_W1_seed2026103001', predicate=predicate, starts=1,
                                 correct_latency_completions=good, total=len(data), attainment_percent=100*good/len(data),
                                 denominator_seconds=window, goodput_requests_s=good/window,
                                 evidence='TRACE_RESCORING_OF_MEASURED_OUTPUTS_NOT_NEW_GPU_EXECUTION',
                                 denominator_scope='Frozen open-loop arrival window. Original B outputs fail parity.'))
    with (ROOT / 'historical_vs_current_slo_rescoring.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for model in ('ouro', 'huginn'):
        print(model)
        for row in rows:
            if row['model'] == model and (row['trace_family'].startswith('current') or row['comparison_baseline'] == 'PD_G1'):
                print(row['trace_family'], row['method'], row['predicate'],
                      f"{row['correct_latency_completions']}/{row['total']}", round(row['goodput_requests_s'], 6))
    # Record the multiplicative decomposition of the old matched goodput ratio.
    decomposition = []
    for model in ('ouro', 'huginn'):
        pair = [row for row in rows if row['model'] == model and row['comparison_baseline'] == 'PD_G1'
                and row['trace_family'].startswith('prior') and row['predicate'] == 'historical_absolute150']
        before, after = next(x for x in pair if x['method'] == 'PD_G1'), next(x for x in pair if x['method'] == 'B')
        decomposition.append(dict(model=model, qualified_count_ratio=after['correct_latency_completions']/before['correct_latency_completions'],
                                  cohort_span_ratio=before['denominator_seconds']/after['denominator_seconds'],
                                  goodput_ratio=after['goodput_requests_s']/before['goodput_requests_s'],
                                  evidence='STRUCTURAL_DECOMPOSITION_OF_MEASURED_COUNTS_AND_COHORT_SPANS'))
    write(ROOT / 'configs/historical_goodput_decomposition.json', decomposition)
    print('DECOMPOSITION', decomposition)


if __name__ == '__main__':
    main()
