"""Summarize completed fixed-depth native cohorts. No candidate speedup claim."""
import argparse
from collections import Counter
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import table, svg


def summarize(directory):
    status = read(directory / 'status.json')
    assert status['success']
    contract = read(directory / 'protocol.json')
    protocol = contract['protocol']
    raw = [read(path) for path in sorted(directory.glob('C*_N*_repeat*.json'))]
    expected = len(protocol['cohort_sizes']) * len(protocol['contexts']) * protocol['repeats']
    assert len(raw) == expected
    assert len({(r['C'], r['context'], r['repeat']) for r in raw}) == expected
    rows, traces = [], read(directory / 'schedule_trace.json')
    assert len(traces) == len(protocol['cohort_sizes']) * len(protocol['contexts'])
    for count in protocol['cohort_sizes']:
        for context in protocol['contexts']:
            selected = [r for r in raw if r['C'] == count and r['context'] == context]
            assert len(selected) == protocol['repeats']
            assert all(r['generated_tokens'] == count * protocol['generated_tokens'] for r in selected)
            trace = next(t for t in traces if t['C'] == count and t['context'] == context)
            scheduled, samples = [], []
            for rank in trace['ranks']:
                scheduled.append(dict(Counter(e['scheduled_request_count'] for e in rank['events']
                                               if e['event'] == 'SCHEDULED_FORWARD')))
                samples.append(dict(Counter(e['sampled_tensor_numel'] for e in rank['events']
                                             if e['event'] == 'NATIVE_SAMPLE')))
            rows.append(dict(size=raw[0]['size'], tp=raw[0]['tp'], C=count, context=context,
                repeats=len(selected), median_cohort_makespan_ms=statistics.median(r['wall_ms'] for r in selected),
                median_cohort_output_TPS=statistics.median(r['cohort_output_TPS'] for r in selected),
                minimum_wall_ms=min(r['wall_ms'] for r in selected),
                maximum_wall_ms=max(r['wall_ms'] for r in selected),
                generated_tokens=count * protocol['generated_tokens'],
                source=str(directory.relative_to(ROOT)), evidence_class='LIVE_E2E',
                metric='Finite submitted-cohort makespan and output TPS. Constant-token hardware work.',
                natural_quality='NOT_RUN'))
            rows[-1]['scheduled_forward_count_histograms'] = json.dumps(scheduled, sort_keys=True)
            rows[-1]['native_sample_numel_histograms'] = json.dumps(samples, sort_keys=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    completed, unavailable = [], []
    for size in MODELS:
        short = size.replace('.', '').replace('B', '')
        for tp in (1, 2, 4):
            directory = ROOT / f'outputs/native_cohort_base{short}_P{tp}_v1'
            guard, status = directory / 'guard_status.json', directory / 'status.json'
            if not guard.exists() or not status.exists() or not read(guard).get('success'):
                unavailable.append(str(directory.relative_to(ROOT)))
                continue
            rows.extend(summarize(directory))
            completed.append(str(directory.relative_to(ROOT)))
    assert rows
    args.output.mkdir(parents=True)
    lookup = {(r['size'], r['tp'], r['C'], r['context']): r for r in rows}
    for row in rows:
        native = lookup.get((row['size'], 1, row['C'], row['context']))
        row['TP1_median_over_TP_median_makespan_ratio'] = (
            native['median_cohort_makespan_ms'] / row['median_cohort_makespan_ms'] if native else None)
        one = lookup.get((row['size'], row['tp'], 1, row['context']))
        row['TPS_over_same_TP_C1'] = row['median_cohort_output_TPS'] / one['median_cohort_output_TPS']
    table(args.output / 'summary.csv', rows)
    write(args.output / 'summary.json', dict(rows=rows, completed_jobs=completed, unavailable=unavailable,
        quality='NOT_RUN. Candidate performance NOT_RUN.',
        scope='Generic fixed-R4 native cohort baseline. Model jobs execute sequentially.',
        uncertainty='Five repetitions of two repeated calibration prompt patterns. C1 uses the first pattern only. No population confidence claim.',
        admission='Scheduled membership and sampler tensor counts do not certify equal progress or committed-output row identity.',
        comparisons='Ratios compare medians within this same constant-token protocol. Do not compare to older changing-teacher E2E values.'))
    for size in MODELS:
        for context in (1024, 4096):
            selected = [r for r in rows if r['size'] == size and r['context'] == context]
            if not selected:
                continue
            for field, label in [('median_cohort_makespan_ms', 'Cohort makespan, ms'),
                                 ('median_cohort_output_TPS', 'Finite-cohort output tokens/s')]:
                series = [dict(name=f'TP{tp}', points=[(r['C'], r[field]) for r in selected if r['tp'] == tp])
                          for tp in sorted({r['tp'] for r in selected})]
                svg(args.output / f'{size}_N{context}_{field}.svg', series,
                    f'{size} fixed R4 native control, N={context}', 'Submitted cohort C', label)
    print(json.dumps(dict(rows=len(rows), completed_jobs=len(completed), unavailable=len(unavailable))))


if __name__ == '__main__':
    main()
