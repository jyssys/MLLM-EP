"""Describe paired Host-path changes and Decode load without inferring GPU idle."""
import argparse
import csv
import math
from bisect import bisect_right
from collections import defaultdict
from .common import *
from .cost_analysis import quantiles
from .runtime_observations import records


PARTS = (
    'arrival_to_admit', 'admit_to_P_start_receipt',
    'P_start_to_release_receipt', 'P_release_to_wire_commit_receipt',
    'wire_commit_to_D_received_receipt', 'D_received_to_output_delivery',
)


def durations(arrival, output, created, timings):
    keys = ('admitted', 'prefill_started', 'prefill_compute_completed',
            'prefill_submitted', 'd_received')
    if any(key not in timings for key in keys):
        return None
    edges = [arrival, *(created+timings[key] for key in keys), output]
    return dict(zip(PARTS, [1000*(b-a) for a,b in zip(edges, edges[1:])]))


def decode_bins(rows):
    """Count live Host-visible Decode requests at the preceding output."""
    starts = sorted(row['first_token'] for row in rows)
    ends = sorted(row['finish'] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        for event in row['token_events']:
            if event['first']:
                continue
            start, end = event['previous_time'], event['time']
            concurrent = bisect_right(starts, start)-bisect_right(ends, start)
            assert concurrent >= 1 and end >= start
            groups[concurrent].append(1000*(end-start))
    return {key:quantiles(values) for key,values in sorted(groups.items())}


def load_cell(name):
    cfg_path = ROOT/'configs'/(name+'.json')
    cfg = json.loads(cfg_path.read_text())
    folder = ROOT/'raw'/name
    rows = sorted((row for row in records(folder/'requests.jsonl') if not row.get('warmup')),
                  key=lambda row:row['sample_id'])
    assert len(rows) == cfg['count'] and analysis().metrics(name)['complete']
    native = {row['request_id']:row for row in json.loads((folder/'stats.json').read_text())['metrics']}
    result = []
    for row in rows:
        prefix = native.get(f'r0_q{row["sample_id"]}')
        if prefix is None:
            continue
        parts = durations(row['arrival'], row['first_token'], prefix['created_monotonic'], prefix['timings'])
        if parts is not None:
            assert math.isclose(sum(parts.values()), row['ttft_ms'], rel_tol=1e-6, abs_tol=0.001)
            result.append(dict(sample_id=row['sample_id'], ttft_ms=row['ttft_ms'], **parts))
    return cfg, rows, result


def compare(reference, candidate):
    a_cfg, a_raw, a_rows = load_cell(reference)
    b_cfg, b_raw, b_rows = load_cell(candidate)
    for key in ('model', 'model_revision', 'length', 'depth', 'cap', 'count',
                'arrival_seed', 'arrival_offsets', 'seed', 'gpus'):
        assert a_cfg[key] == b_cfg[key], key
    assert [(r['sample_id'],r['input_ids']) for r in a_raw] == [
        (r['sample_id'],r['input_ids']) for r in b_raw]
    a_index = {r['sample_id']:r for r in a_rows}
    diffs = [dict(sample_id=r['sample_id'], **{key:r[key]-a_index[r['sample_id']][key]
             for key in (*PARTS, 'ttft_ms')}) for r in b_rows if r['sample_id'] in a_index]
    report = dict(reference=reference, candidate=candidate, paired_receipts=len(diffs),
        expected_requests=a_cfg['count'], arrival_seed=a_cfg['arrival_seed'],
        candidate_minus_reference_ms={key:quantiles([r[key] for r in diffs])
                                       for key in (*PARTS, 'ttft_ms')},
        negative_component_counts={
            name:{key:sum(row[key]<0 for row in rows) for key in PARTS}
            for name,rows in ((reference,a_rows),(candidate,b_rows))},
        delivered_ITL_by_live_Decode_at_previous_output={
            reference:decode_bins(a_raw), candidate:decode_bins(b_raw)},
        scope=('Host milestones on matched requests. Each signed partition sums to request TTFT. '
            'Means over the same pairs can be added; percentiles cannot. Host receipt order can '
            'differ from device execution order. The partition does not measure Service TTFT, '
            'exclusive compute time, DMA time, or counterfactual P completion.'),
        decode_scope=('Live Decode counts include requests from their first output delivery '
            'through their last output delivery. These counts are not physical batch sizes or '
            'Decode-ready queue lengths. Conditioning on them is descriptive, not causal.'))
    output = ROOT/'paired_host_attribution'
    output.mkdir(exist_ok=True)
    label = candidate+'_vs_'+reference
    write(output/(label+'.json'), report)
    with (output/(label+'.csv')).open('w') as handle:
        writer = csv.DictWriter(handle, ['sample_id', *PARTS, 'ttft_ms'])
        writer.writeheader()
        writer.writerows(diffs)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('candidates', nargs='+')
    args = parser.parse_args()
    initialize()
    for name in args.candidates:
        print(json.dumps(compare(args.reference, name)))


if __name__ == '__main__':
    main()
