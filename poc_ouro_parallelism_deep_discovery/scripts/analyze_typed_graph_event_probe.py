"""Analyze one bounded observer lane. No E2E extrapolation or quality claim."""
import argparse
import statistics
import numpy as np

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, write
from poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = ROOT / args.input, ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    assert read(source / 'guard_status.json')['success'] and read(source / 'status.json')['success']
    data = read(source / 'typed_event_observation.json')
    assert data['P'] == 1
    comparisons = []
    baseline = data['before']['captures'][0]
    a = np.load(baseline['path'])
    for lane in ['observed', 'after']:
        candidate = data[lane]['captures'][0]
        assert baseline['positions'] == candidate['positions'] and baseline['KV_names'] == candidate['KV_names']
        b = np.load(candidate['path'])
        comparisons.append(dict(lane=lane, rank=0,
            values={name: stats(b[name], a[name]) for name in ['logits', 'hidden', 'KV']},
            bitwise_equal={name: bool(np.array_equal(a[name], b[name])) for name in ['logits', 'hidden', 'KV']},
            finite={name: bool(np.isfinite(a[name]).all() and np.isfinite(b[name]).all())
                    for name in ['logits', 'hidden', 'KV']},
            top1_match=(a['logits'].argmax(-1) == b['logits'].argmax(-1)).tolist()))
    envelopes = {lane: [row['duration_ms'] for row in data[lane]['intervals'][0]['records']
                        if row['kind'] == 'graph_replay'] for lane in ['before', 'observed', 'after']}
    assert all(len(rows) == 3 for rows in envelopes.values())
    # Skip first-replay effects in all lanes. Instrumented replay1 includes mutation.
    clean = envelopes['before'][1:] + envelopes['after'][1:]
    observed = envelopes['observed'][1:]
    native_median, instrumented_median = statistics.median(clean), statistics.median(observed)
    rank = data['records'][0]
    assert rank['topology_restored'] and rank['original_torch_replay_used']
    assert rank['total_observed_replays'] == 3 and len(rank['records']) == 1
    internal = rank['records'][0]
    assert internal['typed_dependencies_preserved'] and internal['observed_replays'] == 3
    write(target, dict(input=str(source.relative_to(ROOT)), size=data['size'], P=1, R=4,
        numerical_comparisons=comparisons, graph_replay_envelopes_ms=envelopes,
        clean_later_replay_median_ms=native_median,
        instrumented_later_replay_median_ms=instrumented_median,
        later_replay_difference_ms=instrumented_median-native_median,
        later_replay_ratio=instrumented_median/native_median,
        observed_count=2, clean_count=4, internal_last_replay=internal,
        clean_request_wall_ms={label: [r['request_wall_ns']/1e6 for r in data[label]]
                               for label in ['clean_before', 'clean_after']},
        scope='One request per observer lane. Four controlled outputs, three cached replay calls. Not a population observer-overhead estimate.',
        timing='Internal events describe only replay3 and fused kernel-group cuts. No semantic-pass or request-total estimate.',
        physical_fact='Runtime assertions preserved original edge data and restored the original typed fingerprint.',
        raw_boundary='This run retains the original DOT, fingerprints, and edge-data counts. It does not retain complete before/after typed snapshots.',
        natural_quality='NOT_RUN', candidate_speedup='NOT_RUN', request_extrapolation='FORBIDDEN'))
    print(dict(numerical_exact=all(all(row['bitwise_equal'].values()) for row in comparisons),
               last_kernel_groups_ms=internal['kernel_group_intervals_ms'],
               later_replay_ratio=instrumented_median/native_median))


if __name__ == '__main__':
    main()
