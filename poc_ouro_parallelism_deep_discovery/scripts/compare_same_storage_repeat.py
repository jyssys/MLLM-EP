"""Compare two fresh runs of the same storage arm. No alias-effect claim."""
import argparse
import statistics
from pathlib import Path
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import read, write, sha
from poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--repeat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    configs = [read(path/'engine_config.json') for path in [args.first, args.repeat]]
    for config in configs:
        config['compilation_config'].pop('cache_dir')
    assert configs[0] == configs[1]
    assert sha(args.first/'workload.json') == sha(args.repeat/'workload.json')
    storage = [read(path/'storage_control.json') for path in [args.first, args.repeat]]
    assert all(row['mode'] == 'alias' for arm in storage for row in arm)
    trials = [[read(path) for path in sorted(root.glob('P*_request*_repeat*.json'))]
              for root in [args.first, args.repeat]]
    assert all(len(rows) == 30 for rows in trials)
    timing = []
    for context in [1024, 4096, 8192]:
        medians = [statistics.median(row['wall_ms'] for row in rows if row['prompt'] == context)
                   for rows in trials]
        timing.append(dict(context=context, first_median_ms=medians[0], repeat_median_ms=medians[1],
                           first_over_repeat=medians[0]/medians[1]))
    capture_rows = [read(path/'parity_capture.json')['evidence'] for path in [args.first, args.repeat]]
    numerical = []
    for a in capture_rows[0]:
        b = next(row for row in capture_rows[1] if row['rank'] == a['rank'])
        assert a['positions'] == b['positions'] and a['KV_names'] == b['KV_names']
        left, right = np.load(a['path']), np.load(b['path'])
        numerical.append(dict(rank=a['rank'], comparisons={key: stats(right[key], left[key])
                                                          for key in ['logits', 'hidden', 'KV']},
                              top1_match=(left['logits'].argmax(-1) == right['logits'].argmax(-1)).tolist()))
    write(args.output, dict(first=str(args.first), repeat=str(args.repeat), timing=timing, numerical=numerical,
          interpretation='Same alias arm, independent cold compilation and processes. This tests process/compilation variability, not its unique cause.',
          evidence_class='LIVE_E2E', output_policy='128 controlled teacher tokens. Natural quality NOT_RUN.',
          uncertainty='Two request shapes, five repeats each, sequential run order. No broad request-level performance CI.',
          alias_vs_copy_conclusion='NONE', HBM_traffic='NOT_MEASURED'))
    print(timing)
    print([(row['rank'], row['comparisons']['logits']['relative_L2'], row['top1_match']) for row in numerical])


if __name__ == '__main__':
    main()
