"""Compare bounded saved captures. No natural-accuracy or timing claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, write
from poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-output', type=Path, required=True)
    parser.add_argument('--baseline-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    debug = read(args.debug_output/'parity_capture.json')
    baseline = read(args.baseline_output/'parity_capture.json')
    base_rows = baseline['evidence']
    rows = []
    for candidate in debug['capture']:
        reference = next(row for row in base_rows if row['rank'] == candidate['rank'])
        assert reference['positions'] == candidate['positions']
        assert reference['KV_names'] == candidate['KV_names']
        a, b = np.load(reference['path']), np.load(candidate['path'])
        values = {name: stats(b[name], a[name]) for name in ['logits', 'hidden', 'KV']}
        rows.append(dict(rank=candidate['rank'], comparisons=values,
                         top1_match=(a['logits'].argmax(-1) == b['logits'].argmax(-1)).tolist()))
    assert not args.output.exists()
    write(args.output, dict(rows=rows, scope='Four teacher-forced positions. Debug capture versus a separate native process.',
                           quality='Natural generation and task accuracy NOT_RUN.',
                           timing='NOT_MEASURED. Graph retention and capture hooks are diagnostic.',
                           numerical_cause='UNRESOLVED for any differences. Independent compilation and process state can differ.',
                           debug_path=str(args.debug_output), baseline_path=str(args.baseline_output)))
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
