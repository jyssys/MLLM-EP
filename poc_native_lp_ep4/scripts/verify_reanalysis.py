"""Compare full CPU reanalysis from published archives to local raw results."""
import argparse
import json
import math
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json

IDENTIFIERS = ('run', 'wave', 'arm', 'modality', 'repeat', 'comparison',
               'control', 'phase', 'layer', 'iteration')


def canonical(value, field=None):
    if isinstance(value, dict):
        return {k: canonical(v, k) for k, v in value.items()}
    if isinstance(value, list):
        rows = [canonical(v) for v in value]
        if rows and all(isinstance(v, dict) for v in rows):
            rows.sort(key=lambda v: json.dumps({k: v[k] for k in IDENTIFIERS if k in v}, sort_keys=True))
        return rows
    if field == 'run' and isinstance(value, str):
        return Path(value).name
    return value


def compare(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        return sum(compare(a[k], b[k], path + '.' + k) for k in a)
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        return sum(compare(x, y, path + f'[{i}]') for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and not isinstance(b, bool), path
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), (path, a, b)
        return 1
    assert a == b, (path, a, b)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--rebuilt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    count = compare(canonical(read_json(a.original)), canonical(read_json(a.rebuilt)))
    atomic_json(a.output, {'status': 'PASS', 'original': str(a.original), 'rebuilt': str(a.rebuilt),
        'numeric_leaf_comparisons': count, 'absolute_float_tolerance': 1e-7,
        'relative_float_tolerance': 1e-10,
        'normalization': 'Run-directory prefixes and metadata-keyed row order only; no numeric fields removed',
        'scope': 'All final metrics rebuilt from SHA-verified restored compressed actual JSON/JSONL; no GPUs/model execution'})
    print('ARCHIVE_REANALYSIS_PASS', count, 'numeric fields')


if __name__ == '__main__':
    main()
