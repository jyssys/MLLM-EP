"""Record real work segments and preserve predecessor evidence read-only."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_parallelism_deep_discovery_spec.md')
PREDECESSORS = {
    'async': Path('/home/esjung/MLLM-EP-async-recurrent-pp'),
    'asymmetry': Path('/home/esjung/MLLM-EP-recurrent-asymmetry'),
    'depth_v3': Path('/home/esjung/MLLM-EP-decode-first-depth-dp-v3'),
    'factorial': Path('/home/esjung/MLLM-EP-depth-cdb-factorial'),
    'original': Path('/home/esjung/MLLM-EP-github'),
}

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def command(args, cwd=PROJECT):
    return subprocess.check_output(args, cwd=cwd)

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def snapshot():
    target = ROOT/'artifacts/preservation_snapshot.json'
    if target.exists():
        raise SystemExit('A preservation snapshot already exists. Do not overwrite it.')
    values = {}
    for name, path in PREDECESSORS.items():
        values[name] = {
            'path': str(path),
            'HEAD': command(['git', 'rev-parse', 'HEAD'], path).decode().strip(),
            'staged_sha256': digest(command(['git', 'diff', '--cached', '--binary'], path)),
            'unstaged_sha256': digest(command(['git', 'diff', '--binary'], path)),
        }
    values['spec_sha256'] = digest(SPEC.read_bytes())
    values['recorded_utc'] = now()
    write(target, values)
    copied = ROOT/'reports/spec_snapshot.md'
    copied.parent.mkdir(parents=True, exist_ok=True)
    if copied.exists():
        raise SystemExit('The frozen specification already exists.')
    copied.write_bytes(SPEC.read_bytes())
    write(ROOT/'research_state.json', {
        'campaign': 'Ouro Parallelism Deep Discovery',
        'status': 'ACTIVE', 'phase': 'evidence_and_primary_sources',
        'first_recorded_utc': '2026-10-19T09:52:53+00:00',
        'updated_utc': now(), 'minimum_substantive_seconds': 36000,
        'normal_completion_allowed': False,
        'physical_gpu_pool': [4, 5, 6, 7],
        'active_owned_experiments': [],
        'base_commit': command(['git', 'rev-parse', 'HEAD']).decode().strip(),
        'training': 'FORBIDDEN', 'P8': 'PROJECTION_ONLY',
        'next_action': 'Audit sources, semantics, and previous failures before candidate probes.',
    })

def verify():
    values = json.loads((ROOT/'artifacts/preservation_snapshot.json').read_text())
    for name, path in PREDECESSORS.items():
        saved = values[name]
        assert command(['git', 'rev-parse', 'HEAD'], path).decode().strip() == saved['HEAD'], name
        assert digest(command(['git', 'diff', '--cached', '--binary'], path)) == saved['staged_sha256'], name
        assert digest(command(['git', 'diff', '--binary'], path)) == saved['unstaged_sha256'], name
    assert digest(SPEC.read_bytes()) == values['spec_sha256']
    assert digest((ROOT/'reports/spec_snapshot.md').read_bytes()) == values['spec_sha256']
    print('PRESERVATION_PASS')

def segment(args):
    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end) if args.end else dt.datetime.now(dt.timezone.utc)
    assert start.tzinfo and end.tzinfo and end >= start
    row = {'start_utc': start.isoformat(), 'end_utc': end.isoformat(),
           'elapsed_seconds': (end-start).total_seconds(), 'actor': args.actor,
           'category': args.category, 'description': args.description,
           'evidence': args.evidence, 'accounting': 'Recorded actual segment, not multiplied by concurrent workers.'}
    path = ROOT/'artifacts/time_ledger.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True)+'\n')
    print(json.dumps(row))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('snapshot'); sub.add_parser('verify')
    p = sub.add_parser('segment')
    p.add_argument('--start', required=True); p.add_argument('--end')
    p.add_argument('--actor', default='coordinator')
    p.add_argument('--category', required=True); p.add_argument('--description', required=True)
    p.add_argument('--evidence', nargs='+', required=True)
    args = parser.parse_args()
    if args.mode == 'snapshot': snapshot()
    elif args.mode == 'verify': verify()
    else: segment(args)

if __name__ == '__main__': main()
