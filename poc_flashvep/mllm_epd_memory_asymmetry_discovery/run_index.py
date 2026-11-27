"""Keep a compact reproducibility index without committing external model or image data."""

import argparse
import hashlib
import json
from pathlib import Path


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(run):
    source = run / 'state/environment.json'
    if not source.is_file():
        return None
    environment = json.loads(source.read_text())
    finish = run / 'state/complete.json'
    failure = run / 'state/failure.json'
    status = 'COMPLETE' if finish.is_file() else 'FAILED' if failure.is_file() else 'NOT_RUN_OR_INCOMPLETE'
    receipts = []
    for folder, pattern in (('configs', '*_arrival.json'), ('raw', '*_prometheus.json'),
                            ('analysis', '*_metrics.json'), ('analysis', 'correctness_full28.json')):
        for path in sorted((run / folder).glob(pattern)):
            receipts.append({'path': str(path.relative_to(run)), 'sha256': hash_file(path), 'bytes': path.stat().st_size})
    original_request_sha = {}
    for path in sorted((run / 'state').glob('*_complete.json')):
        record = json.loads(path.read_text())
        if 'raw_SHA256' in record:
            original_request_sha[record['run']] = record['raw_SHA256']
    return {'name': run.name, 'status': status, 'environment_SHA256': hash_file(source),
            'observed_git_head': next((environment[key] for key in ('HEAD', 'base_head') if key in environment), None),
            'frozen_campaign_module_SHA256': environment.get('campaign_source'),
            'registered_GPU_UUIDs': environment.get('authorized_uuid_by_gpu'),
            'manifest_SHA256': environment.get('manifest_SHA256'),
            'physical_run_directory': str(run), 'reduced_and_config_receipts': receipts,
            'external_request_ledgers_sha256': original_request_sha,
            'failure': json.loads(failure.read_text()) if failure.is_file() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = [item for directory in sorted(args.root.resolve().iterdir())
               if directory.is_dir() and (item := build(directory)) is not None]
    args.out.write_text(json.dumps({'runs': records,
        'scope': 'External directories contain per-run configs, exact source overlays, process logs and request evidence. '
                 'This index excludes model tensors, original images, runtime credentials and session ownership nonces.'},
        indent=2) + '\n')


if __name__ == '__main__':
    main()
