"""Select completed physical evidence for publication, without model tensors.

This never deletes local failures or changes the original dirty worktree. Logs
are preserved locally; a bounded failure excerpt and SHA keep exclusions auditable.
Run only after all GPU producer-control executions have stopped.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json

ROOT = Path('poc_native_lp_ep4')


def main():
    plan = read_json(ROOT / 'artifacts/campaign_blocks_v7.json')
    producer = read_json(ROOT / 'artifacts/producer_control_v7.json')
    assert plan['status'] == 'MEASUREMENTS_COMPLETE' and producer['status'] == 'COMPLETE'
    runs = list(dict.fromkeys(plan['correctness_runs'] + plan['discovery_runs'] +
                plan['final_runs'] + plan['component_runs'] + plan['route_runs'] + producer['runs']))
    assert all(read_json(Path(run) / 'status.json')['ok'] for run in runs)
    target = ROOT / 'artifacts/compact_raw'
    subprocess.run([sys.executable, '-m', 'poc_native_lp_ep4.scripts.export_compact',
                    '--runs', *runs, '--output', str(target)], check=True)
    subprocess.run([sys.executable, '-m', 'poc_native_lp_ep4.scripts.verify_compact', str(target)], check=True)
    index = read_json(target / 'integrity.json')
    files = [r for r in index['files'] if 'path' in r]
    assert max(r['compressed_bytes'] for r in files) < 90_000_000
    log = ROOT / 'artifacts/component_block3_v6.log'
    lines = log.read_text().splitlines()
    failed = ROOT / 'artifacts/runs/component_block3_v6'
    atomic_json(ROOT / 'artifacts/exclusion_audit.json', {
        'failed_run': str(failed), 'status': read_json(failed / 'status.json'),
        'full_local_log': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
        'bounded_root_cause_excerpt': [line for line in lines if 'EADDRINUSE' in line],
        'before_measured_wave': not any((failed / f'worker.dp{r}.json').exists() for r in (0, 1)),
        'excluded_from_headlines': True,
        'other_excluded_runs': [
            {'version': 'v1/v2', 'reason': 'client-poll clock/administrative interruption; not engine-completion acceptance'},
            {'version': 'v3 release', 'reason': 'startup request-id typo; no accepted timing'},
            {'version': 'v4 discovery', 'reason': 'wave admission timeout; no completed discovery comparison'},
            {'version': 'v5 timing', 'reason': 'nonuniform resident initialization; replaced before valid selection'},
            {'version': 'v6 legacy component', 'reason': 'Full/visual 10.721% overhead; diagnostic retained; fresh pooled trace collected'},
            {'version': 'v7 Full/text component', 'reason': '10.302% overhead; retained UNVALIDATED, not used for EP-common conclusion'},
        ],
        'deletion_or_overwrite': 'NONE; all local old raw evidence retained',
    })
    atomic_json(ROOT / 'artifacts/publication_index.json', {
        'schema': 'native-lp-publication-index-v1', 'runs': runs,
        'compressed_file_count': len(files), 'compressed_bytes': sum(r['compressed_bytes'] for r in files),
        'raw_json_bytes': sum(r['raw_bytes'] for r in files),
        'largest_compressed_file_bytes': max(r['compressed_bytes'] for r in files),
        'tensors': 'LOCAL_ONLY; no checkpoint/image/hidden/logit tensor upload',
        'immutable_clean_source': 'v6 discovery + five two-repeat clean final blocks',
        'trace_continuation': 'v7 pooled light + rich; same native.py/contract.py math',
        'producer_control': 'separate secondary v7 cached diagnostic, three repeats',
        'original_runtime_checkpoint': '6e318988cb0cca168850fb31211ff2f8ae14c4fc',
        'current_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py')) if 'artifacts' not in p.parts},
        'inherited_driver_sha_limit': 'campaign_blocks_v7.driver_sha256 came from v6. Current completion driver hash is separately included here and in code; not relabeled as the inherited hash.',
        'archive_verification': 'PASS compressed/raw SHA256 and actual decompression',
        'independent_reanalysis': 'See archive_reanalysis_verification.json; separate fresh restore and full CPU reanalysis',
    })
    print('COMPACT_PUBLICATION', len(runs), 'runs', len(files), 'files',
          sum(r['compressed_bytes'] for r in files), 'compressed bytes')


if __name__ == '__main__':
    main()
