"""Record the evidence commit, executed checks and authorized burner state.

Read-only process inspection: this command never stops or launches GPU jobs.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from poc_native_lp_ep4.contract import atomic_json, read_json

ROOT = Path('poc_native_lp_ep4')
MASK = '0,1,4,5'


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def burner_ancestor(pid):
    for _ in range(12):
        proc = Path(f'/proc/{pid}')
        assert proc.stat().st_uid == os.getuid(), 'foreign process on an authorized GPU'
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if '/home/esjung/vllm-ep/utilize.py --gpus ' + MASK + ' ' in cmd:
            return pid
        status = (proc / 'status').read_text()
        pid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        if pid <= 1:
            break
    raise AssertionError('authorized GPU process is not a scoped existing-policy burner')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence-commit', required=True)
    a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', a.evidence_commit)
    assert command(['git', 'rev-parse', a.evidence_commit + '^{commit}']).strip() == a.evidence_commit
    output = ROOT / 'artifacts/release.json'
    if output.exists():
        raise FileExistsError(output)
    reference = read_json(ROOT / 'artifacts/runs/release_correctness_v4/source_signature.json')
    for path in ('poc_native_lp_ep4/native.py', 'poc_native_lp_ep4/contract.py'):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == reference[path]
    restored = read_json(ROOT / 'artifacts/archive_reanalysis_verification.json')
    assert restored['status'] == 'PASS'
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                           '-s', str(ROOT / 'tests'), '-v'], capture_output=True, text=True)
    assert test.returncode == 0, test.stdout + test.stderr
    count = int(re.search(r'Ran (\d+) tests', test.stderr).group(1))
    command([sys.executable, '-m', 'compileall', '-q', str(ROOT)])
    command(['git', 'diff', '--check'])
    fields = 'index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,temperature.gpu'
    rows = [dict(zip(fields.split(','), [v.strip() for v in line.split(',')]))
            for line in command(['nvidia-smi', '--query-gpu=' + fields,
                                 '--format=csv,noheader,nounits']).splitlines()]
    allowed = [r for r in rows if r['index'] in MASK.split(',')]
    uuids = {r['uuid'] for r in allowed}
    apps = []
    for line in command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                         '--format=csv,noheader']).splitlines():
        uuid, pid, name = [v.strip() for v in line.split(',', 2)]
        if uuid in uuids:
            apps.append({'uuid': uuid, 'pid': int(pid), 'process_name': name,
                         'own_burner_ancestor_pid': burner_ancestor(int(pid))})
    assert len(apps) == 4 and {r['uuid'] for r in apps} == uuids
    atomic_json(ROOT / 'artifacts/test_execution.json', {
        'status': 'PASS', 'unit_tests': count, 'unit_command': test.args,
        'unit_stdout': test.stdout, 'unit_stderr': test.stderr,
        'compileall': 'PASS', 'git_diff_check': 'PASS',
        'actual_native_validation': 'Separate numerical/stage/routing evidence; not inferred from these unit tests',
    })
    atomic_json(output, {
        'schema': 'native-lp-evidence-release-v1', 'status': 'COMPLETE',
        'wall_ns': time.time_ns(), 'evidence_commit': a.evidence_commit,
        'branch': command(['git', 'branch', '--show-current']).strip(),
        'measurement_plan': 'artifacts/campaign_blocks_v7.json',
        'historical_plan_reporting_field': 'PENDING at measurement completion; this release attests completed reporting without rewriting that historical record',
        'headline_metrics': 'artifacts/final_metrics.json', 'verdict': read_json(ROOT / 'artifacts/final_verdict.json'),
        'publication_index': 'artifacts/publication_index.json',
        'archive_reanalysis': restored,
        'actual_test_count': count,
        'native_core_unchanged_since_accepted_v4': True,
        'release_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py')) if 'artifacts' not in p.parts},
        'final_summary_sha256': hashlib.sha256((ROOT / 'reports/final_summary.md').read_bytes()).hexdigest(),
        'part1_completion_bundle_sha256': hashlib.sha256(Path(
            'poc_mllm_ep_foundation/validation_runs/20260924_completion_v1/frozen_baseline_bundle.json').read_bytes()).hexdigest(),
        'authorized_gpu_final_state': allowed,
        'authorized_gpu_processes': apps,
        'gpu_cleanup': 'PASS; only existing-policy own burners on authorized UUIDs; no PoC model remains',
        'other_gpus': 'Never allocated or stopped; shared-fabric foreign activity audited only',
        'new_communication_method_implemented': False,
        'release_note': 'Evidence commit contains complete code/measurements/reports. A following metadata-only commit records its SHA and this release; no self-referential commit hash.',
    })
    print('RELEASE_COMPLETE', count, 'tests; verified archives; four authorized burners restored')


if __name__ == '__main__':
    main()
