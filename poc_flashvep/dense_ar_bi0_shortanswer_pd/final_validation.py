"""Check the closed campaign before publication. Do not dispatch GPU work."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .common import REPO, ROOT, SOURCE, digest, preserve, resources, write
from .publication import DOCUMENTS


def command(arguments):
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True)
    row = dict(arguments=arguments, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr)
    assert result.returncode == 0, row
    return row


def main():
    preserve()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    tests = command([sys.executable, '-m', 'unittest', 'discover', '-s',
                     'poc_flashvep/dense_ar_bi0_shortanswer_pd', '-t', '.', '-p', 'test_*.py'])
    reports = [REPO/'poc_flashvep/reports'/name for name in DOCUMENTS]
    linter = Path('/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py')
    clarity = command([sys.executable, str(linter), *map(str, reports), '--json'])
    sources = sorted(SOURCE.glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    links = []
    for path in reports:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            target = target.split('#', 1)[0].strip('<>')
            assert (path.parent/target).exists(), (str(path), target)
            links.append(dict(report=path.name, target=target))
    metrics = json.loads((ROOT/'analysis/BI0_primary_metric_reproduction_audit_v1.json').read_text())
    assert metrics['all_passed']
    archive = json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    assert archive['verified_lossless']
    for row in archive['parts']:
        assert digest(REPO/row['path']) == row['SHA256']
    prior = json.loads((ROOT/'state/prior_report_hashes.json').read_text())
    assert all(digest(REPO/'poc_flashvep/reports'/name) == sha for name, sha in prior.items())
    owner = resources()
    burners, active = owner.burners(), owner.processes()
    assert len(active) == 8 and {r['gpu'] for r in active} == set(range(8))
    assert all(os.getsid(r['pid']) == burners.get(r['gpu']) and
               Path('/proc', str(r['pid'])).stat().st_uid == os.getuid() for r in active)
    preserve()
    write(ROOT/'state/final_validation_receipt.json', dict(
        time=time.time(), passed=True, CPU_tests=tests, clarity_scan=clarity,
        parsed_source_files=len(sources), checked_local_report_links=links,
        recomputed_metric_cells=len(metrics['rows']), all_metric_cells_passed=True,
        archive_members=len(archive['original_files']), archive_parts_verified=len(archive['parts']),
        prior_reports_unchanged=True, user_checkout_index_external_runtime_preserved=True,
        all8_approved_burners_running=True,
        scope='CPU and resource checks only. These checks do not add scientific evidence or server starts.'))
    write(ROOT/'state/progress.json', dict(time=time.time(), phase='CLOSED_PUBLICATION',
        status='COMPLETED_RESEARCH_HOLD', scientific_verdict='HOLD_LAYER_AWARE_DECODE_SAFE_DYNAMIC',
        scope='GPU campaign is closed. Static results are measured. Active-decode Dynamic remains unqualified.'))
    print('FINAL_VALIDATION_PASSED', len(metrics['rows']), len(sources), flush=True)


if __name__ == '__main__':
    main()
