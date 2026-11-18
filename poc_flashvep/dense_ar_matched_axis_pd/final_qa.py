"""Validate derived reports, figure provenance and the CPU regression suite."""
import json
import re
import subprocess
import sys
import time
from .common import REPO, ROOT, SOURCE, digest, preserve, write
from .publication import DOCUMENTS
from .runtime_freeze import check


def main():
    preserve(); check()
    assert json.loads((ROOT/'analysis/final_integrity_validation.json').read_text())['passed']
    report_manifest = json.loads((ROOT/'analysis/scoped_final_report_manifest.json').read_text())
    for row in report_manifest['documents']:
        assert digest(REPO/row['path']) == row['SHA256'], row['path']
    figures = []
    for filename in ('batch1_figure_manifest.json', 'scoped_final_figure_manifest.json'):
        manifest = json.loads((ROOT/'analysis'/filename).read_text())
        assert manifest['statistics_SHA256'] == digest(ROOT/'analysis/independent_start_statistics.json')
        for row in manifest['files']:
            assert digest(REPO/row['path']) == row['SHA256'], row['path']
            figures.append(row['path'])
    links = []
    for name in DOCUMENTS:
        path = REPO/'poc_flashvep/reports'/name
        text = path.read_text()
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                continue
            local = target.split('#', 1)[0]
            assert (path.parent/local).exists(), (name, target)
            links.append(dict(document=name, target=target))
    for path in SOURCE.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        'poc_flashvep/dense_ar_matched_axis_pd', '-t', '.', '-p', 'test_*.py'],
        cwd=REPO, capture_output=True, text=True)
    log = ROOT/'logs/final_CPU_regression_suite_20261118.log'
    log.write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0, result.stderr[-3000:]
    count = re.search(r'Ran (\d+) tests', result.stderr)
    assert count and int(count.group(1)) >= 39
    preserve(); check()
    write(ROOT/'analysis/final_report_and_CPU_QA.json', dict(time=time.time(), passed=True,
        CPU_tests=int(count.group(1)), CPU_log_SHA256=digest(log),
        documents=len(DOCUMENTS), checked_relative_links=links,
        verified_figure_files=sorted(set(figures)),
        final_statistics_SHA256=digest(ROOT/'analysis/independent_start_statistics.json'),
        source_SHA256={p.name:digest(p) for p in sorted(SOURCE.glob('*.py'))},
        user_state_and_frozen_runtime_preserved=True,
        scope='CPU checks and artifact provenance only. No new GPU performance or profiling is claimed.'))
    print('FINAL_REPORT_AND_CPU_QA', count.group(1), len(DOCUMENTS), len(set(figures)), flush=True)


if __name__ == '__main__':
    main()
