"""Reproduce initial-environment loss with the installed title extension. CPU only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from poc_ouro_parallelism_deep_discovery.contract import NATIVE_PYTHON, ROOT, sha, write


CHILD_CODE = r'''
import json, os, sys
from pathlib import Path
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
import setproctitle
from setproctitle import _setproctitle
assert setproctitle.__version__ == '1.3.7'
assert setproctitle.setproctitle is _setproctitle.setproctitle
expected = dict(LOOPED_KV_OUTPUT='CPU_SPT_NOENV_PUBLIC_FIXTURE',
                CUDA_VISIBLE_DEVICES='', VLLM_WORKER_MULTIPROC_METHOD='spawn')
def guard_snapshot():
    entries = Path('/proc/self/environ').read_bytes().split(b'\0')
    return dict(logical={key: dict(present=key in os.environ,
        matches=os.environ.get(key)==value) for key,value in expected.items()},
        initial_proc={key: dict(present=any(entry.startswith((key+'=').encode()) for entry in entries),
        matches=(key+'='+value).encode() in entries) for key,value in expected.items()})
before = guard_snapshot()
assert all(item['present'] and item['matches'] for items in before.values() for item in items.values())
rows = []
for title in ('VLLM::Worker', 'VLLM::Worker_TP0'):
    setproctitle.setproctitle(title)
    title_applied = setproctitle.getproctitle() == title
    cmdline_title_matches = Path('/proc/self/cmdline').read_bytes().split(b'\0')[0] == title.encode()
    assert title_applied and cmdline_title_matches
    rows.append(dict(requested_title=title, title_applied=title_applied,
        cmdline_title_matches=cmdline_title_matches, guards=guard_snapshot()))
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import inspect_process_leases
try:
    inspect_process_leases([os.getpid()], os.getpgid(0), os.getsid(0),
                          expected['LOOPED_KV_OUTPUT'], ancestor_pid=os.getpid())
except RuntimeError as error:
    assert str(error) == f'A declared process {os.getpid()} lost its guard output identity.'
    strict_lease_pass = False
else:
    strict_lease_pass = True
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
print(json.dumps(dict(pid=os.getpid(), package_version=setproctitle.__version__,
    package_path=setproctitle.__file__, extension_path=_setproctitle.__file__,
    before_first_title=before, title_calls=rows, strict_lease_pass=strict_lease_pass,
    torch_imported=False, vllm_imported=False, diagnostic_only=True)))
'''


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Require a new artifact path inside this campaign.')
    if sys.executable != NATIVE_PYTHON:
        raise ValueError('Use the pinned native Python interpreter.')
    native = Path(NATIVE_PYTHON).parents[1] / 'lib/python3.14/site-packages'
    protected = [ROOT / 'scripts/process_title_guard_cpu.py',
                 ROOT / 'artifacts/process_title_guard_cpu_repro_v1.json',
                 ROOT / 'scripts/process_lease_spawn_chain_cpu_v1.py',
                 ROOT / 'artifacts/process_lease_spawn_chain_cpu_v1.json',
                 ROOT / 'reports/process_lease_diagnostic_independent_review_v1.md']
    protected_before = {str(path): sha(path) for path in protected}
    source_paths = [Path(__file__).resolve(), ROOT / 'tp2_dp2_cohort.py',
        native / 'setproctitle/__init__.py',
        native / 'setproctitle/_setproctitle.cpython-314-x86_64-linux-gnu.so',
        native / 'setproctitle-1.3.7.dist-info/METADATA']
    sources = {str(path): sha(path) for path in source_paths}
    start, cases = utc_now(), []
    for enabled in (False, True):
        environment = dict(os.environ, LOOPED_KV_OUTPUT='CPU_SPT_NOENV_PUBLIC_FIXTURE',
                           CUDA_VISIBLE_DEVICES='', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        environment.pop('SPT_NOENV', None)
        if enabled:
            environment['SPT_NOENV'] = '1'
        case_start = utc_now()
        process = subprocess.run([sys.executable, '-c', CHILD_CODE], env=environment,
                                 text=True, capture_output=True, timeout=10)
        case_end = utc_now()
        if process.returncode != 0:
            raise RuntimeError(f'A CPU child failed with exit code {process.returncode}.')
        result = json.loads(process.stdout)
        assert result['package_path'] == str(native / 'setproctitle/__init__.py')
        for row in result['title_calls']:
            assert all(item['present'] and item['matches'] for item in row['guards']['logical'].values())
            assert all(item['present'] == enabled and item['matches'] == enabled
                       for item in row['guards']['initial_proc'].values())
        assert result['strict_lease_pass'] == enabled
        cases.append(dict(SPT_NOENV_enabled_before_exec=enabled, start_utc=case_start,
                          end_utc=case_end, child_returncode=process.returncode, result=result))
    protected_after = {str(path): sha(path) for path in protected}
    assert protected_after == protected_before
    record = dict(start_utc=start, end_utc=utc_now(),
        command=[sys.executable, '-m',
            'poc_ouro_parallelism_deep_discovery.scripts.setproctitle_noenv_red_green_cpu_v1',
            '--output', str(output)], cases=cases, red_green_passed=True,
        source_sha256_before_run=sources, protected_sha256_before=protected_before,
        protected_sha256_after=protected_after, protected_files_unchanged=True,
        guards_exported_as_booleans_only=True, torch_imported_in_children=False,
        scope='Installed Linux title extension and strict self lease. No native engine, model, or GPU execution.',
        import_timing='The child imports setproctitle before any runtime environment edit. No configure_native_imports call precedes title setup.',
        limitations=['Short titles fit this subprocess command line. Truncation limits are documented, not measured here.',
            'The first title call already loses initial proc keys without SPT_NOENV.',
            'This CPU test does not identify every operation in a full native startup.'])
    write(output, record)
    print('SPT_NOENV_CPU: default strict lease REJECTED, SPT_NOENV strict lease PASSED, both requested titles applied in both children')


if __name__ == '__main__':
    main()
