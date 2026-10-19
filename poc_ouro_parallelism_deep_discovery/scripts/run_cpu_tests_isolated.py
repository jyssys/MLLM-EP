"""Run CPU regression modules separately. Some tests assert import isolation."""
import argparse
import datetime as dt
import re
import subprocess
import sys

from poc_ouro_parallelism_deep_discovery.contract import PROJECT, ROOT, sha, write


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = ROOT / args.output
    if target.exists():
        raise FileExistsError(target)
    started, rows = stamp(), []
    source_paths = sorted(list(ROOT.glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py')))
    source_hashes = {str(path.relative_to(ROOT)):sha(path) for path in source_paths}
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        module = '.'.join(path.relative_to(PROJECT).with_suffix('').parts)
        command = [sys.executable, '-m', 'unittest', module]
        before = stamp()
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        match = re.search(r'Ran (\d+) tests?', output)
        rows.append(dict(module=module, command=command, start_utc=before, end_utc=stamp(),
                         returncode=result.returncode, tests=int(match[1]) if match else None,
                         output=output if len(output) < 20000 else output[:12000] + '\nTRUNCATED\n' + output[-4000:]))
        print(module, result.returncode, rows[-1]['tests'], flush=True)
    failed = [r for r in rows if r['returncode']]
    source_drift = [relative for relative,digest in source_hashes.items() if sha(ROOT/relative) != digest]
    write(target, dict(start_utc=started, end_utc=stamp(), interpreter=sys.executable,
        modules=len(rows), tests=sum(r['tests'] or 0 for r in rows), failed_modules=len(failed),
        status='PASS' if not failed and not source_drift else 'FAIL', rows=rows,
        source_sha256=source_hashes, source_drift=source_drift,
        source_scope='Campaign package, scripts and test modules. External dependencies and binaries are not fully receipted.',
        scope='CPU regressions in fresh interpreter processes. No model or GPU test.',
        previous_attempts=[
            dict(interpreter='Python3.11 reference environment', mode='Single-process discovery',
                 result='98 tests with one import error. Storage-clone test requires the native Python3.14 Torch package.'),
            dict(interpreter='Dedicated Python3.14 environment', mode='Single-process discovery',
                 result='100 tests with two failures. Earlier test imports Torch, violating later tests that require a Torch-free process.')],
        correction='Use the intended interpreter and one fresh process per module. Do not relax import-isolation assertions.'))
    raise SystemExit(bool(failed or source_drift))


if __name__ == '__main__':
    main()
