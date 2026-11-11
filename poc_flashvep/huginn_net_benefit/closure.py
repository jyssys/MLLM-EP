"""Check campaign outputs without changing the user's index or external Runtime."""
import hashlib
import subprocess
import sys
from .common import *


def inherited_source_audit():
    packages = ('completion_aware_loop_pd', 'idle_intervention_dynamic_loop',
        'native_loop_2d_oracle', 'native_split_deep_validation', 'looped_pd_headroom_recovery',
        'looped_pd_atomic_prefill', 'looped_pd_abc', 'adaptive_recurrence_cut',
        'dynamic_loop_split_serving', 'fluidtoken_loop_split', 'ouro_budget_joint_cg')
    lines = subprocess.check_output(['git','ls-tree','-r',BASE,'--',
        *[f'poc_flashvep/{package}' for package in packages]], cwd=REPO).decode().splitlines()
    verified = 0
    differences = []
    for line in lines:
        meta, path = line.split('\t', 1)
        if not path.endswith(('.py','.cc')):
            continue
        current = REPO/path
        if not current.exists():
            differences.append(dict(path=path, status='MISSING'))
            continue
        data = current.read_bytes()
        local_blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if local_blob != meta.split()[-1]:
            differences.append(dict(path=path, status='CONTENT_DIFFERS'))
        else:
            verified += 1
    write(ROOT/'state/inherited_source_audit.json', dict(base=BASE, packages=packages,
        matching_source_blobs=verified, differences=differences,
        scope='Read-only comparison of reused worktree source to the publication parent. Per-run frozen source archives remain authoritative.'))
    return verified, differences


def main():
    configure()
    from ..completion_aware_loop_pd.publish import preserved, git
    preserved()
    inherited_source_audit()
    from .exit_audit import main as exit_audit
    from .ledger import main as ledger
    exit_audit()
    ledger()
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s',
            'poc_flashvep/huginn_net_benefit', '-t', '.', '-v'],
        [sys.executable, '-m', 'unittest', 'discover', '-s',
            'poc_flashvep/completion_aware_loop_pd', '-t', '.', '-v'],
        [sys.executable, '-m', 'unittest',
            'poc_flashvep.idle_intervention_dynamic_loop.test_protocol', '-v'],
        [sys.executable, '-m', 'compileall', '-q', str(SOURCE)]]
    checks = []
    for command in commands:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        checks.append(dict(command=command, returncode=result.returncode,
            output=result.stdout+result.stderr))
    write(ROOT/'state/final_source_verification.json', dict(time=time.time(),
        checks=checks, all_passed=all(row['returncode'] == 0 for row in checks),
        active_branch=git('branch', '--show-current'), preservation_passed=True,
        source_hashes={str(path.relative_to(REPO)):digest(path) for path in SOURCE.rglob('*')
            if path.is_file() and path.suffix in ('.py', '.cc')},
        scope='Final source hashes are not substituted for the frozen source snapshot of any earlier run. The active checkout and external Runtime retain their original state.'))
    assert all(row['returncode'] == 0 for row in checks)
    print('SOURCE_AND_PRESERVATION_CHECKS_PASS')


if __name__ == '__main__':
    main()
