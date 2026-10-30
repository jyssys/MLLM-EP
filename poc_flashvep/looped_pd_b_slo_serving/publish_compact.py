"""Publish a compact tree while retaining the rejected commit and local raw JSON."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .common import ROOT, REPO, BASE, HARD_STOP, write


OWN = ['poc_flashvep/looped_pd_b_slo_serving', 'poc_flashvep/reports/looped_pd_b_slo_serving']


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env, input=data).decode().strip()


def outside_index():
    command = ['git', 'ls-files', '--stage', '-z', '--', '.', *[':!' + path for path in OWN],
               ':!poc_flashvep/reports/looped_pd_b_slo_serving_poc_spec.md']
    return hashlib.sha256(subprocess.check_output(command, cwd=REPO)).hexdigest()


def main():
    branch = git('branch', '--show-current')
    if branch != 'flashvep/looped-pd-b-slo-serving-poc':
        raise RuntimeError('Unexpected branch. No publication action.')
    old = git('rev-parse', 'HEAD')
    if git('rev-parse', old + '^') != BASE:
        raise RuntimeError('Review unpublished history before changing its publication parent')
    for path in git('diff', '--name-only', BASE, old).splitlines():
        if not any(path == prefix or path.startswith(prefix + '/') for prefix in OWN):
            raise RuntimeError('The unpublished commit changes a path outside this PoC')
    before = outside_index()
    expected = json.loads((ROOT / 'research_state.json').read_text())['outside_index_sha256']
    if before != expected:
        raise RuntimeError('Outside staged work changed. Review publication without modifying it.')
    if git('ls-remote', 'origin', 'refs/heads/' + branch):
        raise RuntimeError('Remote branch exists. Refuse any history replacement or forced push.')
    storage = json.loads((ROOT / 'configs/dataset_storage.json').read_text())
    if not storage['exact_decompression_verified'] or not (ROOT / storage['original']).exists():
        raise RuntimeError('The original dataset or exact compressed representation is missing')
    backup = 'refs/heads/flashvep/looped-pd-b-slo-serving-local-unpublished-' + old[:12]
    existing = subprocess.run(['git', 'rev-parse', '--verify', backup], cwd=REPO, capture_output=True)
    if existing.returncode == 0 and existing.stdout.decode().strip() != old:
        raise RuntimeError('Existing archive reference differs')
    if existing.returncode:
        git('update-ref', backup, old)
    with tempfile.TemporaryDirectory(prefix='looped-b-publication-') as temporary:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git('read-tree', old, env=environment)
        git('add', '--', *OWN, env=environment)
        original = str((ROOT / storage['original']).relative_to(REPO))
        git('update-index', '--force-remove', '--', original, env=environment)
        tree = git('write-tree', env=environment)
        for line in git('ls-tree', '-r', '-l', tree, '--', *OWN).splitlines():
            fields = line.split(None, 4)
            if len(fields) == 5 and fields[3].isdigit() and int(fields[3]) >= 100_000_000:
                raise RuntimeError('An intended publication blob still exceeds the safety limit: ' + fields[4])
        sha = git('commit-tree', tree, '-p', BASE,
                  '-m', 'Finalize B SLO serving diagnosis, prior-result comparison, and exact compact reproduction')
    git('update-ref', 'refs/heads/flashvep/looped-pd-b-slo-serving-publication-draft', sha)
    write(ROOT / 'state/publication_candidate.json', dict(commit=sha, original_commit=old,
          archive_ref=backup, original_raw_preserved=True, outside_index_sha256=before, time=time.time()))
    result = subprocess.run(['git', 'push', 'origin', sha + ':refs/heads/' + branch], cwd=REPO,
                            capture_output=True, timeout=600)
    (ROOT / 'logs/publication_compact_push.log').write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Compact push failed. Both commits and local raw inputs remain preserved.')
    remote = git('ls-remote', 'origin', 'refs/heads/' + branch).split()[0]
    if remote != sha:
        raise RuntimeError('Remote SHA verification failed')
    git('update-ref', 'refs/heads/' + branch, sha, old)
    # This changes only the intended staged paths. It never restores working files.
    git('restore', '--staged', '--source=' + sha, '--', *OWN)
    if outside_index() != before:
        raise RuntimeError('Outside staged index differs after publication')
    git('update-ref', 'refs/remotes/origin/' + branch, sha)
    git('branch', '--set-upstream-to=origin/' + branch, branch)
    required = ['09_final_decision.md', '10_why_previous_gain_did_not_transfer.md', '11_final_summary_ko.md',
                'e2e_component_percent.csv', 'historical_vs_current_slo_rescoring.csv', 'data/W1_long.json.gz',
                'configs/dataset_storage.json', 'reproduce.md']
    for name in required:
        git('cat-file', '-e', sha + ':' + str((ROOT / name).relative_to(REPO)))
    write(ROOT / 'state/publication_receipt.json', dict(commit=sha, remote=remote, verified=True,
          branch=branch, previous_unpublished_commit=old, archive_ref=backup,
          outside_index_sha256=before, original_raw_preserved=(ROOT / storage['original']).exists(),
          original_hard_stop=HARD_STOP, publication_after_original_hard_stop=time.time() > HARD_STOP,
          reason='User resumed report/publication closeout after a capacity interruption. No new GPU research.', time=time.time()))
    print(json.dumps(dict(branch=branch, commit=sha, remote_verified=remote == sha,
                          unpublished_commit_preserved=backup, original_raw_preserved=True), indent=2))


if __name__ == '__main__':
    main()
