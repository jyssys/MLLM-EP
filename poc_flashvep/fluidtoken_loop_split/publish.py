"""Publish this campaign without changing the user's index or checkout."""
import argparse
import tempfile
from .common import *


def main(action, push=False):
    configure()
    from ..looped_pd_autonomous_headroom import publish_campaign as inherited
    inherited.ROOT = ROOT
    if action == 'archive':
        inherited.archive()
        return
    final = REPO / 'poc_flashvep/reports/looped_pd_fluidtoken_vs_dynamic_loop_split_final_summary.md'
    spec = final.with_name('looped_pd_fluidtoken_vs_dynamic_loop_split_poc_spec.md')
    assert final.exists() and spec.exists()
    prior.preserve()
    git = inherited.git
    active = git('branch', '--show-current')
    assert active != BRANCH
    head = git('rev-parse', 'refs/heads/' + BRANCH)
    manifest = json.loads((ROOT / 'archives/manifest.json').read_text())
    replaced = {ROOT / row['original'] for row in manifest['originals']}
    paths = list(SOURCE.rglob('*.py')) + list(SOURCE.rglob('*.cc')) + [spec, final]
    paths += [p for p in ROOT.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and not p.name.endswith(('.tmp', '.lock'))
              and '.tmp.' not in p.name and p.name != 'burner.log']
    files = sorted({str(p.relative_to(REPO)) for p in paths if p not in replaced})
    assert all((REPO / name).stat().st_size < 100_000_000 for name in files)
    inherited.BRANCH = BRANCH
    with tempfile.TemporaryDirectory(prefix='fluidtoken_publication_') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git('read-tree', head, env=env)
        batches, batch, size = [], [], 0
        for name in files:
            length = (REPO / name).stat().st_size
            if batch and size + length > 400 * 1024**2:
                batches.append(batch)
                batch, size = [], 0
            batch.append(name)
            size += length
        if batch:
            batches.append(batch)
        for number, batch in enumerate(batches, 1):
            for index in range(0, len(batch), 100):
                git('add', '-f', '--', *batch[index:index + 100], env=env)
            changed = git('diff', '--cached', '--name-only', head, env=env).splitlines()
            assert set(changed) <= set(files)
            if not changed:
                continue
            prose = [p for p in changed if p.endswith(('.py', '.md')) and '/raw/' not in p]
            if prose:
                check = subprocess.run(['git', 'diff', '--cached', '--check', head, '--', *prose],
                    cwd=REPO, env=env, capture_output=True, text=True)
                if check.returncode:
                    # Preserve the frozen diagnostic instead of changing its historical bytes.
                    frozen = ROOT / 'huginn_recovery_diagnostic_notes.md'
                    allowed = str(frozen.relative_to(REPO)) + ':220: new blank line at EOF.'
                    assert check.stdout.strip() == allowed and not check.stderr, check.stdout + check.stderr
                    write(ROOT / 'state/publication_frozen_format_exception.json', dict(
                        file=str(frozen.relative_to(REPO)), SHA256=digest(frozen),
                        diagnostic=allowed, action='Preserve the frozen file byte for byte. All other whitespace checks remain mandatory.'))
            tree = git('write-tree', env=env)
            previous = head
            head = git('commit-tree', tree, '-p', previous, '-m',
                       f'Record Huginn recovery and split comparison evidence ({number}/{len(batches)})', env=env)
            git('update-ref', 'refs/heads/' + BRANCH, head, previous)
            prior.preserve()
            if push:
                inherited.push(head)
            print(json.dumps(dict(commit=head, batch=number, batches=len(batches))), flush=True)
    prior.preserve()
    assert git('branch', '--show-current') == active
    remote = inherited.push(head) if push else None
    write(ROOT / 'state/publication_receipt.json', dict(branch=BRANCH, commit=head,
          remote_SHA=remote, verified=remote == head if push else False,
          original_index_preserved=True, external_runtime_preserved=True, files=len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('archive', 'publish'))
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    main(args.action, args.push)
