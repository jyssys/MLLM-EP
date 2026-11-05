"""Publish this namespace through an alternate index and ordinary pushes."""
import argparse
import tempfile
from .common import *
from .report import FINAL


def main(action, push=False):
    initialize()
    from ..looped_pd_autonomous_headroom import publish_campaign as inherited
    inherited.ROOT, inherited.BRANCH = ROOT, BRANCH
    if action == 'archive':
        inherited.archive()
        return
    prior.parent.prior.preserve()
    git = inherited.git
    active = git('branch', '--show-current')
    assert active != BRANCH
    head = git('rev-parse', 'refs/heads/' + BRANCH)
    spec = FINAL.with_name('ouro_huginn_loop_specific_native_split_deep_validation_poc_spec.md')
    assert FINAL.exists() and spec.exists()
    manifest = json.loads((ROOT / 'archives/manifest.json').read_text())
    replaced = {ROOT / row['original'] for row in manifest['originals']}
    files = list(SOURCE.rglob('*.py')) + [spec, FINAL]
    files += [path for path in ROOT.rglob('*') if path.is_file()
              and '__pycache__' not in path.parts and not path.name.endswith(('.lock', '.tmp'))
              and '.tmp.' not in path.name and path.name not in ('burner.log', 'final_remote_verification.json')]
    names = sorted({str(path.relative_to(REPO)) for path in files if path not in replaced})
    assert all((REPO / name).stat().st_size < 100_000_000 for name in names)
    with tempfile.TemporaryDirectory(prefix='native_split_deep_publication_') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git('read-tree', head, env=env)
        batches, batch, size = [], [], 0
        for name in names:
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
            assert set(changed) <= set(names)
            if not changed:
                continue
            prose = [name for name in changed if name.endswith(('.py', '.md')) and '/raw/' not in name]
            if prose:
                git('diff', '--cached', '--check', head, '--', *prose, env=env)
            tree = git('write-tree', env=env)
            previous = head
            head = git('commit-tree', tree, '-p', previous, '-m',
                f'Record native split deep validation evidence ({number}/{len(batches)})', env=env)
            git('update-ref', 'refs/heads/' + BRANCH, head, previous)
            prior.parent.prior.preserve()
            if push:
                inherited.push(head)
            print(json.dumps(dict(commit=head, batch=number, batches=len(batches), pushed=push)), flush=True)
        if push:
            remote = inherited.push(head)
            receipt = ROOT / 'state/publication_receipt.json'
            write(receipt, dict(time=time.time(), branch=BRANCH, evidence_commit=head,
                evidence_remote_SHA=remote, verified=remote == head,
                original_index_preserved=True, external_runtime_preserved=True, files=len(names),
                scope='This committed receipt verifies the evidence parent. The final metadata commit has a separate local remote-SHA check.'))
            metadata = [str(receipt.relative_to(REPO)), str((ROOT / 'state/publication_attempts.jsonl').relative_to(REPO))]
            git('add', '-f', '--', *metadata, env=env)
            tree = git('write-tree', env=env)
            previous = head
            head = git('commit-tree', tree, '-p', previous, '-m', 'Record verified native split publication receipt', env=env)
            git('update-ref', 'refs/heads/' + BRANCH, head, previous)
            remote = inherited.push(head)
            write(ROOT / 'state/final_remote_verification.json', dict(time=time.time(), branch=BRANCH,
                commit=head, remote_SHA=remote, verified=remote == head,
                scope='Local verification of the final metadata commit. The committed receipt verifies its evidence parent.'))
        prior.parent.prior.preserve()
        assert git('branch', '--show-current') == active
        print(json.dumps(dict(branch=BRANCH, commit=head, pushed=push)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('archive', 'publish'))
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    main(args.action, args.push)
