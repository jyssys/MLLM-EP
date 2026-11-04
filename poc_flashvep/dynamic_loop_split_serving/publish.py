"""Publish this namespace without changing the real index or checkout."""
import argparse
import tempfile
from .common import *


def check_prose_whitespace(git, head, prose, env):
    frozen = str((ROOT / 'campaign_resume_checkpoint.md').relative_to(REPO))
    ordinary = [name for name in prose if name != frozen]
    if ordinary:
        git('diff', '--cached', '--check', head, '--', *ordinary, env=env)
    if frozen in prose:
        # Keep this earlier checkpoint byte-identical. Waive only blank EOF.
        # Every other whitespace rule and every other file remains checked.
        git('-c', 'core.whitespace=-blank-at-eof', 'diff', '--cached', '--check',
            head, '--', frozen, env=env)


def main(action, push=False):
    configure()
    from ..looped_pd_autonomous_headroom import publish_campaign as inherited
    inherited.ROOT = ROOT
    if action == 'archive':
        inherited.archive()
        return
    final = REPO / 'poc_flashvep/reports/ouro_huginn_fluidtoken_dynamic_loop_split_serving_final_summary.md'
    spec = final.with_name('ouro_huginn_fluidtoken_dynamic_loop_split_serving_poc_spec.md')
    assert final.exists() and spec.exists()
    prior.preserve()
    git = inherited.git
    active = git('branch', '--show-current')
    assert active != BRANCH
    head = git('rev-parse', 'refs/heads/' + BRANCH)
    manifest = json.loads((ROOT / 'archives/manifest.json').read_text())
    replaced = {ROOT / row['original'] for row in manifest['originals']}
    paths = list(SOURCE.rglob('*.py')) + list(SOURCE.rglob('*.cc')) + [spec, final]
    paths += [path for path in ROOT.rglob('*') if path.is_file()
              and '__pycache__' not in path.parts and not path.name.endswith(('.tmp', '.lock'))
              and '.tmp.' not in path.name and path.name != 'burner.log']
    files = sorted({str(path.relative_to(REPO)) for path in paths if path not in replaced})
    assert all((REPO / name).stat().st_size < 100_000_000 for name in files)
    inherited.BRANCH = BRANCH
    with tempfile.TemporaryDirectory(prefix='dynamic_loop_split_publication_') as temporary:
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
            prose = [name for name in changed if name.endswith(('.py', '.md')) and '/raw/' not in name]
            if prose:
                check_prose_whitespace(git, head, prose, env)
            tree = git('write-tree', env=env)
            previous = head
            head = git('commit-tree', tree, '-p', previous, '-m',
                f'Record exact Token and Dynamic Loop-Split serving evidence ({number}/{len(batches)})', env=env)
            git('update-ref', 'refs/heads/' + BRANCH, head, previous)
            prior.preserve()
            if push:
                inherited.push(head)
            print(json.dumps(dict(commit=head, batch=number, batches=len(batches))), flush=True)
    prior.preserve()
    assert git('branch', '--show-current') == active
    remote = inherited.push(head) if push else None
    write(ROOT / 'state/publication_receipt.json', dict(time=time.time(), branch=BRANCH,
        commit=head, remote_SHA=remote, verified=remote == head if push else False,
        original_index_preserved=True, external_runtime_preserved=True, files=len(files)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('archive', 'publish'))
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    main(args.action, args.push)
