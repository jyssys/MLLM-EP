"""Publish only this campaign through a temporary index and ordinary pushes."""
import argparse
import gzip
import tempfile
from .common import *

FINAL = REPO/'poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom_final_summary.md'


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env, text=True).strip()


def archive():
    from ..looped_pd_headroom_recovery import archive_evidence as inherited
    inherited.ROOT = ROOT
    result = inherited.build()
    # Interrupted profiler fragments are evidence too. Preserve them losslessly.
    records = {r['original']: r for r in result['originals']}
    for path in sorted((ROOT/'raw').glob('**/*.tmp')):
        relative = str(path.relative_to(ROOT))
        sha = digest(path)
        if relative in records and records[relative]['SHA256'] == sha:
            continue
        parts = []
        with path.open('rb') as stream:
            for index in range(100000):
                block = stream.read(64*1024**2)
                if not block:
                    break
                name = f'{sha}.part{index:04d}.gz'
                target = ROOT/'archives'/name
                if not target.exists():
                    with target.open('wb') as raw:
                        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as handle:
                            handle.write(block)
                with gzip.open(target, 'rb') as check:
                    assert check.read() == block
                parts.append(dict(file=name, original_chunk_bytes=len(block), compressed_bytes=target.stat().st_size,
                    compressed_SHA256=digest(target), original_chunk_SHA256=hashlib.sha256(block).hexdigest()))
        records[relative] = dict(original=relative, bytes=path.stat().st_size, SHA256=sha, parts=parts,
            incomplete_profiler_fragment=True,
            reconstruction='Decompress numbered parts and concatenate them in order. Verify the original SHA256. The original trace can be incomplete.')
    result['originals'] = sorted(records.values(), key=lambda r: r['original'])
    write(ROOT/'archives/manifest.json', result)
    return result


def push(head):
    result = subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'push', 'origin',
        f'{head}:refs/heads/{BRANCH}'], cwd=REPO, capture_output=True, text=True)
    remote = git('ls-remote', 'origin', 'refs/heads/'+BRANCH).split()
    actual = remote[0] if remote else None
    append(ROOT/'state/publication_attempts.jsonl', dict(time=time.time(), commit=head, returncode=result.returncode,
        remote_SHA=actual, verified=actual == head, stdout=result.stdout[-2000:], stderr=result.stderr[-2000:]))
    assert actual == head, 'Remote SHA verification failed. No force push is permitted.'
    return actual


def publish(do_push=False):
    preserve()
    assert FINAL.exists()
    active = git('branch', '--show-current')
    assert active != BRANCH
    parent = git('rev-parse', 'refs/heads/'+BRANCH)
    manifest = json.loads((ROOT/'archives/manifest.json').read_text())
    replaced = {ROOT/r['original'] for r in manifest['originals']}
    reports = list((REPO/'poc_flashvep/reports').glob('looped_pd_autonomous_*.md'))
    files = list(SOURCE.rglob('*.py')) + list(SOURCE.glob('LICENSE*')) + reports
    files += [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not p.name.endswith(('.lock', '.tmp')) and '.tmp.' not in p.name and p.name != 'burner.log']
    files = sorted({str(p.relative_to(REPO)) for p in files if p not in replaced},
        key=lambda name: ('/archives/' in name, name))
    assert all((REPO/p).stat().st_size < 100_000_000 for p in files)
    allowed = set(files)
    with tempfile.TemporaryDirectory(prefix='autonomous_results_index_') as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree', parent, env=env)
        batches, batch, size = [], [], 0
        for name in files:
            length = (REPO/name).stat().st_size
            if batch and size+length > 400*1024**2:
                batches.append(batch)
                batch, size = [], 0
            batch.append(name)
            size += length
        if batch:
            batches.append(batch)
        head = parent
        for number, batch in enumerate(batches, 1):
            for index in range(0, len(batch), 100):
                git('add', '-f', '--', *batch[index:index+100], env=env)
            changed = git('diff', '--cached', '--name-only', head, env=env).splitlines()
            assert set(changed) <= allowed
            if not changed:
                continue
            prose = [p for p in changed if p.endswith(('.md', '.py')) and '/raw/' not in p]
            if prose:
                git('diff', '--cached', '--check', head, '--', *prose, env=env)
            tree = git('write-tree', env=env)
            previous = head
            head = git('commit-tree', tree, '-p', previous, '-m',
                f'Record autonomous PD headroom evidence ({number}/{len(batches)})', env=env)
            git('update-ref', 'refs/heads/'+BRANCH, head, previous)
            preserve()
            if do_push:
                push(head)
            print(json.dumps(dict(batch=number, total=len(batches), commit=head, pushed=do_push)), flush=True)
    preserve()
    assert git('branch', '--show-current') == active
    remote = push(head) if do_push else None
    write(ROOT/'state/publication_receipt.json', dict(time=time.time(), branch=BRANCH, commit=head,
        remote_SHA=remote, verified=remote == head if do_push else False, original_index_preserved=True,
        external_runtime_preserved=True, files=len(files)))
    print(json.dumps(dict(branch=BRANCH, commit=head, remote_SHA=remote)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('archive', 'publish', 'restore'))
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--destination')
    args = parser.parse_args()
    if args.action == 'archive':
        archive()
    elif args.action == 'restore':
        assert args.destination, 'An explicit destination is required'
        from ..looped_pd_headroom_recovery import archive_evidence as inherited
        inherited.ROOT = ROOT
        inherited.restore(args.destination)
    else:
        publish(args.push)
