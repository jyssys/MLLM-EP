"""Stage explicit files without a large pathspec directory search."""
import argparse
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from poc_flashvep.dense_ar_slo_layer_pd.common import REPO, ROOT, preserve, write
from poc_flashvep.dense_ar_slo_layer_pd import publish

ORIGINAL_RUN = subprocess.run


def command(args, env, payload=None):
    value = ORIGINAL_RUN(['git', *args], cwd=REPO, env=env,
        input=payload, capture_output=True)
    assert value.returncode == 0, value.stderr.decode(errors='replace')[-4000:]
    return value.stdout


def stage(paths, env):
    index = Path(env['GIT_INDEX_FILE'])
    assert 'dynamic_layer_private_index_' in str(index) or 'fast_stage_proof_' in str(index)
    relative = [str(Path(path).relative_to(REPO)) if Path(path).is_absolute() else path for path in paths]
    assert all(name.isascii() and not any(char.isspace() for char in name) for name in relative)
    hashes = command(['hash-object', '-w', '--stdin-paths'], env,
        ''.join(name + '\n' for name in relative).encode()).decode().splitlines()
    assert len(hashes) == len(relative)
    entries = []
    for name, object_id in zip(relative, hashes):
        path = REPO / name
        value = path.stat()
        assert stat.S_ISREG(value.st_mode) and not path.is_symlink()
        mode = '100755' if value.st_mode & 0o111 else '100644'
        assert len(object_id) == 40 and all(char in '0123456789abcdef' for char in object_id)
        entries.append(f'{mode} {object_id}\t{name}\0'.encode())
    command(['update-index', '--add', '-z', '--index-info'], env, b''.join(entries))


def proof():
    paths = [
        REPO / 'poc_flashvep/dense_ar_slo_layer_pd/policy.py',
        REPO / 'poc_flashvep/reports/dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md',
        ROOT / 'configs/final_confirmation_selection.json',
        ROOT / 'figures/final_9_confirmed_capacity_with_backoff.svg',
        ROOT / 'archives/large_evidence.tar.zst.part0004',
    ]
    preserve()
    with tempfile.TemporaryDirectory(prefix='fast_stage_proof_') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder) / 'index'))
        assert command(['config', '--bool', 'core.filemode'], env).strip() == b'true'
        command(['read-tree', '--empty'], env)
        payload = b''.join(str(path.relative_to(REPO)).encode() + b'\0' for path in paths)
        command(['add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], env, payload)
        reference = command(['write-tree'], env).decode().strip()
        command(['read-tree', '--empty'], env)
        stage(paths, env)
        observed = command(['write-tree'], env).decode().strip()
        assert observed == reference, 'Fast staging changed a Git tree'
    preserve()
    write(ROOT / 'state/private_fast_staging_proof.json', dict(passed=True,
        sample_files=[str(path.relative_to(REPO)) for path in paths],
        reference_tree=reference, fast_tree=observed,
        scope='Both methods use a temporary index. File content and executable modes give the same Git tree. The user index and working tree are unchanged.'))
    print('FAST_PRIVATE_STAGING_TREE_EQUAL', reference, flush=True)


def intercepted_run(args, *positional, **kwargs):
    if args[:2] == ['git', 'add'] and '--pathspec-from-file=-' in args:
        env = kwargs['env']
        names = [name.decode() for name in kwargs['input'].split(b'\0') if name]
        stage(names, env)
        print('FAST_PRIVATE_INDEX_STAGED', len(names), flush=True)
        return subprocess.CompletedProcess(args, 0, stdout=b'', stderr=b'')
    return ORIGINAL_RUN(args, *positional, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--push', action='store_true')
    options = parser.parse_args()
    proof()
    if options.selftest:
        return
    subprocess.run = intercepted_run
    import sys
    sys.argv = [sys.argv[0]] + (['--push'] if options.push else [])
    publish.main()


if __name__ == '__main__':
    main()
