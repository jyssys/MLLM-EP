"""Publish only this new namespace with a private Git index."""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time

from .common import BASE, BRANCH, REPO, ROOT, SOURCE, digest, preserve, preservation, write

DOCUMENTS = [
    'duetserve_agg_dp2_tp2_tbt_crossover_reproduction_spec.md',
    'duetserve_agg_dp2_tp2_tbt_crossover_primary_scope_revision_v2.md',
    'duetserve_agg_dp2_tp2_tbt_crossover_final_summary.md',
    'aggregated_dp2_qps_degradation_results.md',
    'aggregated_tp2_qps_degradation_results.md',
    'chunked_prefill_token_budget_sensitivity.md',
    'native_pd_decode_isolation_comparison.md',
    'aggregated_decode_interference_root_cause.md',
]


def archive():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as previous
    for key, value in dict(ROOT=ROOT, REPO=REPO, digest=digest, write=write).items():
        setattr(previous, key, value)
    from .common import resources
    previous.resources = resources
    select_raw = previous.selected_files
    def select_with_large_prompt_pools():
        pools = [path for path in (ROOT / 'data').glob('*')
                 if path.is_file() and path.stat().st_size >= 1024**2]
        return sorted(set(select_raw()) | set(pools), key=lambda path: (path.name, str(path)))
    previous.selected_files = select_with_large_prompt_pools
    previous.main()


def candidates():
    archived = {row['path'] for row in json.loads((ROOT / 'archives/large_evidence_manifest.json').read_text())['original_files']}
    paths, excluded = [], []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(REPO))
        reason = None
        if 'worker_source' in path.parts:
            reason = 'EXTRACTED_DUPLICATE_OF_VERIFIED_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:
            reason = 'REGENERABLE_CACHE_SOURCE_AND_HASH_PRESERVED'
        elif relative in archived:
            reason = 'VERIFIED_LOSSLESS_ARCHIVE'
        elif path.suffix == '.pdf' and 'references' in path.relative_to(ROOT).parts:
            reason = 'EXTERNAL_PAPER_URL_AND_HASH_PRESERVED_REDISTRIBUTION_NOT_ASSUMED'
        elif path.suffix in ('.lock', '.local') or '.tmp.' in path.name:
            reason = 'LOCAL_OPERATIONAL_FILE'
        elif path.name in ('publication_receipt.json', 'publication_progress.json'):
            reason = 'POST_COMMIT_LOCAL_RECEIPT'
        elif path.name == 'burner.log':
            reason = 'LIVE_BURNER_LOG_CLOSEOUT_SNAPSHOT_PRESERVED'
        elif path.suffix == '.bin' and 'state_transport' in path.parts:
            reason = 'EXPIRED_REGISTRATION_DESCRIPTOR_HASH_PRESERVED'
        if reason:
            excluded.append(dict(path=relative, bytes=path.stat().st_size, reason=reason))
            continue
        assert not path.is_symlink() and path.stat().st_size < 100 * 1024**2, relative
        paths.append(path)
    paths.extend(p for p in SOURCE.glob('*.py'))
    paths.extend(p for p in SOURCE.glob('*.cc'))
    paths.extend(REPO / 'poc_flashvep/reports' / name for name in DOCUMENTS)
    quick_stems = ('duetserve_qps_slo_discovery_o200', 'duetserve_qps_mean_ttft_discovery_o200',
                   'duetserve_qps_mean_tbt_discovery_o200', 'duetserve_qps_output_tokens_s_discovery_o200',
                   'duetserve_qps_four_metrics_discovery_o200')
    paths.extend(REPO / 'poc_flashvep' / (stem + suffix) for stem in quick_stems for suffix in ('.png', '.pdf')
                 if (REPO / 'poc_flashvep' / (stem + suffix)).exists())
    paths.extend(REPO / 'poc_flashvep' / name for name in
        ('duetserve_qps_four_metrics_discovery_o200_data.csv',
         'duetserve_qps_four_metrics_discovery_o200_metadata.json')
        if (REPO / 'poc_flashvep' / name).exists())
    confirmed = [stem.replace('_discovery_', '_confirmed_') for stem in quick_stems]
    paths.extend(REPO / 'poc_flashvep' / (stem + suffix) for stem in confirmed for suffix in ('.png', '.pdf')
                 if (REPO / 'poc_flashvep' / (stem + suffix)).exists())
    paths.extend(REPO / 'poc_flashvep' / name for name in
        ('duetserve_qps_four_metrics_confirmed_o200_data.csv',
         'duetserve_qps_four_metrics_confirmed_o200_metadata.json')
        if (REPO / 'poc_flashvep' / name).exists())
    assert all(p.exists() for p in paths)
    return sorted(set(paths)), excluded


def git(args, env=None, payload=None):
    result = subprocess.run(['git', *args], cwd=REPO, env=env, input=payload, capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-4000:]
    return result.stdout


def stage(paths, env):
    assert 'duetserve_private_' in env['GIT_INDEX_FILE']
    relative = [str(p.relative_to(REPO)) for p in paths]
    assert all(n.isascii() and not any(c.isspace() for c in n) for n in relative)
    hashes = git(['hash-object', '-w', '--stdin-paths'], env,
                 ''.join(n + '\n' for n in relative).encode()).decode().splitlines()
    assert len(hashes) == len(relative)
    entries = []
    for path, name, sha in zip(paths, relative, hashes):
        assert not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
        mode = '100755' if path.stat().st_mode & 0o111 else '100644'
        entries.append(f'{mode} {sha}\t{name}\0'.encode())
    git(['update-index', '--add', '-z', '--index-info'], env, b''.join(entries))


def proof():
    paths = [SOURCE / 'publication.py', ROOT / 'configs/frozen_SLO_regimes.json',
             REPO / 'poc_flashvep/duetserve_qps_four_metrics_discovery_o200.png',
             REPO / 'poc_flashvep/duetserve_qps_four_metrics_discovery_o200.pdf',
             REPO / 'poc_flashvep/duetserve_qps_four_metrics_discovery_o200_data.csv']
    preserve()
    with tempfile.TemporaryDirectory(prefix='duetserve_private_proof_') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git(['read-tree', '--empty'], env)
        payload = b''.join(str(p.relative_to(REPO)).encode() + b'\0' for p in paths)
        git(['add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], env, payload)
        expected = git(['write-tree'], env).decode().strip()
        git(['read-tree', '--empty'], env)
        stage(paths, env)
        observed = git(['write-tree'], env).decode().strip()
        assert expected == observed
    preserve()
    write(ROOT / 'state/private_staging_tree_proof.json', dict(time=time.time(),
        expected_tree=expected, observed_tree=observed, passed=True,
        files=[str(p.relative_to(REPO)) for p in paths],
        scope='Normal git add and explicit object staging create the same temporary-index tree. The user index is untouched.'))
    print('PRIVATE_STAGING_TREE_EQUAL', observed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        proof()
        return
    assert (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    if args.archive:
        archive()
        return
    before = preservation()
    paths, excluded = candidates()
    coverage = ROOT / 'state/publication_coverage.json'
    write(coverage, dict(selected_files=len(paths), selected_bytes=sum(p.stat().st_size for p in paths),
        excluded=excluded, source_parent=BASE,
        scope='Only new source, reports, and evidence are staged. Original raw files and large prompt pools remain local. Every archived member has a verified decompressed SHA256. No external paper is redistributed.'))
    paths.append(coverage)
    parent = git(['rev-parse', BRANCH]).decode().strip()
    assert parent == BASE or git(['merge-base', BASE, parent]).decode().strip() == BASE
    with tempfile.TemporaryDirectory(prefix='duetserve_private_') as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / 'index'))
        git(['read-tree', parent], env)
        stage(paths, env)
        tree = git(['write-tree'], env).decode().strip()
        if tree != git(['rev-parse', parent + '^{tree}']).decode().strip():
            message = b'DuetServe QPS characterization with exactness and interference gates\n\nPreserve independent starts, long-window controls, and failed captures.\n'
            commit = git(['commit-tree', tree, '-p', parent], env, message).decode().strip()
            git(['update-ref', 'refs/heads/' + BRANCH, commit, parent])
        else:
            commit = parent
    if args.push:
        result = subprocess.run(['git', '-c', 'pack.threads=2', 'push', 'origin',
                                 commit + ':refs/heads/' + BRANCH], cwd=REPO, capture_output=True, text=True)
        write(ROOT / 'state/push_command_result.json', dict(time=time.time(), returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, result.stderr[-4000:]
    remote = git(['ls-remote', 'origin', 'refs/heads/' + BRANCH]).decode().split()
    remote_sha = remote[0] if remote else None
    if args.push:
        assert remote_sha == commit
    preserve()
    assert before == preservation()
    write(ROOT / 'state/publication_receipt.json', dict(time=time.time(), branch=BRANCH, SHA=commit,
        Remote_SHA=remote_sha, pushed=args.push, remote_matches=remote_sha == commit,
        user_checkout_and_index_preserved=True, external_runtime_preserved=True,
        scope='Private-index publication. The post-commit receipt remains local. No force push or checkout occurs.'))
    print(json.dumps(dict(branch=BRANCH, SHA=commit, Remote_SHA=remote_sha)), flush=True)


if __name__ == '__main__':
    main()
