"""Publish this campaign through a private index. Preserve the user checkout."""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
from .common import BASE, BRANCH, REPO, ROOT, SOURCE, digest, preserve, preservation, write
from ..dense_ar_layer_token_pd.publication import git, stage


MAX_BLOB_BYTES = 100 * 1024**2


DOCUMENTS = (
    'dense_ar_longbench400_dataset_contract.md',
    'dense_ar_longbench400_motivation_agg_native.md',
    'dense_ar_longbench400_input_output_axis_winner_map.md',
    'dense_ar_layer_token_2d_hybrid_design.md',
    'dense_ar_layer_token_2d_hybrid_correctness.md',
    'dense_ar_layer_token_2d_hybrid_performance.md',
    'dense_ar_layer_token_2d_continuous_serving.md',
    'dense_ar_layer_token_2d_dynamic_scheduler.md',
    'dense_ar_longbench400_layer_token_2d_final_summary.md',
    'dense_ar_longbench400_bi0_numerical_reference_audit.md',
    'dense_ar_longbench400_isowork_hybrid_checkpoint.md',
    'dense_ar_longbench400_pipeline_load_reference_audit.md',
    'dense_ar_longbench400_kv_work_accounting.md',
    'dense_ar_longbench400_decode_tail_recovery.md',
    'dense_ar_longbench400_source_credit_chunk_causality.md',
    'dense_ar_longbench400_figures_and_reproduction.md',
    'dense_ar_longbench400_controlled_continuous_bridge.md',
    'dense_ar_longbench400_axis_cost_model.md',
    'dense_ar_longbench400_decode_tail_intervention_contract.md',
    'dense_ar_longbench400_interim_20261121_1715.md',
    'dense_ar_longbench400_itl_causal_recovery.md',
    'dense_ar_longbench400_operator_cost_checkpoint_20261121.md',
    'dense_ar_longbench400_related_work_scope.md',
    'dense_ar_longbench400_request_registration_tail.md',
    'dense_ar_longbench400_request_registration_tail_q28.md',
    'dense_ar_longbench400_registration_controlled_bridge.md',
    'dense_ar_longbench400_decode_compile_residual.md',
    'dense_ar_longbench400_scoped_decode_compile_results.md',
    'dense_ar_longbench400_eager_prefill_native_decode_results.md',
)


def archive_metadata_payload(archive, proof):
    """Keep a large ledger intact through a verified gzip and small index."""
    if archive.stat().st_size < MAX_BLOB_BYTES:
        return [archive], archive
    compressed = archive.with_name(archive.name + '.gz')
    index = archive.with_name(archive.stem + '.index.json')
    expected_hash, expected_bytes = digest(archive), archive.stat().st_size
    if not compressed.exists():
        with archive.open('rb') as source, compressed.open('wb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as stream:
                shutil.copyfileobj(source, stream, length=1024**2)
    observed_hash = hashlib.sha256()
    observed_bytes = 0
    with gzip.open(compressed, 'rb') as stream:
        while block := stream.read(8 * 1024**2):
            observed_hash.update(block)
            observed_bytes += len(block)
    assert observed_bytes == expected_bytes and observed_hash.hexdigest() == expected_hash
    assert compressed.stat().st_size < MAX_BLOB_BYTES, 'Split a larger metadata stream before publication'
    document = dict(schema='LOSSLESS_COMPRESSED_ARCHIVE_LEDGER_V1',
        verified_lossless=proof['verified_lossless'],
        expanded_manifest=dict(path=str(archive.relative_to(REPO)), bytes=expected_bytes, SHA256=expected_hash),
        compressed_manifest=dict(path=str(compressed.relative_to(REPO)),
            bytes=compressed.stat().st_size, SHA256=digest(compressed)),
        public_archive=proof['public_archive'], local_archive=proof.get('local_archive'),
        scope='The expanded ledger remains local. The published gzip restores its exact original bytes. '
              'Verify the compressed SHA256,then decompress and verify the expanded SHA256. '
              'The ledger preserves all member hashes and original failure records. No raw evidence is changed.')
    write(index, document)
    assert index.stat().st_size < MAX_BLOB_BYTES
    return [compressed, index], index


def prepare():
    archive = ROOT / 'archives/evidence_archive_manifest_v1.json'
    proof = json.loads(archive.read_text())
    assert proof['verified_lossless']
    public_metadata, public_index = archive_metadata_payload(archive, proof)
    public_metadata = set(public_metadata)
    archived = {row['path'] for row in proof['local_members']}
    public_parts = {row['path'] for row in proof['public_archive']['parts']}
    selected, excluded = [], []
    local_names = {'publication_receipt.json', 'publication_progress.json', 'post_publication_final_check.json'}
    generated = {'publication_coverage.json', 'publication_SHA256_manifest.json', 'publication_exclusions.jsonl.gz'}
    for folder, directories, names in os.walk(ROOT):
        for name in list(directories):
            if name in ('worker_source', '__pycache__', 'tools', '.cache'):
                excluded.append(dict(path=str((Path(folder) / name).relative_to(REPO)), reason='DUPLICATE_SOURCE_OR_REGENERABLE_CACHE'))
                directories.remove(name)
        for name in names:
            path = Path(folder) / name
            assert not path.is_symlink(), path
            relative = str(path.relative_to(REPO))
            if name in generated:
                continue
            reason = None
            if relative in archived:
                reason = 'VERIFIED_LOSSLESS_ARCHIVE'
            elif 'archives' in path.relative_to(ROOT).parts and path not in public_metadata and relative not in public_parts:
                reason = 'LOCAL_ARCHIVE_OR_RECONSTRUCTABLE_PUBLIC_STREAM'
            elif name == 'burner.log':
                reason = 'LIVE_LOG_FROZEN_CLOSEOUT_SNAPSHOT_PUBLISHED'
            elif name == 'background_final_delivery_v1.log':
                reason = 'LIVE_PUBLICATION_CONTROLLER_LOG_LOCAL_RECEIPT'
            elif name in local_names:
                reason = 'POST_COMMIT_LOCAL_RECEIPT'
            elif path.suffix in ('.lock', '.local') or '.tmp.' in name:
                reason = 'LOCAL_OPERATIONAL_FILE'
            if reason:
                excluded.append(dict(path=relative, bytes=path.stat().st_size, reason=reason))
            else:
                selected.append(path)
    selected.extend(p for p in SOURCE.glob('*') if p.suffix in ('.py', '.cc', '.cu', '.h'))
    selected.extend(REPO / 'poc_flashvep/reports' / name for name in DOCUMENTS)
    assert all(p.exists() and p.is_file() and not p.is_symlink() for p in selected)
    assert all(p.stat().st_size < MAX_BLOB_BYTES for p in selected), 'Archive each oversized evidence file before staging'
    exclusions = ROOT / 'state/publication_exclusions.jsonl.gz'
    with gzip.open(exclusions, 'wt', encoding='utf-8') as stream:
        for row in excluded:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    payload = sorted(set(selected + [exclusions]))
    coverage = ROOT / 'state/publication_coverage.json'
    write(coverage, dict(base=BASE, selected_files=len(payload) + 2,
        payload_bytes=sum(p.stat().st_size for p in payload),
        exclusion_counts=dict(collections.Counter(row['reason'] for row in excluded)),
        archive_manifest=str(public_index.relative_to(REPO)),
        scope='Publish only new source, reports, figures and necessary evidence. '
            'Verified public archive parts include failed runs. Large diagnostic tensors remain in a verified local archive. '
            'Every local-only tensor has a hash in the archive manifest. No user-index or checkout change occurs.'))
    hashes = ROOT / 'state/publication_SHA256_manifest.json'
    write(hashes, dict(files=[dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size, SHA256=digest(p))
        for p in payload + [coverage]], self_included=False,
        scope='The final Git commit binds this manifest. Post-commit receipts remain local.'))
    return sorted(set(payload + [coverage, hashes]))


def selftest():
    preserve()
    paths = [SOURCE / 'publication.py', ROOT / 'configs/campaign_contract_v1.json']
    with tempfile.TemporaryDirectory(prefix='layer_token_private_longbench400_proof_') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder) / 'index'))
        git(['read-tree', '--empty'], env)
        git(['add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], env,
            b''.join(str(p.relative_to(REPO)).encode() + b'\0' for p in paths))
        expected = git(['write-tree'], env).decode().strip()
        git(['read-tree', '--empty'], env)
        stage(paths, env)
        observed = git(['write-tree'], env).decode().strip()
    assert expected == observed
    preserve()
    write(ROOT / 'state/private_staging_tree_proof.json', dict(passed=True, expected_tree=expected,
        observed_tree=observed, user_checkout_and_index_preserved=True))
    print('PRIVATE_INDEX_PROOF_PASS', observed, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    assert (ROOT / 'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    before = preservation()
    paths = prepare()
    if args.prepare:
        print('PUBLICATION_PREPARED', len(paths), flush=True)
        return
    parent = git(['rev-parse', BRANCH]).decode().strip()
    assert parent == BASE or git(['merge-base', BASE, parent]).decode().strip() == BASE
    chunks, current, size = [], [], 0
    for path in sorted(paths, key=lambda p: ('.tar.zst.part' in p.name, str(p))):
        if current and size + path.stat().st_size > 192 * 1024**2:
            chunks.append(current)
            current, size = [], 0
        current.append(path)
        size += path.stat().st_size
    if current:
        chunks.append(current)
    receipt = ROOT / 'state/publication_progress.json'
    progress = json.loads(receipt.read_text()) if receipt.exists() else []
    with tempfile.TemporaryDirectory(prefix='layer_token_private_longbench400_') as folder:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder) / 'index'))
        git(['read-tree', parent], env)
        for index, chunk in enumerate(chunks):
            stage(chunk, env)
            tree = git(['write-tree'], env).decode().strip()
            if tree != git(['rev-parse', parent + '^{tree}']).decode().strip():
                message = (f'LongBench400 2D PD evidence {index + 1}/{len(chunks)}\n\n'
                    'Preserve correctness failures, paired decode protection, and evidence scope.\n').encode()
                commit = git(['commit-tree', tree, '-p', parent], env, message).decode().strip()
                git(['update-ref', 'refs/heads/' + BRANCH, commit, parent])
                parent = commit
            if args.push:
                result = subprocess.run(['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false',
                    '-c', 'pack.threads=2', 'push', 'origin', parent + ':refs/heads/' + BRANCH],
                    cwd=REPO, capture_output=True, text=True)
                progress.append(dict(time=time.time(), checkpoint=index + 1, SHA=parent,
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                write(receipt, progress)
                assert result.returncode == 0, result.stderr[-3000:]
            preserve()
            print('PUBLICATION_CHECKPOINT', index + 1, len(chunks), parent, flush=True)
    remote = git(['ls-remote', 'origin', 'refs/heads/' + BRANCH]).decode().split()
    remote_sha = remote[0] if remote else None
    if args.push:
        assert remote_sha == parent
    preserve()
    assert preservation() == before
    write(ROOT / 'state/publication_receipt.json', dict(time=time.time(), branch=BRANCH, SHA=parent,
        Remote_SHA=remote_sha, pushed=args.push, remote_matches=remote_sha == parent,
        user_checkout_and_index_preserved=True))
    print(json.dumps(dict(branch=BRANCH, SHA=parent, Remote_SHA=remote_sha)), flush=True)


if __name__ == '__main__':
    main()
