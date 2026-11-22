"""Preserve raw evidence locally and publish a smaller lossless evidence set."""
import argparse
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from .common import ROOT, REPO, digest, write, preserve


def files():
    """Skip only duplicate source copies and reproducible local caches."""
    selected, omitted = [], []
    for top in ('raw', 'data', 'references'):
        for folder, directories, names in os.walk(ROOT / top):
            for name in list(directories):
                if name in ('worker_source', '__pycache__', '.cache'):
                    path = Path(folder) / name
                    omitted.append(dict(path=str(path.relative_to(REPO)),
                        reason='DUPLICATE_FROZEN_SOURCE' if name == 'worker_source' else 'REGENERABLE_CACHE'))
                    directories.remove(name)
            for name in names:
                path = Path(folder) / name
                assert not path.is_symlink(), path
                if path.suffix in ('.lock', '.local') or '.tmp.' in name:
                    omitted.append(dict(path=str(path.relative_to(REPO)), reason='LOCAL_OPERATIONAL_FILE'))
                    continue
                selected.append(path)
    return sorted(selected), omitted


@cache
def canonical_sources():
    values={}
    for path in sorted((ROOT/'references/source_capsules').glob('*.zip')):
        values.setdefault(digest(path),path)
    return values


@cache
def source_reference(path):
    if path.name!='source.zip' or path.parent.parent!=ROOT/'raw':
        return None
    receipt=path.parent/'dispatcher_result.json'
    if not receipt.exists():
        return None
    sha=json.loads(receipt.read_text()).get('source_SHA256')
    canonical=canonical_sources().get(sha)
    if canonical is None:
        return None
    assert digest(path)==sha
    return dict(path=str(path.relative_to(REPO)),canonical_path=str(canonical.relative_to(REPO)),SHA256=sha)


def public_reason(path):
    if source_reference(path) is not None:
        return 'BYTE_IDENTICAL_SOURCE_CAPSULE_PUBLISHED_ONCE'
    if path.suffix in ('.pt', '.pth', '.npy', '.npz', '.safetensors'):
        return 'LOCAL_LOSSLESS_TENSOR_ARCHIVE_WITH_SHA256'
    if path.suffix == '.pdf' and 'references' in path.relative_to(ROOT).parts:
        return 'EXTERNAL_PAPER_NOT_REDISTRIBUTED'
    if path.suffix == '.bin' and any(p in path.parts for p in ('state_transport','hidden_transport')):
        return 'EXPIRED_TRANSPORT_DESCRIPTOR_LOCAL_ARCHIVE_WITH_SHA256'
    return None


def pack(paths, destination):
    """Create an unmodified tar stream. Verify every decompressed member."""
    assert not destination.exists(), destination
    ledger = []
    with destination.open('wb') as output:
        child = subprocess.Popen(['zstd', '-q', '-T2', '-3', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=child.stdin, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    before = path.stat()
                    row = dict(path=str(path.relative_to(REPO)), bytes=before.st_size, SHA256=digest(path))
                    archive.add(path, arcname=row['path'], recursive=False)
                    after = path.stat()
                    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
                    ledger.append(row)
            child.stdin.close()
            assert child.wait() == 0
        except BaseException:
            child.stdin.close()
            child.wait()
            raise
    verify(destination, ledger)
    return ledger


def verify(path, expected):
    wanted = {row['path']: row for row in expected}
    observed = set()
    child = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=child.stdout, mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in wanted and member.name not in observed, member.name
                stream = archive.extractfile(member)
                value = hashlib.sha256()
                count = 0
                while block := stream.read(8 * 1024**2):
                    value.update(block)
                    count += len(block)
                row = wanted[member.name]
                assert count == row['bytes'] and value.hexdigest() == row['SHA256'], member.name
                observed.add(member.name)
        assert child.wait() == 0
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.terminate()
            child.wait()
    assert observed == set(wanted), 'The archive must preserve every selected original file'


def split(path, size=48 * 1024**2):
    parts = []
    with path.open('rb') as source:
        index = 0
        while block := source.read(size):
            destination = path.with_name(path.name + f'.part{index:04d}')
            assert not destination.exists()
            with destination.open('wb') as output:
                output.write(block)
            parts.append(dict(path=str(destination.relative_to(REPO)), bytes=len(block),
                SHA256=hashlib.sha256(block).hexdigest()))
            index += 1
    merged = hashlib.sha256()
    for row in parts:
        with (REPO / row['path']).open('rb') as stream:
            while block := stream.read(8 * 1024**2):
                merged.update(block)
    assert merged.hexdigest() == digest(path)
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        # The real stream codec and member verifier operate on small existing files.
        with tempfile.TemporaryDirectory(prefix='longbench400_archive_test_', dir=ROOT / 'tools') as folder:
            destination = Path(folder) / 'test.tar.zst'
            paths = [ROOT / 'configs/campaign_contract_v1.json', Path(__file__)]
            rows = pack(paths, destination)
            assert len(rows) == 2
        print('LOSSLESS_ARCHIVE_STREAM_SELFTEST_PASS', flush=True)
        return
    assert (ROOT / 'state/CAMPAIGN_CLOSED.json').exists(), 'Do not archive mutable GPU evidence'
    preserve()
    output = ROOT / 'archives'
    output.mkdir(exist_ok=True)
    result = output / 'evidence_archive_manifest_v1.json'
    assert not result.exists(), 'Keep the verified archive manifest immutable'
    selected, omissions = files()
    local_path = output / 'local_full_evidence_v1.tar.zst'
    local_members = pack(selected, local_path)
    public_files = [p for p in selected if public_reason(p) is None]
    public_path = output / 'public_essential_evidence_v1.tar.zst'
    public_members = pack(public_files, public_path)
    parts = split(public_path)
    public_names = {row['path'] for row in public_members}
    local_only = [dict(row, reason=public_reason(REPO / row['path']))
        for row in local_members if row['path'] not in public_names]
    write(result, dict(time=time.time(), verified_lossless=True,
        local_archive=dict(path=str(local_path.relative_to(REPO)), bytes=local_path.stat().st_size,
            SHA256=digest(local_path), public=False),
        public_archive=dict(path=str(public_path.relative_to(REPO)), bytes=public_path.stat().st_size,
            SHA256=digest(public_path), parts=parts),
        local_members=local_members, public_members=public_members,
        local_only=local_only, excluded_directory_or_operational_paths=omissions,
        duplicate_source_references=[source_reference(p) for p in selected if source_reference(p) is not None],
        scope='Every selected member passed decompressed byte-count and SHA256 checks. '
            'Original raw files remain local. Public parts preserve all selected textual evidence, including failed runs. '
            'Large diagnostic tensors remain in the verified local archive with per-file hashes. '
            'Byte-identical per-job source ZIPs also remain in the local archive. '
            'Their public canonical capsule path and identical SHA256 are recorded for reconstruction. '
            'External papers remain local. Extracted source copies refer to immutable source capsules. '
            'Reconstruct the public stream by concatenating parts in manifest order. Decompress with zstd. '
            'Extract at the repository root only after checking archive paths and SHA256.'))
    preserve()
    print('LOSSLESS_LOCAL_AND_PUBLIC_ARCHIVES_PASS', len(local_members), len(public_members), len(parts), flush=True)


if __name__ == '__main__':
    main()
