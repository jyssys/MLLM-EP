"""Package a redundant dataset below GitHub's limit without deleting its source."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from .common import ROOT, write


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'data/W1_long.json'
    packed = ROOT / 'data/W1_long.json.gz'
    manifest = ROOT / 'configs/dataset_storage.json'
    if args.package:
        with source.open('rb') as incoming, packed.open('wb') as output:
            with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0) as archive:
                shutil.copyfileobj(incoming, archive)
        digest = hashlib.sha256()
        with gzip.open(packed, 'rb') as incoming:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                digest.update(chunk)
        expected = sha(source)
        if digest.hexdigest() != expected:
            raise RuntimeError('Compressed dataset does not reconstruct the original bytes')
        write(manifest, dict(original=str(source.relative_to(ROOT)), compressed=str(packed.relative_to(ROOT)),
                             original_bytes=source.stat().st_size, original_sha256=expected,
                             compressed_bytes=packed.stat().st_size, compressed_sha256=sha(packed),
                             exact_decompression_verified=True, original_local_file_preserved=source.exists(),
                             scope='Storage-only change after GPU experiments. No prompt, tokenizer, arrival, or model change.'))
    if args.restore:
        data = json.loads(manifest.read_text())
        if source.exists():
            if sha(source) != data['original_sha256']:
                raise RuntimeError('Existing dataset differs. No file was overwritten.')
        else:
            if sha(packed) != data['compressed_sha256']:
                raise RuntimeError('Compressed dataset checksum differs')
            with gzip.open(packed, 'rb') as incoming, source.open('xb') as output:
                shutil.copyfileobj(incoming, output)
            if sha(source) != data['original_sha256']:
                raise RuntimeError('Restored dataset checksum differs')
    print(json.loads(manifest.read_text()))


if __name__ == '__main__':
    main()
