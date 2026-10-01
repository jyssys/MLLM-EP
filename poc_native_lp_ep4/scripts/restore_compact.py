"""Restore published compact JSON evidence for independent CPU reanalysis."""
import argparse
import gzip
import hashlib
from pathlib import Path

from poc_native_lp_ep4.contract import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('compact',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=read_json(a.compact/'integrity.json')
    count=0
    a.output.mkdir(parents=True)
    for item in manifest['files']:
        if 'path' not in item:continue
        path=Path(item['path'])
        if path.is_absolute() or '..' in path.parts or len(path.parts)!=2:
            raise ValueError('invalid archived relative path')
        source=a.compact/path
        packed=source.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=item['compressed_sha256']:
            raise ValueError(f'compressed integrity failure: {path}')
        raw=gzip.decompress(packed)
        if hashlib.sha256(raw).hexdigest()!=item['raw_sha256']:
            raise ValueError(f'raw integrity failure: {path}')
        target=a.output/path.with_suffix('');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        count+=1
    print('RESTORED_VERIFIED_JSON_FILES',count,a.output)


if __name__=='__main__':main()
