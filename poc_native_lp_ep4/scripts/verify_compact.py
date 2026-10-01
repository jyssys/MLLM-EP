import argparse
import gzip
import hashlib
from pathlib import Path
from poc_native_lp_ep4.contract import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);args=p.parse_args()
    count=0
    for row in read_json(args.bundle/'integrity.json')['files']:
        if 'path' not in row:continue
        path=args.bundle/row['path'];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['compressed_sha256'],path
        assert hashlib.sha256(gzip.decompress(data)).hexdigest()==row['raw_sha256'],path
        count+=1
    print(f'PASS {count} compressed raw evidence files')


if __name__=='__main__':main()
