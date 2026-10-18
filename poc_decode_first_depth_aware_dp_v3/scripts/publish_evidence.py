"""Create lossless compact evidence. Keep every original file on the server."""
import gzip
import hashlib
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    records=[]
    for source in sorted((ROOT/'outputs').glob('*/rank_*/*.json')):
        target=Path(str(source)+'.gz')
        raw=source.read_bytes()
        if target.exists():
            assert gzip.decompress(target.read_bytes())==raw
        else:
            with target.open('xb') as stream:
                stream.write(gzip.compress(raw,compresslevel=9,mtime=0))
        records.append(dict(original=str(source.relative_to(ROOT)),
            published=str(target.relative_to(ROOT)),original_sha256=hashlib.sha256(raw).hexdigest(),
            original_bytes=len(raw),published_bytes=target.stat().st_size,lossless=True))
    write(ROOT/'artifacts/raw_compression_manifest.json',dict(files=records,
        original_files_preserved_locally=True,
        original_bytes=sum(r['original_bytes'] for r in records),
        published_bytes=sum(r['published_bytes'] for r in records)))

if __name__=='__main__':main()
