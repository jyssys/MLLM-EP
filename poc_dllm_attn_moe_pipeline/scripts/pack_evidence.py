"""Lossless gzip export; failed logs and all captured metadata are retained."""
import argparse
import gzip
import hashlib
from poc_dllm_attn_moe_pipeline.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
    if a.restore:
        manifest=json.loads((ROOT/'evidence/integrity.json').read_text())
        for row in manifest['files']:
            raw=ROOT/row['raw']
            data=gzip.decompress((ROOT/row['packed']).read_bytes())
            assert hashlib.sha256(data).hexdigest()==row['sha256']
            if raw.exists():
                assert sha(raw)==row['sha256'],f'Do not overwrite existing divergent evidence: {raw}'
                continue
            raw.parent.mkdir(parents=True,exist_ok=True)
            raw.write_bytes(data)
        print('RESTORED_IDENTICAL_BYTES',len(manifest['files']))
        return
    records=[]
    for path in sorted((ROOT/'raw').rglob('*')):
        if not path.is_file():
            continue
        assert path.suffix in ('.json','.log'),f'Unexpected tensor/secret file: {path}'
        relative=path.relative_to(ROOT/'raw')
        destination=ROOT/'evidence'/relative.parent/(relative.name+'.gz')
        data=path.read_bytes()
        packed=gzip.compress(data,compresslevel=9,mtime=0)
        assert gzip.decompress(packed)==data
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(packed)
        records.append(dict(raw=str(path.relative_to(ROOT)),packed=str(destination.relative_to(ROOT)),
            bytes=len(data),packed_bytes=len(packed),sha256=hashlib.sha256(data).hexdigest()))
    write(ROOT/'evidence/integrity.json',dict(lossless=True,contains_model_weights_or_hidden=False,
        files=records,total_bytes=sum(r['bytes'] for r in records),packed_bytes=sum(r['packed_bytes'] for r in records)))
    print('PACKED',len(records),sum(r['packed_bytes'] for r in records))


if __name__=='__main__':
    main()
