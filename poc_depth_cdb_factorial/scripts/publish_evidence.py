"""Publish lossless compressed raw traces. Keep original files on the server."""
import gzip
import hashlib
from poc_depth_cdb_factorial.contract import *

def main():
    sources=list((ROOT/'outputs').glob('*/rank_*/*.json'))
    for output in (ROOT/'outputs').iterdir():
        if not output.is_dir():continue
        for p in output.glob('*.json'):
            if p.name.startswith(('smoke_','speed_','trace_speed_','quality_','sensitivity_','order_control_')):
                d=read(p)
                if isinstance(d.get('case'),dict):sources.append(p)
    sources += [ROOT/'artifacts/rank_timeline.jsonl',ROOT/'artifacts/rank_timeline_summary.csv']
    rows=[]
    for source in sorted(sources):
        if not source.exists():continue
        raw=source.read_bytes()
        packed=gzip.compress(raw,compresslevel=6,mtime=0)
        target=Path(str(source)+'.gz')
        target.write_bytes(packed)
        assert gzip.decompress(target.read_bytes())==raw
        rows.append(dict(original=str(source.relative_to(ROOT)),published=str(target.relative_to(ROOT)),
            original_bytes=len(raw),compressed_bytes=len(packed),sha256=hashlib.sha256(raw).hexdigest(),
            original_preserved=True))
    write(ROOT/'artifacts/raw_compression_manifest.json',rows)
    print('LOSSLESS',len(rows),'files;',sum(r['original_bytes'] for r in rows),'->',sum(r['compressed_bytes'] for r in rows),'bytes')

if __name__=='__main__':main()
