"""Copy numerical arrays and heavy traces outside Git. Keep original files."""
import shutil
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    destination=ASSETS/'raw_evidence'
    records=[]
    for pattern in ['*.npz','torch_events.rank*.json.gz']:
        for source in sorted((ROOT/'results').rglob(pattern)):
            relative=source.relative_to(ROOT)
            target=destination/relative
            digest=sha(source)
            if target.exists():assert sha(target)==digest,'Never overwrite a different archived artifact'
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
            assert sha(target)==digest
            records.append(dict(original_relative=str(relative),external_path=str(target),sha256=digest,bytes=source.stat().st_size))
    write(ROOT/'results/large_evidence_manifest.json',dict(files=records,
        scope='Original arrays and traces remain intact. External copies preserve exact bytes. Git retains this manifest.'))
    print('ARCHIVED',len(records),sum(r['bytes'] for r in records))


if __name__=='__main__':main()
