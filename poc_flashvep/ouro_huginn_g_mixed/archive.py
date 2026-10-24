"""Lossless portable evidence. Keep every original file in place."""
import json
from pathlib import Path
import tarfile
from .common import ROOT,write,digest
from .analyze import csvfile

def main():
    statuses={}
    for line in (ROOT/'run_manifest.jsonl').read_text().splitlines():
        row=json.loads(line);statuses[row['job']]=row['status']
    assert not [k for k,v in statuses.items() if v=='RUNNING'],'Do not archive a live attempt'
    folder=ROOT/'archives';folder.mkdir(exist_ok=True)
    rows=[];archives=[]
    packages=[]
    for raw in sorted((ROOT/'raw').iterdir()):
        if raw.is_dir():
            files=sorted(f for f in raw.rglob('*') if f.is_file())
            files+=sorted((ROOT/'logs').glob(raw.name+'_gpu*.log'))
            packages.append((raw.name,files))
    packages.append(('large_derived_CSV',[ROOT/'batch_plan_trace.csv',ROOT/'all_ITL_gaps.csv']))
    for label,files in packages:
        path=folder/f'{label}.tar.xz'
        assert not path.exists(),'Archive attempt is immutable'
        with tarfile.open(path,'x:xz',preset=6) as archive:
            for file in files:
                assert file.suffix not in ['.pt','.safetensors','.bin']
                relative=str(file.relative_to(ROOT))
                archive.add(file,arcname=relative,recursive=False)
                rows.append(dict(path=relative,bytes=file.stat().st_size,sha256=digest(file),archive=str(path.relative_to(ROOT))))
        assert path.stat().st_size<95*1024**2
        archives.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
    write(ROOT/'raw_archive_manifest.json',dict(files=rows,archives=archives,
        originals_retained=True,contains_model_weights=False))
    csvfile('evidence_hashes.csv',rows)
    print('ARCHIVED',len(rows),'files',sum(a['bytes'] for a in archives),'compressed bytes')

if __name__=='__main__':main()
