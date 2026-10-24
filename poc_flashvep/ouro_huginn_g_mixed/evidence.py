"""Freeze source evidence and preserve the removable runtime patch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .common import ROOT,REPO,FORK,MODELS,UPSTREAM,SPEC,write,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--weights',action='store_true');a=p.parse_args()
    folder=ROOT/'provenance'/a.label
    assert not folder.exists();folder.mkdir(parents=True)
    if a.weights:
        rows=[]
        for name,snapshot in MODELS.items():
            for file in sorted(snapshot.glob('*.safetensors')):
                actual=digest(file);blob=file.resolve().name
                if len(blob)==64:assert actual==blob,(name,file,actual,blob)
                rows.append(dict(model=name,path=str(file),bytes=file.stat().st_size,
                    sha256=actual,cache_blob=blob,LFS_SHA_matches=actual==blob))
        write(folder/'model_weight_hashes.json',rows)
        print('CHECKPOINT_FILES_HASHED',len(rows));return
    patch=subprocess.check_output(['git','diff','--binary'],cwd=FORK)
    # This file is a generated Git artifact, not a source rewrite.
    (folder/'runtime.patch').write_bytes(patch)
    rows=[]
    for path in sorted((FORK/'vllm_rlt').rglob('*.py')):
        rows.append(dict(path=str(path.relative_to(FORK)),sha256=digest(path)))
    write(folder/'source_hashes.json',rows)
    write(folder/'spec_snapshot.json',dict(path=str(SPEC),sha256=digest(SPEC),text=SPEC.read_text()))
    write(folder/'model_files.json',{
        name:[dict(name=f.name,bytes=f.stat().st_size,sha256=digest(f))
            for f in sorted(path.iterdir()) if f.is_file() and f.suffix in ('.json','.py','.md')]
        for name,path in MODELS.items()})
    owners=[]
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for line in output.splitlines():
        uuid,pid=line.split(',');proc=Path('/proc')/pid.strip()
        owners.append(dict(uuid=uuid.strip(),pid=int(pid),uid=proc.stat().st_uid,
            sid=os.getsid(int(pid)),cmd=(proc/'cmdline').read_bytes().decode().split('\0'),
            stdout=str((proc/'fd/1').resolve())))
    write(folder/'process_owners.json',owners)
    write(folder/'repository.json',dict(upstream=UPSTREAM,
        parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        indexed_diff_sha=__import__('hashlib').sha256(subprocess.check_output(
            ['git','diff','--cached','--binary'],cwd=REPO)).hexdigest()))

if __name__=='__main__':main()
