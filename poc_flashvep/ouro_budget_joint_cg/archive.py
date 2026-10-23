"""Preserve closed raw attempts locally and publish their exact hashes."""
import csv
import hashlib
import json
import subprocess
import tarfile
import time
from .common import ROOT,REPO,write,digest
from .cost_predictor import csv_file

def main():
    processes=subprocess.check_output(['ps','-eo','args'],text=True)
    active=[line for line in processes.splitlines() if ' -m poc_flashvep.ouro_budget_joint_cg.worker --job ' in line]
    assert not active,('Do not archive active attempts',active)
    archive_root=ROOT/'archives';archive_root.mkdir(exist_ok=True)
    index_path=ROOT/'raw_archive_manifest.json'
    if index_path.exists():
        index=json.loads(index_path.read_text())
        for item in index['archives']:
            assert digest(item['absolute_path'])==item['sha256'],'Existing raw archive changed'
    else:
        entries=[]
        for phase in (0,1,2,3,5,6):
            folders=sorted((ROOT/'raw').glob(f'cb_p{phase}_*'))
            assert all((p/'completed.json').exists() or (p/'failure.json').exists() for p in folders)
            target=archive_root/f'phase{phase}_raw_v1.tar.gz'
            assert not target.exists(),'Preserve a partial or existing archive. Review it before a new version.'
            members=[p for folder in folders for p in folder.rglob('*') if p.is_file()]
            members+=sorted((ROOT/'logs').glob(f'cb_p{phase}_*.log'))
            with tarfile.open(target,'w:gz',compresslevel=1) as archive:
                for file in members:
                    archive.add(file,arcname=str(file.relative_to(ROOT)),recursive=False)
            entries.append(dict(phase=phase,attempts=len(folders),members=len(members),
                raw_bytes=sum(p.stat().st_size for p in members),archive_bytes=target.stat().st_size,
                absolute_path=str(target),sha256=digest(target),uploaded=False))
        write(index_path,dict(archives=entries,scope='Local raw evidence archives. Not uploaded. Derived tables/code/reports are separately committed.'))
    starts={};ledger=[]
    for line in (ROOT/'run_manifest.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['status']=='RUNNING':
            starts[row['job']]=row
        elif row['status'] in ('COMPLETED','FAILED'):
            begin=starts[row['job']]
            ledger.append(dict(run_id=row['job'],GPU=row['GPU'],status=row['status'],start_wall_s=begin['wall_start'],
                end_wall_s=row['wall_end'],process_elapsed_s=row['wall_end']-begin['wall_start'],
                scope='Worker process including load, warm-up, measurement, and output. Not GPU kernel busy time.'))
    csv_file(ROOT/'time_compute_ledger.csv',ledger)
    files=[]
    for base in (ROOT,REPO/'poc_flashvep/ouro_budget_joint_cg'):
        for file in sorted(base.rglob('*')):
            if not file.is_file():
                continue
            parts=set(file.relative_to(base).parts)
            if parts&{'raw','logs','state','archives','__pycache__'} or file.name in ('artifact_hash_manifest.json','burner.log') or file.name.endswith('.tmp'):
                continue
            files.append(dict(path=str(file.relative_to(REPO)),bytes=file.stat().st_size,sha256=digest(file)))
    write(ROOT/'artifact_hash_manifest.json',dict(created_wall_s=time.time(),source_head_at_generation=
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),files=files,
        exclusion='Self, live resource state/logs, local raw files/archives, and bytecode. Raw archive hashes are published separately.'))
    print('Archived closed attempts; immutable artifact hashes',len(files))

if __name__=='__main__':
    main()
