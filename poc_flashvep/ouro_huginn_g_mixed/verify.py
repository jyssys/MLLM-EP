"""Validate completed files, portable archives, depth ledgers, and preserved index."""
import csv
import hashlib
import json
import subprocess
import tarfile
from .common import ROOT,REPO,digest,write

def main():
    successful=[];failed=[]
    for raw in sorted((ROOT/'raw').iterdir()):
        if (raw/'completed.json').exists():
            record=json.loads((raw/'completed.json').read_text())
            for row in record['artifacts']:
                if row['path']=='telemetry.jsonl':continue
                assert digest(raw/row['path'])==row['sha256'],(raw,row)
            successful.append(raw.name)
        elif (raw/'failure.json').exists():failed.append(raw.name)
        else:raise AssertionError(('Unfinished raw attempt',raw))
    counts=json.loads((ROOT/'analysis_counts.json').read_text())
    assert failed==['dp_ouro_w1_hybrid_s0_gpu0'],failed
    for stage in ['support','smoke','calibration','screen','audit','no_decode','confirm','dp_retry',
            'mechanism','full_state','dp_confirm','dp_warm_confirm','dp_warm_cross']:
        plan=json.loads((ROOT/'plans'/f'{stage}.json').read_text())
        groups=plan.get('groups',[dict(jobs=plan.get('jobs',[]))])
        for group in groups:
            for job in group['jobs']:assert job['id'] in successful,(stage,job['id'])
    assert counts['prompt_ledger_failures']==0 and counts['decode_ledger_failures']==0
    assert json.loads((ROOT/'dp_summary.json').read_text())['clusters']==30
    index_hash=hashlib.sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=REPO)).hexdigest()
    assert index_hash=='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',index_hash
    archive=json.loads((ROOT/'raw_archive_manifest.json').read_text())
    expected={r['path']:r for r in archive['files']};seen=set()
    for row in archive['archives']:
        path=ROOT/row['path'];assert digest(path)==row['sha256']
        with tarfile.open(path,'r:xz') as stream:
            for member in stream:
                if not member.isfile():continue
                assert member.name in expected and member.name not in seen
                content=stream.extractfile(member)
                h=hashlib.sha256()
                while part:=content.read(1024**2):h.update(part)
                assert h.hexdigest()==expected[member.name]['sha256']
                seen.add(member.name)
    assert seen==set(expected)
    write(ROOT/'completion_audit.json',dict(successful_worker_attempts=len(successful),
        preserved_failed_worker_attempts=failed,verified_portable_files=len(seen),
        verified_archives=len(archive['archives']),unchanged_unrelated_index=index_hash,
        full_depth_prompt_failures=0,full_depth_decode_failures=0,status='PASS'))
    print('VERIFIED',len(successful),'successful attempts',len(failed),'preserved failures',len(seen),'portable files')

if __name__=='__main__':main()
