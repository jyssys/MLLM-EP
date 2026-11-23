"""Archive immutable completed evidence without changing raw files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from .common import ROOT,write,digest,preserve

ZSTD='/home/esjung/anaconda3/bin/zstd'
PART_BYTES=45*1024**2

def compress(paths,target,base,split=True):
    assert not target.exists(),target
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix(target.suffix+'.pending')
    assert not temporary.exists(),temporary
    with temporary.open('wb') as output:
        process=subprocess.Popen([ZSTD,'-q','-T2','-6','-c'],stdin=subprocess.PIPE,stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|') as archive:
                for path in paths:archive.add(path,arcname=str(path.relative_to(base)),recursive=False)
        finally:
            process.stdin.close()
        assert process.wait()==0,'Compression failed'
    temporary.rename(target)
    subprocess.run([ZSTD,'-q','-t',str(target)],check=True)
    if not split or target.stat().st_size<PART_BYTES:
        return [dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,SHA256=digest(target))]
    parts=[]
    with target.open('rb') as source:
        index=0
        while data:=source.read(PART_BYTES):
            path=target.with_name(target.name+f'.part{index:03d}')
            assert not path.exists(),path
            with path.open('wb') as output:output.write(data)
            parts.append(dict(path=str(path.relative_to(ROOT)),bytes=len(data),SHA256=hashlib.sha256(data).hexdigest()))
            index+=1
    # Retain the full archive locally. Only the bounded parts need publication.
    return parts

def raw_snapshot():
    preserve();destination=ROOT/'publication/evidence';destination.mkdir(parents=True,exist_ok=True)
    done=0
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():continue
        marker=destination/'manifests'/(folder.name+'.json')
        if marker.exists():continue
        before=digest(receipt)
        files=[];hashes=[];local_states=[];derived=[];sources=[]
        for path in sorted(folder.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(folder)
            if '__pycache__' in relative.parts:
                derived.append(str(relative));continue
            if relative.parts[0]=='worker_source':
                # The exact worker ZIP and its original file hashes are retained.
                continue
            record=dict(path=str(relative),bytes=path.stat().st_size,SHA256=digest(path))
            if path.suffix=='.pt':local_states.append(record);continue
            if relative==Path('source.zip'):
                archive=destination/'sources'/(record['SHA256']+'.zip')
                archive.parent.mkdir(parents=True,exist_ok=True)
                if not archive.exists():shutil.copyfile(path,archive)
                assert digest(archive)==record['SHA256']
                sources.append(dict(record,published=str(archive.relative_to(ROOT))));continue
            files.append(path);hashes.append(record)
        assert digest(receipt)==before,'Run changed during evidence snapshot'
        parts=compress(files,destination/'runs'/(folder.name+'.tar.zst'),folder)
        write(marker,dict(time=time.time(),run=folder.name,dispatcher_SHA256=before,
            included=hashes,compressed_parts=parts,exact_worker_source_archives=sources,
            local_tensor_states=local_states,derived_cache_files_not_archived=derived,
            raw_directory_unchanged=True,
            scope='The archive preserves every non-tensor runtime log,configuration,request and timeline outside the reconstructed worker tree. '
                'The exact worker ZIP is deduplicated by SHA256. Tensor states remain at their original local paths with byte counts and SHA256. '
                'No raw file is removed. This is a publication subset,not an archive of all raw bytes.'))
        done+=1
        print('ARCHIVED_IMMUTABLE_RUN',folder.name,flush=True)
    preserve();print('NEW_EVIDENCE_ARCHIVES',done,flush=True)

def dataset_snapshot():
    preserve();target=ROOT/'local_dataset_archives/longbench400_v2_data.tar.zst'
    if target.exists():raise AssertionError('Dataset snapshot already exists')
    paths=sorted(p for p in (ROOT/'data').glob('*.json') if p.name!='combined_original_candidate_pool_v2.json')
    records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,SHA256=digest(p)) for p in paths]
    parts=compress(paths,target,ROOT)
    write(ROOT/'local_dataset_archives/dataset_archive_manifest_v2.json',dict(time=time.time(),files=records,parts=parts,
        scope='Lossless snapshot of frozen data artifacts. The original full candidate text pool remains local. '
            'Official source revisions,selection IDs and candidate natural outputs permit reconstruction. '
            'This archive does not replace dataset provenance or grant new rights to source texts. '
            'Keep this bulk-text archive local. Publish the identity/hash rosters and the reconstruction code instead.'))
    preserve();print('DATASET_ARCHIVE_VERIFIED',len(paths),flush=True)

def tensor_snapshot():
    preserve();destination=ROOT/'local_tensor_archives';destination.mkdir(parents=True,exist_ok=True)
    done=0
    for manifest in sorted((ROOT/'publication/evidence/manifests').glob('*.json')):
        record=json.loads(manifest.read_text());states=record['local_tensor_states']
        if not states:continue
        marker=destination/(record['run']+'.json')
        if marker.exists():continue
        base=ROOT/'raw'/record['run'];paths=[base/r['path'] for r in states]
        assert all(digest(p)==r['SHA256'] for p,r in zip(paths,states))
        archive=destination/(record['run']+'.tensors.tar.zst')
        files=compress(paths,archive,base,split=False)
        write(marker,dict(time=time.time(),run=record['run'],tensor_files=states,archive=files[0],
            zstd_integrity_test_passed=True,original_raw_files_retained=True,
            scope='Lossless local tensor archive. The original tensors remain unchanged. '
                'Publication contains this manifest,not the large tensor archive.'))
        done+=1;print('LOCAL_TENSOR_ARCHIVE_VERIFIED',record['run'],flush=True)
    preserve();print('NEW_LOCAL_TENSOR_ARCHIVES',done,flush=True)

def minimal_snapshot():
    """Publish structured replay evidence,not bulk prompts or captured tensors."""
    preserve();destination=ROOT/'publication/minimal_evidence';done=0
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not folder.is_dir() or not receipt.exists():continue
        marker=destination/'manifests'/(folder.name+'.json')
        if marker.exists():continue
        files=[]
        for path in sorted(folder.rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(folder)
            if relative.parts[0]=='worker_source' or '__pycache__' in relative.parts:continue
            if path.suffix not in ('.json','.jsonl','.csv'):continue
            if 'numerical_probe' in path.name:continue
            files.append(path)
        before=digest(receipt)
        records=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,SHA256=digest(p)) for p in files]
        parts=compress(files,destination/'runs'/(folder.name+'.tar.zst'),folder)
        assert digest(receipt)==before
        write(marker,dict(time=time.time(),run=folder.name,dispatcher_SHA256=before,files=records,compressed_parts=parts,
            scope='Structured runtime evidence for independent metric,queue,coverage and capture recomputation. '
                'No original dataset prompt/context archive,tensor capture,worker tree or unstructured log is included. '
                'Exact worker ZIPs are deduplicated in the full archive manifest. Full logs and tensors remain unchanged locally. '
                'The separate full archive manifest preserves omitted-file hashes. No failed or unrun receipt is promoted.'))
        done+=1
    preserve();print('NEW_MINIMAL_PUBLICATION_ARCHIVES',done,flush=True)

def derived_snapshot():
    """Preserve oversized derived tables in lossless bounded publication parts."""
    preserve();destination=ROOT/'publication/derived_tables';records=[]
    for directory,pattern in (('figures','*.csv'),('analysis','*.json')):
        for path in sorted((ROOT/directory).glob(pattern)):
            if path.stat().st_size<90*1024**2:continue
            marker=destination/(directory+'_'+path.name+'.manifest.json')
            if marker.exists():
                old=json.loads(marker.read_text());assert digest(path)==old['source_SHA256']
                records.append(old);continue
            before=digest(path)
            parts=compress([path],destination/(directory+'_'+path.name+'.tar.zst'),ROOT)
            assert digest(path)==before
            value=dict(time=time.time(),source=str(path.relative_to(ROOT)),source_bytes=path.stat().st_size,
                source_SHA256=before,compressed_parts=parts,original_retained=True,
                scope='Lossless archive of an oversized derived CSV or JSON table. '
                    'Concatenate ordered parts before decompression when the manifest lists several parts. '
                    'The original table remains unchanged. This archive contains no bulk input dataset or tensor capture.')
            write(marker,value);records.append(value)
    write(destination/'index_v2.json',dict(time=time.time(),tables=records,passed=True))
    preserve();print('OVERSIZED_DERIVED_TABLES_ARCHIVED',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('raw_snapshot','dataset_snapshot','tensor_snapshot','minimal_snapshot','derived_snapshot'))
    args=p.parse_args();globals()[args.action]()
