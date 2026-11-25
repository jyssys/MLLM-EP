"""Recheck published lossless evidence only after all serving work closes."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile

from .common import ROOT,REPO,digest,preserve,resources,write
from .resource_closeout import require_final_closeout

PRIMARY=('ec_NATIVE_C11_','base_C14_','token_CONFIRM_C37_','whole_DEFLECTION_C43R4_',
    'interference_PROBE_C40R4','suffix_QUANTA_C41R3','ec_INLINE_CAUSAL_C44','batch_COUNTER_C46')


def confined_path(repo,root,value):
    relative=Path(value)
    assert not relative.is_absolute() and '..' not in relative.parts
    actual=(repo/relative).resolve()
    assert root.resolve() in actual.parents and actual.is_file()
    return actual


def verify_raw_archive(record,repo=REPO,root=ROOT):
    source=confined_path(repo,root,record['source'])
    compressed=confined_path(repo,root,record['compressed_path'])
    assert source.stat().st_size==record['source_bytes'] and digest(source)==record['source_SHA256']
    assert compressed.stat().st_size==record['compressed_bytes'] and digest(compressed)==record['compressed_SHA256']
    checksum=hashlib.sha256();size=0
    with gzip.open(compressed,'rb') as stream:
        while block:=stream.read(4*1024**2):checksum.update(block);size+=len(block)
    assert size==record['source_bytes'] and checksum.hexdigest()==record['source_SHA256']
    return dict(source=record['source'],compressed_path=record['compressed_path'],source_bytes=size,
        source_SHA256=checksum.hexdigest(),compressed_SHA256=record['compressed_SHA256'],status='LOSSLESS_VERIFIED')


def verify_source_archive(path,expected_sha):
    assert path.is_file() and digest(path)==expected_sha
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('source_manifest.json'));hashes={}
        assert len({r['member'] for r in manifest})==len(manifest)
        for item in manifest:
            member=Path(item['member'])
            assert not member.is_absolute() and '..' not in member.parts
            payload=archive.read(item['member']);sha=hashlib.sha256(payload).hexdigest()
            assert sha==item['SHA256']
            if 'bytes' in item:assert len(payload)==item['bytes']
            if str(member.parent)=='poc_flashvep/mllm_epd_prefill_heavy_token_split':hashes[member.name]=sha
    return dict(files=len(manifest),campaign_member_SHA256=hashes,archive_SHA256=expected_sha)


def tensor_receipts(root):
    """Find earlier tensor-file checksums without accepting ephemeral EC paths."""
    found={};raw=(root/'raw').resolve()
    def walk(value,manifest):
        if isinstance(value,dict):
            for name,checksum in [('tensor_file','tensor_SHA256'),('file','SHA256'),('path','SHA256')]:
                path,sha=value.get(name),value.get(checksum)
                if not isinstance(path,str) or not isinstance(sha,str) or Path(path).suffix!='.safetensors':continue
                target=Path(path).resolve()
                if raw in target.parents:
                    found.setdefault(target,[]).append(dict(manifest=str(manifest.relative_to(root)),SHA256=sha))
            for item in value.values():walk(item,manifest)
        elif isinstance(value,list):
            for item in value:walk(item,manifest)
    for path in raw.rglob('*.json'):walk(json.loads(path.read_text()),path)
    return found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);args=parser.parse_args()
    assert args.version.startswith('v') and args.version[1:].isdigit();preserve()
    output=ROOT/'analysis'/f'archive_integrity_closeout_{args.version}.json';assert not output.exists()
    require_final_closeout()
    owner=resources();burners=owner.burners();processes=owner.processes()
    assert set(burners)==set(range(8)),'Run the expensive evidence audit after final burner restoration.'
    for row in processes:
        proc=Path('/proc')/str(row['pid'])
        assert proc.stat().st_uid==os.getuid() and os.getsid(row['pid'])==burners[row['gpu']]
    assert {r['gpu'] for r in processes}==set(range(8))
    leases=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        row=json.loads(path.read_text())
        assert row.get('finished') and not Path('/proc',str(row['guardian'])).exists(),path
        leases.append((path,row))
    raw=[];source=[];unavailable=[];errors=[];receipt_rows=[];seen_archives={}
    for receipt in sorted((ROOT/'archives').glob('checkpoint_raw_records_v*.json')):
        record=json.loads(receipt.read_text());assert record['passed']
        receipt_rows.append(dict(path=str(receipt.relative_to(REPO)),SHA256=digest(receipt),files=len(record['files'])))
        for item in record['files']:
            try:raw.append(dict(receipt=receipt.name,**verify_raw_archive(item)))
            except Exception as error:
                errors.append(dict(kind='RAW_ARCHIVE',receipt=receipt.name,source=item.get('source'),
                    error=type(error).__name__+': '+str(error)))
    for path,lease in leases:
        primary=path.name.removeprefix('server_').startswith(PRIMARY)
        if 'source_archive' not in lease:
            unavailable.append(dict(lease=path.name,primary=primary,source_hashes_recorded=bool(lease.get('source_SHA256')),
                status='SOURCE_SNAPSHOT_NOT_AVAILABLE',scope='Do not reconstruct historical boot bytes from the current source.'))
            if primary:errors.append(dict(kind='PRIMARY_SOURCE_MISSING',lease=path.name))
            continue
        try:
            archive=Path(lease['source_archive']);assert ROOT in archive.resolve().parents
            if archive not in seen_archives:
                seen_archives[archive]=verify_source_archive(archive,lease['source_archive_SHA256'])
            checked=seen_archives[archive];assert checked['archive_SHA256']==lease['source_archive_SHA256']
            missing=[name for name,sha in lease['source_SHA256'].items() if checked['campaign_member_SHA256'].get(name)!=sha]
            source.append(dict(lease=path.name,primary=primary,archive=str(archive.relative_to(REPO)),
                archive_SHA256=checked['archive_SHA256'],files=checked['files'],unmatched_recorded_members=missing,
                status='EXACT_RECORDED_BOOT_SOURCE_VERIFIED' if not missing else 'ARCHIVE_VALID_BOOT_HASH_RECORD_DIFFERS'))
            if missing:errors.append(dict(kind='BOOT_HASH_RECORD_DIFFERS',lease=path.name,members=missing))
        except Exception as error:
            errors.append(dict(kind='SOURCE_ARCHIVE',lease=path.name,error=type(error).__name__+': '+str(error)))
    receipts=tensor_receipts(ROOT);tensors=[]
    for path in sorted((ROOT/'raw').rglob('*.safetensors')):
        actual=digest(path);expected=receipts.get(path.resolve(),[]);hashes={r['SHA256'] for r in expected}
        passed=not hashes or hashes=={actual}
        tensors.append(dict(file=str(path.relative_to(REPO)),bytes=path.stat().st_size,SHA256=actual,
            earlier_references=expected,historical_checksum_verified=bool(hashes) and passed,
            status='HISTORICAL_CHECKSUM_VERIFIED' if hashes and passed else
                'CHECKSUM_RECORDED_NOW_NO_EARLIER_REFERENCE' if not hashes else 'HISTORICAL_CHECKSUM_MISMATCH',
            scope='Original local tensor bytes. No tensor copy, deletion or claim that a newly recorded checksum reconstructs historical contents.'))
        if not passed:errors.append(dict(kind='TENSOR_CHECKSUM_MISMATCH',file=str(path),expected=sorted(hashes),actual=actual))
    assert receipt_rows and raw and source and tensors
    write(output,dict(time=time.time(),status='INTEGRITY_PASS' if not errors else 'INTEGRITY_FAILURE',
        passed=not errors,receipts=receipt_rows,raw_archives=raw,source_archives=source,
        historical_missing_snapshots=unavailable,errors=errors,unique_source_archives=len(seen_archives),
        original_tensor_files=tensors,original_tensor_bytes=sum(r['bytes'] for r in tensors),
        actual_all_GPU_burners_verified=burners,
        scope='Original local tensor bytes and earlier available tensor receipts, published gzip round trips, source ZIP members and recorded boot hashes only. '
            'Historical missing snapshots remain unavailable. This does not establish task quality, runtime GO or complete scientific gates.'))
    preserve();assert not errors,'Preserved integrity failures require inspection. Do not overwrite or repair original evidence automatically.'
    print('ARCHIVE_CLOSEOUT_INTEGRITY_PASS',len(raw),len(source),len(unavailable),flush=True)


if __name__=='__main__':main()
