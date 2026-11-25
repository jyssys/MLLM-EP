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
APPEND_ONLY_RAW={'processor_handoff_decode_gpu6.jsonl','processor_handoff_encoder_gpu4.jsonl',
    'processor_handoff_prefill_gpu5.jsonl','proxy_metadata_B1_grid_only.jsonl'}


def confined_path(repo,root,value):
    relative=Path(value)
    assert not relative.is_absolute() and '..' not in relative.parts
    actual=(repo/relative).resolve()
    assert root.resolve() in actual.parents and actual.is_file()
    return actual


def verify_raw_archive(record,repo=REPO,root=ROOT,allow_append=False):
    source=confined_path(repo,root,record['source'])
    compressed=confined_path(repo,root,record['compressed_path'])
    assert compressed.stat().st_size==record['compressed_bytes'] and digest(compressed)==record['compressed_SHA256']
    checksum=hashlib.sha256();size=0
    with gzip.open(compressed,'rb') as stream:
        while block:=stream.read(4*1024**2):checksum.update(block);size+=len(block)
    assert size==record['source_bytes'] and checksum.hexdigest()==record['source_SHA256']
    current_size=source.stat().st_size;current_sha=digest(source)
    status='LOSSLESS_VERIFIED';extra={}
    if current_size!=size or current_sha!=record['source_SHA256']:
        assert allow_append and source.name in APPEND_ONLY_RAW and current_size>size
        prefix=hashlib.sha256();remaining=size
        with source.open('rb') as stream:
            while remaining:
                block=stream.read(min(4*1024**2,remaining));assert block
                prefix.update(block);remaining-=len(block)
            assert prefix.hexdigest()==record['source_SHA256'],'Historical prefix changed, not append-only growth'
            stream.seek(size-1);assert stream.read(1)==b'\n','Historical JSONL record boundary is incomplete'
            appended=0
            for line in stream:
                assert line.endswith(b'\n') and isinstance(json.loads(line),dict);appended+=1
            assert appended>0
        status='HISTORICAL_APPEND_PREFIX_VERIFIED'
        extra=dict(current_source_bytes=current_size,current_source_SHA256=current_sha,appended_JSONL_records=appended,
            scope='The historical compressed snapshot is an exact unchanged prefix. Full current log needs its own final archive receipt. No old receipt is rewritten.')
    return dict(source=record['source'],compressed_path=record['compressed_path'],source_bytes=size,
        source_SHA256=checksum.hexdigest(),compressed_SHA256=record['compressed_SHA256'],status=status,**extra)


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
    return dict(files=len(manifest),campaign_member_SHA256=hashes,archive_SHA256=expected_sha,manifest=manifest)


def verify_executed_source(lease,manifest,root=ROOT):
    folder=Path(lease['executed_source_cwd']).resolve()
    assert root.resolve()/'runtime_sources' in folder.parents
    assert folder.name.startswith(lease['source_archive_SHA256'][:16]+'_')
    record=folder/'verified_source_manifest.json';value=json.loads(record.read_text())
    assert value['archive_SHA256']==lease['source_archive_SHA256'] and value['files']==manifest
    for row in manifest:
        path=(folder/row['member']).resolve();assert folder in path.parents
        assert path.stat().st_size==row['bytes'] and digest(path)==row['SHA256']
    return dict(folder=str(folder),verified_manifest_SHA256=digest(record),all_materialized_members_match=True,
        files=len(manifest),scope='Independent immutable boot materialization bytes. This does not rewrite a disagreeing legacy source_SHA256 map.')


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
    raw=[];source=[];unavailable=[];errors=[];receipt_rows=[];seen_archives={};warnings=[]
    ledger_path=ROOT/'resource_ledger.jsonl'
    starts=[json.loads(line) for line in ledger_path.read_text().splitlines()
        if json.loads(line).get('action')=='START_OWNED_SERVER']
    for receipt in sorted((ROOT/'archives').glob('checkpoint_raw_records_v*.json')):
        record=json.loads(receipt.read_text());assert record['passed']
        receipt_rows.append(dict(path=str(receipt.relative_to(REPO)),SHA256=digest(receipt),files=len(record['files'])))
        for item in record['files']:
            try:raw.append(dict(receipt=receipt.name,**verify_raw_archive(item,allow_append=True)))
            except Exception as error:
                errors.append(dict(kind='RAW_ARCHIVE',receipt=receipt.name,source=item.get('source'),
                    error=type(error).__name__+': '+str(error)))
    for old in [row for row in raw if row['status']=='HISTORICAL_APPEND_PREFIX_VERIFIED']:
        full=[row for row in raw if row['source']==old['source'] and row['status']=='LOSSLESS_VERIFIED'
            and row['source_SHA256']==old['current_source_SHA256'] and row['source_bytes']==old['current_source_bytes']]
        if not full:errors.append(dict(kind='APPENDED_FULL_LOG_NOT_ARCHIVED',source=old['source']))
        else:warnings.append(dict(kind='HISTORICAL_APPEND_PREFIX',source=old['source'],historical_receipt=old['receipt'],
            complete_current_receipts=[row['receipt'] for row in full],old_receipt_changed=False))
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
            materialized=None
            if missing:
                materialized=verify_executed_source(lease,checked['manifest'])
                matches=[row for row in starts if row['name']==lease['name'] and row['pid']==lease['pid']
                    and row['source_archive_SHA256']==lease['source_archive_SHA256']
                    and row['executed_source_cwd']==lease['executed_source_cwd'] and row['command']==lease['command']
                    and row['source_SHA256']==lease['source_SHA256']]
                assert len(matches)==1,'Independent original startup ledger cannot confirm the recorded boot materialization'
                warnings.append(dict(kind='LEGACY_BOOT_METADATA_DIFFERS',lease=path.name,members=missing,
                    immutable_archive_and_materialized_source_verified=materialized,
                    startup_ledger_SHA256=digest(ledger_path),original_metadata_not_rewritten=True,
                    exact_metadata_sampling_cause='NOT_ESTABLISHED',isolated_prefix_probe_enabled=lease.get('isolated_prefix_probe_enabled')))
            source.append(dict(lease=path.name,primary=primary,archive=str(archive.relative_to(REPO)),
                archive_SHA256=checked['archive_SHA256'],files=checked['files'],unmatched_recorded_members=missing,
                status='EXACT_RECORDED_BOOT_SOURCE_VERIFIED' if not missing else 'IMMUTABLE_BOOT_VERIFIED_LEGACY_METADATA_STILL_DIFFERS',
                materialized_source_proof=materialized))
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
    write(output,dict(time=time.time(),status=('INTEGRITY_PASS_WITH_DISCLOSED_PROVENANCE_WARNINGS' if warnings else 'INTEGRITY_PASS') if not errors else 'INTEGRITY_FAILURE',
        passed=not errors,receipts=receipt_rows,raw_archives=raw,source_archives=source,
        historical_missing_snapshots=unavailable,errors=errors,warnings=warnings,unique_source_archives=len(seen_archives),
        original_tensor_files=tensors,original_tensor_bytes=sum(r['bytes'] for r in tensors),
        actual_all_GPU_burners_verified=burners,
        scope='Original local tensor bytes, historical gzip snapshots and complete current append-only logs, immutable source ZIP members and available startup metadata. '
            'Legacy metadata disagreement and historical missing snapshots remain explicit. Materialized source proof does not repair the original source_SHA256 map. '
            'This does not establish task quality, runtime GO or complete scientific gates.'))
    preserve();assert not errors,'Preserved integrity failures require inspection. Do not overwrite or repair original evidence automatically.'
    print('ARCHIVE_CLOSEOUT_INTEGRITY_PASS',len(raw),len(source),len(unavailable),flush=True)


if __name__=='__main__':main()
