"""Preserve append-only history and disclose legacy boot metadata differences."""
import json
from pathlib import Path
import time

from .archive_integrity_closeout import verify_raw_archive,verify_source_archive,verify_executed_source
from .common import ROOT,REPO,digest,preserve,write
from .publication_checkpoint_v4 import compress
from .resource_closeout import require_final_closeout


def main():
    preserve();require_final_closeout()
    source=ROOT/'analysis/archive_integrity_closeout_v0.json';initial=json.loads(source.read_text())
    assert not initial['passed'] and len(initial['errors'])==7
    report=ROOT.parent/'mllm_epd_final_provenance_discrepancies_v1.md';assert not report.exists()
    target=ROOT/'archives/public_checkpoint_v14';receipt=ROOT/'archives/checkpoint_raw_records_v14.json'
    assert not target.exists() and not receipt.exists();target.mkdir()
    old=json.loads((ROOT/'archives/checkpoint_raw_records_v0.json').read_text())['files']
    append=[];records=[];boots=[]
    ledger=ROOT/'resource_ledger.jsonl';starts=[json.loads(line) for line in ledger.read_text().splitlines()]
    for error in initial['errors']:
        if error['kind']=='RAW_ARCHIVE':
            record=next(r for r in old if r['source']==error['source'])
            proof=verify_raw_archive(record,allow_append=True)
            assert proof['status']=='HISTORICAL_APPEND_PREFIX_VERIFIED'
            append.append(proof);records.append(compress(REPO/error['source'],target))
            continue
        assert error['kind']=='BOOT_HASH_RECORD_DIFFERS'
        lease_path=ROOT/'state'/error['lease'];lease=json.loads(lease_path.read_text())
        archive=verify_source_archive(Path(lease['source_archive']),lease['source_archive_SHA256'])
        actual=verify_executed_source(lease,archive['manifest'])
        matching=[r for r in starts if r.get('action')=='START_OWNED_SERVER' and r['name']==lease['name']
            and all(r[k]==lease[k] for k in ['pid','command','source_archive_SHA256','executed_source_cwd','source_SHA256'])]
        assert len(matching)==1 and lease['isolated_prefix_probe_enabled'] is False
        boots.append(dict(lease=error['lease'],lease_SHA256=digest(lease_path),
            archive_SHA256=lease['source_archive_SHA256'],source_materialization=actual,
            disagreeing_members=[dict(name=name,recorded=lease['source_SHA256'][name],
                immutable_archive=archive['campaign_member_SHA256'][name]) for name in error['members']],
            startup_ledger_SHA256=digest(ledger),original_lease_or_archive_changed=False,
            cause_of_metadata_difference='NOT_ESTABLISHED',isolated_prefix_probe_enabled=False))
    assert len(append)==4 and len(boots)==3
    assert not [e for e in initial['errors'] if e['kind']=='TENSOR_CHECKSUM_MISMATCH']
    write(receipt,dict(time=time.time(),files=records,passed=True,
        scope='Complete final versions of four historical append-only JSONL logs. Earlier compressed prefixes remain byte-identical and retain their original receipts. No old raw or metadata is rewritten.'))
    audit=ROOT/'analysis/final_provenance_discrepancies_v0.json'
    write(audit,dict(time=time.time(),initial_integrity_failure_SHA256=digest(source),
        historical_append_only_logs=append,complete_current_log_receipt_SHA256=digest(receipt),
        legacy_boot_metadata_disagreements=boots,
        original_tensor_files=len(initial['original_tensor_files']),original_tensor_bytes=initial['original_tensor_bytes'],
        tensor_checksum_mismatch_count=0,historical_missing_source_snapshots=len(initial['historical_missing_snapshots']),
        original_failed_audit_preserved=True,original_metadata_rewritten=False,
        future_metadata_fix='New boot source_SHA256 is derived from the verified archive. This does not repair old leases or rerun old GPU results.',
        scope='Evidence integrity and known provenance limits only. Not task quality, serving GO or proof of the exact metadata sampling race.'))
    lines=['# 최종 Provenance 검사: Append-only Log와 Boot Metadata 불일치','',
        '초기 최종 감사 v0는7개 항목에서 실패했다. 그 감사와 모든 원본을 보존했다.',
        '아래 재검사는 원본을 고치지 않았다. 역사적 Snapshot과 현재 전체 파일의 분모를 분리했다.',
        '과거 Source Metadata3건의 불일치는 남아 있다. 이를 원래 기록의 완전한 일치로 바꾸지 않는다.','',
        '## 역사적 Raw Snapshot4개','',
        '압축본의 Size·SHA256·Decompression Hash가 원래 Receipt와 일치했다.',
        '압축본 전체가 현재 파일의 같은 길이 Prefix와 Byte 단위로 일치했다.',
        'Prefix 종료는 완전한 JSONL Record 경계였다. 추가 Suffix의 모든 Record도 JSON으로 검사했다.',
        '따라서 이4건은 덮어쓴 Raw가 아니라, 원래 Snapshot 뒤에 Record가 추가된 Log다.',
        '변경된 Prefix·잘린 Log·허용 목록 밖의 Append는 새 검사에서도 실패한다.','',
        '| File | Historical Bytes | Final Bytes | Added JSONL Records |','|---|---:|---:|---:|']
    for row in append:
        lines.append(f'| {Path(row["source"]).name} | {row["source_bytes"]} | {row["current_source_bytes"]} | {row["appended_JSONL_records"]} |')
    lines+=['','현재 전체4개 Log는 별도 v14 압축본과 SHA로 보존했다. 기존 v0 압축본·Receipt는 수정하지 않았다.',
        '최초 검사는 모든 역사적 Snapshot을 현재 전체 파일과 같아야 한다고 검사했다.',
        '후속 검사는 역사적 Prefix의 완전성 및 현재 전체 Snapshot의 별도 보존을 모두 요구한다.','',
        '## Legacy Source Map3건','',
        '대상은 C37 B0 Idle10의 E4/P5/D6 Lease다. 각 source_SHA256 Map은 Boot ZIP과3개 파일에서 다르다.',
        '파일은 servers.py·prefix_probe_contract.py·test_prefix_probe.py다.',
        '각 Boot ZIP의 원래 Hash·CRC·모든 Member Hash는 유효했다.',
        '실행용 Materialized CWD의 모든 파일도 그 ZIP과 일치했다. 시작 Ledger의 PID·Command·CWD·ZIP Hash도 Lease와 일치했다.',
        '해당 실행의 Isolated Prefix Probe Flag는 False였다. 두 버전의 차이는 해당 Probe Opt-in 경로와 그 테스트에 있다.',
        '그러나 정확히 언제·왜 Legacy Map이 다른 바이트를 기록했는지는 확정하지 못했다.',
        '사후 Worktree Hash를 실제 실행 Source라고 대체하지 않았다. 기존 Map·Lease·ZIP을 수정하지 않았다.',
        '이3건의 Source 근거는 원래 ZIP·Member Manifest·실행 CWD·시작 Ledger와 불일치 경고를 함께 사용해야 한다.',
        '이를 모든 Legacy Map이 Byte-exact였다는 증거로 사용하지 않는다.','',
        '| Lease | Immutable Archive SHA256 | Materialized Files Match |', '|---|---|---|']
    for row in boots:
        lines.append(f'| {row["lease"]} | {row["archive_SHA256"]} | {row["source_materialization"]["all_materialized_members_match"]} |')
    lines+=['','향후 Guardian은 검증된 Archive Member에서 source_SHA256 Map을 만든다.',
        '변경 가능한 Host Directory를 나중에 다시 읽지 않는다. 이 Harness 수정은 새 CPU 회귀 테스트로 검사했다.',
        '추가 GPU 실험은 실행하지 않았다. 기존 성능·Source 기록을 이 수정으로 소급 변경하지 않았다.','',
        '## 원본 Tensor와 미확보 Source','',
        f'초기 감사는 원본 Tensor{len(initial["original_tensor_files"])}개·{initial["original_tensor_bytes"]:,}bytes를 검사했다.',
        '기존 Tensor Checksum과의 불일치는0건이었다. 이전 Receipt가 없는 파일의 새 Hash는 역사적 복구 증거가 아니다.',
        f'초기 실행의 Source ZIP 미확보{len(initial["historical_missing_snapshots"])}건도 그대로 남겼다.',
        '현재 Source로 역사적 누락을 복원한 것처럼 표시하지 않았다.',
        '후속 최종 감사는 모든 Tensor·Raw·Archive 검사를 다시 실행한다. 이 문서는 그 결과를 미리 PASS로 선언하지 않는다.',
        'Task Quality·Prefill-heavy·Safe Headroom의 HOLD는 이 Byte 검사와 독립적이다.','',
        '증거는 analysis/final_provenance_discrepancies_v0.json, archives/checkpoint_raw_records_v14.json에 있다.',
        '초기 실패는 analysis/archive_integrity_closeout_v0.json과 원래 Log에 있다.','']
    report.write_text('\n'.join(lines));preserve();print('HISTORICAL_LOG_AND_BOOT_DISCREPANCIES_DISCLOSED',flush=True)


if __name__=='__main__':main()
