# 최종 Provenance 검사: Append-only Log와 Boot Metadata 불일치

초기 최종 감사 v0는7개 항목에서 실패했다. 그 감사와 모든 원본을 보존했다.
아래 재검사는 원본을 고치지 않았다. 역사적 Snapshot과 현재 전체 파일의 분모를 분리했다.
과거 Source Metadata3건의 불일치는 남아 있다. 이를 원래 기록의 완전한 일치로 바꾸지 않는다.

## 역사적 Raw Snapshot4개

압축본의 Size·SHA256·Decompression Hash가 원래 Receipt와 일치했다.
압축본 전체가 현재 파일의 같은 길이 Prefix와 Byte 단위로 일치했다.
Prefix 종료는 완전한 JSONL Record 경계였다. 추가 Suffix의 모든 Record도 JSON으로 검사했다.
따라서 이4건은 덮어쓴 Raw가 아니라, 원래 Snapshot 뒤에 Record가 추가된 Log다.
변경된 Prefix·잘린 Log·허용 목록 밖의 Append는 새 검사에서도 실패한다.

| File | Historical Bytes | Final Bytes | Added JSONL Records |
|---|---:|---:|---:|
| processor_handoff_decode_gpu6.jsonl | 1403026 | 20664481 | 30011 |
| processor_handoff_encoder_gpu4.jsonl | 1396520 | 21922681 | 32152 |
| processor_handoff_prefill_gpu5.jsonl | 1405173 | 20467436 | 29544 |
| proxy_metadata_B1_grid_only.jsonl | 367616004 | 367696101 | 1 |

현재 전체4개 Log는 별도 v14 압축본과 SHA로 보존했다. 기존 v0 압축본·Receipt는 수정하지 않았다.
최초 검사는 모든 역사적 Snapshot을 현재 전체 파일과 같아야 한다고 검사했다.
후속 검사는 역사적 Prefix의 완전성 및 현재 전체 Snapshot의 별도 보존을 모두 요구한다.

## Legacy Source Map3건

대상은 C37 B0 Idle10의 E4/P5/D6 Lease다. 각 source_SHA256 Map은 Boot ZIP과3개 파일에서 다르다.
파일은 servers.py·prefix_probe_contract.py·test_prefix_probe.py다.
각 Boot ZIP의 원래 Hash·CRC·모든 Member Hash는 유효했다.
실행용 Materialized CWD의 모든 파일도 그 ZIP과 일치했다. 시작 Ledger의 PID·Command·CWD·ZIP Hash도 Lease와 일치했다.
해당 실행의 Isolated Prefix Probe Flag는 False였다. 두 버전의 차이는 해당 Probe Opt-in 경로와 그 테스트에 있다.
그러나 정확히 언제·왜 Legacy Map이 다른 바이트를 기록했는지는 확정하지 못했다.
사후 Worktree Hash를 실제 실행 Source라고 대체하지 않았다. 기존 Map·Lease·ZIP을 수정하지 않았다.
이3건의 Source 근거는 원래 ZIP·Member Manifest·실행 CWD·시작 Ledger와 불일치 경고를 함께 사용해야 한다.
이를 모든 Legacy Map이 Byte-exact였다는 증거로 사용하지 않는다.

| Lease | Immutable Archive SHA256 | Materialized Files Match |
|---|---|---|
| server_token_CONFIRM_C37_B0_Idle10_D6.json | 1aef6c2d3d2cca13fb462646a916062e122a094f87ccfa4d1a0510bbbedb0f11 | True |
| server_token_CONFIRM_C37_B0_Idle10_E4.json | 1aef6c2d3d2cca13fb462646a916062e122a094f87ccfa4d1a0510bbbedb0f11 | True |
| server_token_CONFIRM_C37_B0_Idle10_P5.json | 1aef6c2d3d2cca13fb462646a916062e122a094f87ccfa4d1a0510bbbedb0f11 | True |

향후 Guardian은 검증된 Archive Member에서 source_SHA256 Map을 만든다.
변경 가능한 Host Directory를 나중에 다시 읽지 않는다. 이 Harness 수정은 새 CPU 회귀 테스트로 검사했다.
추가 GPU 실험은 실행하지 않았다. 기존 성능·Source 기록을 이 수정으로 소급 변경하지 않았다.

## 원본 Tensor와 미확보 Source

초기 감사는 원본 Tensor1756개·250,822,942,944bytes를 검사했다.
기존 Tensor Checksum과의 불일치는0건이었다. 이전 Receipt가 없는 파일의 새 Hash는 역사적 복구 증거가 아니다.
초기 실행의 Source ZIP 미확보47건도 그대로 남겼다.
현재 Source로 역사적 누락을 복원한 것처럼 표시하지 않았다.
후속 최종 감사는 모든 Tensor·Raw·Archive 검사를 다시 실행한다. 이 문서는 그 결과를 미리 PASS로 선언하지 않는다.
Task Quality·Prefill-heavy·Safe Headroom의 HOLD는 이 Byte 검사와 독립적이다.

증거는 analysis/final_provenance_discrepancies_v0.json, archives/checkpoint_raw_records_v14.json에 있다.
초기 실패는 analysis/archive_integrity_closeout_v0.json과 원래 Log에 있다.
