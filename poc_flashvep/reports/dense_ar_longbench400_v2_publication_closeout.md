# LongBench400-v2 게시 및 자원 종료 검증

## 과학 결과 Snapshot

Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.
결과 Snapshot Commit: `16404c46ecd219ef42794d0f71221d0c6e604d99`.
해당 Commit의 Push와 Remote SHA 일치를 확인했다.
이 문서와 게시 Receipt는 별도의 후속 Commit에 추가한다.

최종 과학적 판정은 `NO-GO_ON_TESTED_GRID`다.
C0의 Output 효과와 C3의 강한 Stock Token 비교는 별도 결과다.
모든 Method의 전체 QPS Grid를 완료했다는 뜻은 아니다.
미실행 조건은 Configuration Inventory에 남겼다.

[최종 연구 보고서](dense_ar_longbench400_v2_output_aware_2d_final_summary.md).
[Snapshot 게시 Receipt](dense_ar_longbench400_v2_20261122/state/publication_final_v2.json).

## Evidence 검증

CPU 회귀 검사 19개를 통과했다.
Client Metric 실행 1303개의 원시 알림을 다시 계산했다.
Metric이 없는 Teacher/Diagnostic 14개는 별도로 기록했다.
실패한 실행과 정합성 Gate를 삭제하거나 성공으로 변경하지 않았다.

무손실 Archive 2882개를 실제로 압축 해제하며 검사했다.
검사한 파일 수는 375318개다.
각 파일의 Byte 수와 SHA256이 원본 Manifest와 일치했다.
검사한 압축 전 크기는 약142.277GiB다.

Full Raw Log, Tensor 및 원문 Dataset Archive는 로컬에 남겼다.
GitHub에는 Structured Evidence, 정확한 Source Capsule 및 파생 Table을 게시했다.
큰 파생 Table은 무손실 Archive와 SHA256 Manifest를 사용한다.
원문 Dataset은 공식 Source와 공개 ID/Hash Roster로 재구성한다.

[Archive 검사](dense_ar_longbench400_v2_20261122/analysis/lossless_archive_integrity_v2.json).
[파생 Table Archive](dense_ar_longbench400_v2_20261122/publication/derived_tables/index_v2.json).

## 게시 경로의 지연 수정

기존 게시 코드는 파일마다 Private Index를 다시 썼다.
소유한 CPU Publisher만 Ref 갱신 전에 종료했다.
GPU 실행, Raw Evidence 및 사용자 Index는 바꾸지 않았다.

같은 세 파일과 같은 Parent Tree로 작은 진단을 실행했다.
Serial Index 갱신은 3.288초였다.
Hash 경로를 유지한 일괄 Index 갱신은 1.211초였다.
Hash와 Index를 모두 일괄 처리한 경로는 1.251초였다.
세 경로의 Git Tree는 같았다.

이 값은 순서가 고정된 단일 운영 진단이다.
독립 반복 CI 또는 Serving 성능 이득이 아니다.
전체 게시 경로도 새 방식으로 실행했고 Remote SHA를 확인했다.

[Private Index 진단](dense_ar_longbench400_v2_20261122/analysis/publication_private_index_timing_v2.json).

## GPU 및 사용자 작업 보존

신규 GPU Dispatch는 시작 후 18시간에 종료했다.
이미 실행 중인 소유 작업은 정상 종료까지 기다렸다.
기존 사용자 Checkout과 Staged Index의 서명이 유지됐다.
다른 사용자 GPU 작업을 종료하지 않았다.

승인된 `run_utilize.sh`로 GPU0–7의 Burner를 복원했다.
각 GPU의 실제 프로세스 소유권과 사용률을 확인했다.
Snapshot Push 이후에도 Burner 실행을 확인했다.
이 사용률은 운영 확인이다. 연구의 Kernel Busy 계측이 아니다.

[GPU 종료 검증](dense_ar_longbench400_v2_20261122/state/resource_closeout_v2.json).
