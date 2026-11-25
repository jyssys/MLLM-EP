# AR Tensor 정리: 2026-11-25

사용자 요청으로 Raw Tensor와 해당 Tensor를 포함한 대형 로컬 백업을 제거했다.
새 아카이브 생성은 사용자 요청으로 취소했다. Tensor를 외부에 별도 보관하지 않았다.
기존 보고서·CSV·JSON·JSONL·로그·Figure·설정·Source는 원래 위치에 보존했다.
기존 공개 결과 압축본과 모든 원래 Hash Manifest도 보존했다.
과거 정합성 PASS는 당시 실측 기록이다. 삭제한 KV·Hidden Tensor를 현재 다시 비교할 수는 없다.
삭제한 Tensor와 대형 로컬 백업은 이번 정리로 복구할 수 없다. 필요하면 GPU 실험을 다시 실행해야 한다.
상세 삭제 기록: `/home/esjung/cleanup_records/ar_tensor_cleanup_1795594739629673163/direct_summary.json`.
