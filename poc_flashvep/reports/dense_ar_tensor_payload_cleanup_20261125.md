# Dense AR Tensor 정리와 결과 보존

사용자 승인에 따라 지정한 세 Namespace의 대형 Tensor와 Tensor 백업을 삭제했다.
새 Tensor 백업과 외부 보관은 수행하지 않았다. 새로운 성능 실험도 실행하지 않았다.

## 용량 변화

`GB=10^9 Bytes`다. 수치는 `du -x -B1`의 할당량이다.

| Namespace | 정리 전 GB | 정리 후 GB |
|---|---:|---:|
| dense_ar_longbench400_2d_20261121 | 342.75 | 33.70 |
| dense_ar_longbench400_v2_20261122 | 236.76 | 28.48 |
| dense_ar_p_critical_path_20261123 | 113.41 | 14.42 |
| 합계 | 692.93 | 76.59 |

원래 세 폴더의 할당량은616.33GB 감소했다. 정리 안내 파일3개의 용량도 정리 후 값에 포함했다.
중단한 작업의 임시 압축 파일2개, 약1.51GB도 제거했다. 이 임시 파일은 원래 폴더 용량 감소에 포함하지 않았다.
공유 파일시스템의 전체 Available 변화는 다른 작업의 영향을 받을 수 있다.

## 제거한 내용

- 원본 Tensor236,370개, 약367.39GB.
- Tensor 전용 압축본195개, 약86.18GB.
- Tensor가 포함된 대형 로컬 전체 압축본2개, 약162.77GB.

처음 시도한 비-Tensor 분리 압축은 사용자 변경 지시로 취소했다.
이 작업의 부분 출력도 제거했다. 완료한 새 Tensor 아카이브는 없다.

## 보존 및 검증

기존 보고서·CSV·JSON·JSONL·요청별 측정값·로그·Figure·설정·Source는 원래 위치에 보존했다.
기존 공개 결과 압축본과 원래 Hash Manifest도 변경하지 않았다.
비-Tensor 기존 파일2,359,588개의 크기·수정 시각·Inode 등 Metadata가 정리 전과 같았다.

세 폴더의 analysis·figures·configs·state·data에 있는 파일의 정렬된 SHA256 출력도 다시 검사했다.
정리 전후 Aggregate SHA256은 모두 `0d0edbed767e2a0078979f0f224092a1baf2cf44901871c319232775fd8b1d92`다.
이는 원래 서버의 경로를 포함한 Aggregate다. 다른 서버의 경로 변경 검사용 Dataset Hash는 아니다.

혼합 압축본의 비-Tensor Member451,623개는 원래 경로에 모두 존재했다. 크기 차이도0개였다.
이 검사는 모든 과거 Archive Version을 새로 복원한 검사는 아니다.

MLLM CPU252개, 인수인계 CPU6개, 정리 안전성 CPU4개 검사를 통과했다.
사용자 Checkout·Index·동결 파일 보존 검사도 통과했다.
현재 MLLM 자료와 Runtime은 삭제 범위에서 제외했다. GPU0–7 Burner는 확인 시 모두100%였다.

## 복구 한계와 새 서버 안내

삭제한 KV·Hidden Tensor와 대형 Tensor 백업은 별도 보관하지 않았다.
이 Tensor로 정합성을 다시 비교하려면 GPU 실험을 재실행해야 한다.
과거 정합성 PASS와 성능 값은 당시 실측 기록으로 남는다. 현재 Tensor가 존재한다는 뜻은 아니다.
이 정리는 기존 과학적 판정이나 성능 값을 변경하지 않는다.

상세 목록 위치와 수치는 [정리 검증 기록](../mllm_epd_handoff/ar_result_retention_20261125.json)에 있다.
새 서버에서는 [짧은 설정 프롬프트](../mllm_epd_handoff/quick_setup_prompt_ko.md)를 사용한다.
