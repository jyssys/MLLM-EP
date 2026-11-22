# Closeout 시작 지연과 보호 Gate

2026-11-22 13:36 KST에 승인된 All8 Burner를 새로 시작했다.
Supervisor는2603862다. Script와 사용자 소유권은 확인했다.
첫 Closeout은60초의 사용률 확인 구간에서70% 기준을 통과하지 못했다.
따라서 성공 Receipt와 CAMPAIGN_CLOSED를 쓰지 않았다.

실제 Burner Log는 이후 Model 초기화·Autotuning·Graph Capture를 계속 기록했다.
일부 Engine은13:37:40 KST에 Generation을 시작했다.
13:38:09 KST의 NVML Snapshot에서는 GPU0–7 모두100%였다.
이는 Burner 시작 지연이다. 연구의 Kernel Busy 또는 Safe Window 측정값은 아니다.

검증 대기시간을300초로 늘렸다.70% 기준은 바꾸지 않았다.
이미 준비된 All8을 다시 종료하지 않고 같은 Fresh Supervisor의 검증만 재개한다.
승인된 Script, 실제 소유권, 최근 GPU 작업 이후 생성된 Supervisor와 모든 GPU를 확인한다.

Archive V1 Controller2606153은 Closeout 성공 전에 잘못 실행됐다.
기존 CAMPAIGN_CLOSED Assertion이 이를 차단했다.
Archive 폴더나 데이터는 생성되지 않았다. 실패 Log는 삭제하지 않았다.
Closeout Receipt를 확인한 뒤 새 Controller Label로 동일 Archive 작업을 시작한다.

실패 Log: `logs/background_final_lossless_archive_v1.log`.
원래 Burner Log: `burner.log`. 최종 성공 시 그 Snapshot을 별도 보존한다.
