# Primary 설정 변경 기록

## 사용자 지시

2026-11-16에 사용자가 Aggregated 기본 설정을 B8192/S1024로 지정했다.
이후 B512/S16의 추가 GPU 실험을 중단한다.
완료한 결과와 기존 그림은 삭제하지 않는다.
전환 당시 진행 중이던 B512/S16 실행은 취소했다.
부분 실행은 성능 평가에서 제외한다.
이는 Runtime Hang이나 수치 정합성 실패와 다른 운영상 취소다.

## 유지하는 조건

Qwen3-8B의 Frozen Revision과 Original BF16을 유지한다.
Exact Attention과 Full36 계산을 유지한다.
GPU0/1, KV 42 GiB, Prefix Cache 비활성을 유지한다.
Prompt Pool, Arrival Seed, Sampling과 SLO를 변경하지 않는다.
Native PD의 기존 P/D 설정도 유지한다.
B8192/S1024는 Aggregated 설정이다.
이를 Native P/D의 Sequence 상한 변경으로 해석하지 않는다.

## 변경한 실행 범위

Primary 비교는 Agg-DP2 B8192/S1024와 Native PD 1P1D다.
Input8000/Output200에서 QPS2–10을 유지한다.
각 조건은 N128과 독립 Arrival Seed 3개를 사용한다.
Primary 실행 수는 총 54회다.
이미 완료한 동일 구성은 재사용한다.
초기 81회 계획과 원래 Freeze 파일은 그대로 보존한다.
변경 기록은 별도의 v2 Manifest에 저장한다.

600초 관측은 B8192/S1024의 QPS2/6에 독립 Start 3개씩 수행한다.
Native는 QPS2/6에 Start 1개씩 수행한다.
긴 관측은 총 8회다.
Output16/64의 추가 비교도 Aggregated B8192/S1024를 사용한다.
각 Output에서 QPS2/6/10을 Native와 비교한다.
추가 비교는 총 12회다.
Output별 N32 정합성을 먼저 검사한다.

## 해석 범위

범위 변경 전에 B512/S16 Confirmation 11회를 완료했다.
그 값은 보충 자료로 보존한다.
수치가 불리하다는 이유로 삭제하거나 실패로 바꾸지 않는다.
기존 Discovery 그림도 덮어쓰지 않는다.
새 반복 그림은 B8192/S1024와 Native만 Primary로 표시한다.

설정은 원래 Confirmation 전에도 존재한 Frozen Arm이다.
다만 Primary 비교군의 축소는 일부 Confirmation 이후의 사용자 결정이다.
이 순서를 최종 보고서에 명시한다.
이 결과로 모든 튜닝된 Colocated 설정보다 우수하다고 주장하지 않는다.
결론은 사용자 지정 기본 설정의 부하 특성에 한정한다.
