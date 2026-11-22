# LongBench400-v2 진행 Checkpoint

이 파일은 진행 중인 연구의 Checkpoint다. 최종 성능 보고서가 아니다.
최대 연구 예산은 20시간이다. 종료 전 약 2시간은 검증과 자원 복원에 사용한다.

400개 Dataset과 Source Revision을 동결했다.
실제 후보 수와 부족한 Output Cell을 선정 전에 보고했다.
Dataset 계약은 `dense_ar_longbench400_v2_dataset_contract.md`에 저장했다.

새 P Compiler 경로는 별도 Cache와 Original Weight를 사용한다.
Layer와 Hybrid의 N1·N8 KV·Hidden·출력 검사를 통과했다.
Natural32·400의 Method별 정합성과 품질 검사는 계속 진행 중이다.
이 검사는 성능 개선 또는 Decode 보호의 증거가 아니다.

정상 Scheduler Preemption과 충돌한 Hidden 관측 버퍼를 확인했다.
정확한 Request ID와 Preemption 위치가 일치할 때만 관측 버퍼를 초기화하는 수정안을 만들었다.
GPU 재검증 전에는 이 수정안을 최종 통과로 표시하지 않는다.

이전 Dataset의 Stock Token Suffix 표를 상속한 설정 오류도 확인했다.
미실행 설정은 `NOT_RUN`으로 취소했다. 새 Dataset의 실제 길이와 Cut으로 표를 재생성했다.
Strong Stock Token의 새 N8·32·400 검사를 완료한 뒤 성능을 측정한다.

실험 Raw와 실패 로그는 `dense_ar_longbench400_v2_20261122`에 보존한다.
C0 Mode0 Token을 Strong C3 Stock Token의 대체 기준선으로 사용하지 않는다.
현재 단계에서는 Layer·Hybrid의 Continuous Serving 우위 또는 Dynamic 이득을 주장하지 않는다.
