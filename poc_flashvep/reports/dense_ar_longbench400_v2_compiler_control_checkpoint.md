# LongBench400-v2: Compiler 및 Evidence Checkpoint

상태: `ACTIVE`. 이 문서는 최종 연구 판정이 아니다.

기준 Commit은 `5ebed239073b4fa3709361038cbec026d1c32b78`이다.
사용자 Checkout과 Index는 변경하지 않았다.
새 Evidence Namespace는 `dense_ar_longbench400_v2_20261122`다.

## 확인한 결과

선택적 D Partial Compiler는 Layer와 Hybrid 모두 N1/N8 Strict 검사를 통과했다.
증거 수준은 `CORRECTNESS_ONLY`다.
새 경로의 N32/N400 검사와 Clean C3 성능은 아직 미확정이다.
같은 Source의 D Eager 경로를 대조군으로 유지한다.

C0 Discovery의 80개 Cell은 모두 N1 측정을 완료했다.
Layer의 Point Win은 48개였다. Token의 Point Win은 32개였다.
Output8의 Layer Point Win은 16개 중 2개였다.
Output192의 Layer Point Win은 16개 중 15개였다.
이는 C0 Mode0 Token과의 비교다. Strong C3 Stock Token 우위를 의미하지 않는다.
N3 Held-out 측정은 계속 진행 중이다.

400개 원문 Dataset을 공식 Snapshot과 동일 Tokenizer로 재구성했다.
재구성 파일의 SHA256은 동결 Dataset과 일치했다.
공개 Roster는 원문 ID, 순서, Metadata 및 Hash를 보존한다.
전체 원문과 Tokenized Dataset은 로컬 Evidence에 그대로 보존한다.

Client Notification에서 535개 완료 실행의 Metric을 별도로 재계산했다.
산술 불일치는 없었다. 실패 실행 4개와 `NOT_RUN` 20개는 그대로 남겼다.
Teacher 등 Client Metric이 없는 실행 10개는 별도 분류했다.
이 수치는 해당 재계산 시점의 Snapshot이다.
산술 검사는 State, Quality, Profiler 또는 Decode Safety Gate를 대체하지 않는다.

## 아직 주장하지 않는 결과

- C3 Main400 및 Held-out192의 Serving 우위는 미확정이다.
- GPU-exclusive GEMM·Attention 비중은 `NOT_MEASURED`다.
- Decode-safe Executable Window는 검증하지 못했다.
- Active-decode Dynamic Insertion은 `NOT_RUN`이다.
- 작은 Stock Suffix의 과거 Strict 불일치는 반복에서 재현되지 않았다.
- 해당 Strict 불일치의 근본 원인은 `ROOT_CAUSE_NOT_IDENTIFIED`다.

## 다음 검사

1. 새 Policy의 N8/N32/N400 정합성을 확인한다.
2. 동일 Calibration96으로 Admission, Credit 및 Compiler를 튜닝한다.
3. Held-out 전에 각 Family의 고정 설정을 동결한다.
4. QPS2.0의 Main400과 Held-out192를 각각 독립 N3로 확인한다.
5. Strong Stock Token, NativeStrong 및 Common-policy Native를 모두 유지한다.
6. ITL p99와 TPOT p95의 두 Native-relative CI Gate를 각각 검사한다.

모든 Frozen SLO와 기존 실패 결과는 유지한다.
Finite-cohort Throughput을 Sustainable Capacity로 표현하지 않는다.
