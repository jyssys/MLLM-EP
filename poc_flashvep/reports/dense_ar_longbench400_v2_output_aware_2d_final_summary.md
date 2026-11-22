# LongBench400-v2 Output-aware 2D Serving

상태: ACTIVE_SNAPSHOT.
생성 시각: 2026-11-22T18:39:15.262927+00:00.
경과 연구 시간: 11.540시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 연구 판정

ACTIVE. 최종 판정 전이다.

## Dataset

선정 요청 수는 400개다. Input 구간별 요청 수는 [100, 100, 100, 100]다.
처음 세 Input 구간은 각 Output Cell에 25개를 확보했다.
20K–36K 구간의 긴 Output 후보는 부족했다. 원문을 수정하지 않고 동결된 재배분 규칙을 적용했다.

| Input | Output1–10 | Output11–50 | Output51–100 | Output101–200 |
| --- | --- | --- | --- | --- |
| 3K–6K | 25 | 25 | 25 | 25 |
| 6K–10K | 25 | 25 | 25 | 25 |
| 10K–20K | 25 | 25 | 25 | 25 |
| 20K–36K | 46 | 31 | 17 | 6 |

Dataset SHA256: `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`.
선정 시 Natural Output과 C3 Compiled C1 Output은 수치 경로가 다르다.
따라서 선정 Output 구간이 실제 C3의 모든 출력 길이를 보장하지 않는다. Dataset을 다시 선정하지 않았다.

## 최종 연구 질문

### Q1. 목표 4×4 Dataset 분포를 달성했는가?

최종 분석 전이다.

### Q2. 긴 Output에서 Layer의 Point Win이 증가하는가?

최종 분석 전이다.

### Q3. Input과 QPS를 통제한 Output 효과가 유지되는가?

최종 분석 전이다.

### Q4. C0 Crossover가 C3에서도 재현되는가?

최종 분석 전이다.

### Q5. Layer 또는 Hybrid가 Strong Stock Token의 TTFT를 개선하는가?

최종 분석 전이다.

### Q6. Decode ITL/TBT/TPOT를 보호하는가?

최종 분석 전이다.

### Q7. SLO-goodput이 증가하는가?

최종 분석 전이다.

### Q8. 실행 가능한 Decode-safe Window가 검증됐는가?

최종 분석 전이다.

### Q9. Dynamic 2D Selector가 Best Static보다 추가 이득을 만드는가?

최종 분석 전이다.

### Q10. 성공하지 못한 조건의 원인은 무엇인가?

최종 분석 전이다.

## Evidence 해석 제한

Point Winner, CI 확정 결과 및 Decode Safety를 별도로 표시한다.
Request나 Token 수를 독립 Server Start 수로 사용하지 않는다.
GPU Host Enclosure를 Operator-exclusive GPU Time으로 표현하지 않는다.
Static Cut/정책 탐색은 측정한 후보 집합에 한정된다. 전역 최적을 주장하지 않는다.
실패한 Gate와 미실행 조건은 삭제하지 않는다. 과거 v1 결과를 덮어쓰지 않는다.

## 상세 보고서

- [dense_ar_longbench400_v2_candidate_availability.md](dense_ar_longbench400_v2_candidate_availability.md)
- [dense_ar_longbench400_v2_dataset_contract.md](dense_ar_longbench400_v2_dataset_contract.md)
- [dense_ar_longbench400_v2_c0_output_crossover.md](dense_ar_longbench400_v2_c0_output_crossover.md)
- [dense_ar_longbench400_v2_c3_continuous_serving.md](dense_ar_longbench400_v2_c3_continuous_serving.md)
- [dense_ar_longbench400_v2_decode_safety_and_windows.md](dense_ar_longbench400_v2_decode_safety_and_windows.md)
- [dense_ar_longbench400_v2_output_prior_selector.md](dense_ar_longbench400_v2_output_prior_selector.md)
- [dense_ar_longbench400_v2_research_decision_log.md](dense_ar_longbench400_v2_research_decision_log.md)
- [dense_ar_longbench400_v2_figures.md](dense_ar_longbench400_v2_figures.md)

## 재현성 및 자원

공식 원문은 별도 라이선스를 따를 수 있다. Bulk Context를 새로 공개 배포하지 않는다.
공개 Identity/Hash Roster와 Reconstruction Code로 Dataset을 복원한다.
[dataset_rosters_v2.json](dense_ar_longbench400_v2_20261122/publication/provenance/dataset_rosters_v2.json)

[reconstruction_validation_v2.json](dense_ar_longbench400_v2_20261122/local_reconstruction_validation/reconstruction_validation_v2.json)

[raw_metric_recalculation_v2.json](dense_ar_longbench400_v2_20261122/analysis/raw_metric_recalculation_v2.json)

[resource_closeout_v2.json](dense_ar_longbench400_v2_20261122/state/resource_closeout_v2.json)

Commit/Push 및 Remote SHA는 별도의 최종 Publication Receipt에 기록한다.
