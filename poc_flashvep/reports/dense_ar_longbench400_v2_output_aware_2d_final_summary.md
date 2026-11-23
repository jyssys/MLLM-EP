# LongBench400-v2 Output-aware 2D Serving

상태: FINAL.
생성 시각: 2026-11-23T02:04:03.379560+00:00.
경과 연구 시간: 18.954시간.
기준 Commit: `5ebed239073b4fa3709361038cbec026d1c32b78`.
연구 Branch: `flashvep/longbench400-v2-output-aware-2d-20261122`.

## 연구 판정

NO-GO_ON_TESTED_GRID — 테스트한 C3 고정 후보에서는 Strong Stock Token 대비 Layer/Hybrid의 TTFT·SLO-goodput 순이익을 확인하지 못했다. C0 Output Crossover는 별도의 CONDITIONAL 결과다. Dynamic은 HOLD/NOT_RUN이다.

## 실행 범위

C0 Discovery80개와 Held-out80개 Cell을 측정했다. Held-out의 모든 Cell은 독립 Start3회다.
Main400의 실제 완료 Start 수는 다음과 같다. 빈칸을 측정값으로 채우지 않는다.

| Offered QPS | Agg | NativeStrong | Layer | Stock Token | Hybrid | Deflection |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 1 | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 1.4 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2.0 | 3 | 3 | 3 | 3 | 3 | 3 |
| 2.6 | 1 | 1 | NOT_RUN | 1 | NOT_RUN | NOT_RUN |
| 3.2 | 1 | 1 | 1 | 1 | NOT_RUN | 1 |
| 4.0 | 1 | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 4.8 | 1 | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| 6.0 | 1 | 1 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

QPS2.0의 별도 Held-out192도 Natural EOS N3를 확보했다.
Equal-output Held-out Replay는 27개 계획 중 26개를 완료했다. Layer/Stock/Hybrid 및 그 두 Native 대조군은 N3다.
마지막 Cap1 Native Replay와 선택적 단일 Role CPC는 예산 때문에 미실행으로 남겼다.
모든 Method의 전체 QPS Grid와 과부하 N3 반복은 완료하지 못했다.
신규 Dispatch는 시작 후 18시간에 종료했다. 이미 실행 중인 소유 작업은 정상 종료까지 기다렸다.

## 검증 및 자원 복원

Configuration 실행 상태: `{'COMPLETE': 1317, 'NOT_RUN': 238, 'FAILED_EXECUTION': 4}`.
Raw Metric 재계산 상태: `{'PASS': 1303, 'NO_CLIENT_METRIC': 14, 'FAILED_EXECUTION': 4, 'NOT_RUN': 20}`.
실행 완료와 정합성·품질·Decode Gate 통과는 서로 다른 판정이다.
CPU 회귀 검사 PASS: `True`.
승인된 GPU0–7 Burner 복원 및 소유권·사용률 확인: `True`.
기존 사용자 Checkout과 Index를 보존했다. 다른 사용자 GPU 작업을 종료하지 않았다.

압축 해제 후 파일별 SHA256 검사 PASS: `True`.
검사 Archive 수: 2882. 검사 파일 수: 375318.
Full Raw Log, Tensor 및 원문 Dataset Archive는 로컬에 보존한다.
GitHub에는 재계산에 필요한 Structured Evidence, 정확한 Source Capsule 및 무손실 파생 Table을 게시한다.
원문 Dataset은 공개 ID/Hash Roster와 공식 Source로 재구성한다. 원문의 개별 License를 유지한다.
[lossless_archive_integrity_v2.json](dense_ar_longbench400_v2_20261122/analysis/lossless_archive_integrity_v2.json)

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

| 측정 대상 | Mean | Median | p10 | p25 | p75 | p90 | p95 | p99 | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Input Tokens | 13633.725 | 9953.000 | 4505.200 | 6013.250 | 18953.000 | 31128.200 | 31444.100 | 32978.020 | 3061.000 | 32988.000 |
| Selection Mode0 C1 Natural Output | 49.960 | 31.000 | 4.000 | 7.000 | 85.250 | 123.000 | 150.000 | 182.050 | 2.000 | 194.000 |
| Original Compiled C1 Natural Output | 53.032 | 31.000 | 4.000 | 7.000 | 84.750 | 120.200 | 147.150 | 187.000 | 2 | 1324 |

Compiled C1에서 200 Tokens를 초과한 요청은 2/400개다.
선정 Manifest의 400개 요청은 400개 고유 Request ID와 287개 고유 Context를 포함한다.
공유 Context 요청은 113개다. 요청 복제 또는 새 Padding/Truncation은 사용하지 않았다.
선정 시 Natural Output과 C3 Compiled C1 Output은 수치 경로가 다르다.
따라서 선정 Output 구간이 실제 C3의 모든 출력 길이를 보장하지 않는다. Dataset을 다시 선정하지 않았다.

## 최종 연구 질문

### Q1. 목표 4×4 Dataset 분포를 달성했는가?

원문 400개와 Input 구간별 100개는 확보했다. 정확한 25×16 배분은 불가능했다. 선정 전 후보 수는 [[430,144,44,54],[278,126,64,79],[759,177,131,86],[86,31,17,6]]이었다. 동결 배분은 앞의 세 행이 각각 [25,25,25,25]이고 마지막 행은 [46,31,17,6]이다. 새 Padding, Truncation 또는 요청 복제는 사용하지 않았다.

선정 Mode0 C1 Output과 실제 Compiled C1 Output은 다르다. 후자에서는 2/400개가 200 Tokens를 초과했다. Dataset을 다시 선정하지 않았다.

### Q2. 긴 Output에서 Layer의 Point Win이 증가하는가?

그렇다. C0 Held-out N3의 Output8/16/64/128/192에서 Layer Point Win은 각각 3/5/8/10/13개다. 각 분모는 Input4×Cut4의 16개 Cell이다. CI로 확정된 Layer 우위는 각각 1/2/5/8/11개다. 양쪽 Decode Gate 통과 Cell은 각각 16/16/9/4/7개다. Point Win 증가를 모든 Cell의 적격 Serving 우위로 표현하지 않는다.

### Q3. Input과 QPS를 통제한 Output 효과가 유지되는가?

C0에서는 같은 Input Bucket, Cut, Prompt 및 절대 QPS/Arrival에서 Output만 변경했다. 따라서 Controlled Output의 총 효과는 관측했다. 하지만 Output 변경은 Decode Residency와 Capacity-relative Load도 함께 바꾼다. 이 총 효과를 한 가지 Queue 또는 GPU 원인의 독립 효과로 확정하지 않는다. 16개 Output192/8 Endpoint 대조 중 13개는 Axis Ratio 대비 CI 상한이 1 미만이다. CI는 Start3회의 Pointwise 결과이며 다중 비교를 보정하지 않았다.

### Q4. C0 Crossover가 C3에서도 재현되는가?

C0의 긴 Output Layer 우위가 C3 Strong Stock Token에서도 유지된다는 근거는 확보하지 못했다. Iso-work C3 Bridge의 Layer/Stock TTFT p95 비율은 짧은 Input의 Output16에서 0.9824 [0.9316,1.0363]이었다. Output192에서는 1.4754 [1.4280,1.5227]이었다. 긴 Input에서는 각각 1.1613 [1.1485,1.1774]와 1.0852 [1.0745,1.0957]였다. 각 조건은 N3다. C0/C3의 QPS, Credit, Compiler 및 실제 Batch 차이가 있으므로 이 Bridge를 단일 변수의 인과 실험으로 표현하지 않는다.

### Q5. Layer 또는 Hybrid가 Strong Stock Token의 TTFT를 개선하는가?

확인하지 못했다. Natural Main400 QPS2.0 N3의 Layer/Stock TTFT p95 비율은 1.0170 [0.7862,1.2199], Hybrid/Stock은 1.0247 [0.7388,1.3917]이다. 별도 Natural Held-out192에서는 각각 1.0471 [0.7719,1.5230]와 1.1270 [0.6096,1.9694]이다. Natural Output 작업량 차이를 분리한 Held-out Fixed-length Replay N3에서는 Layer/Stock이 1.1588 [1.1379,1.1892], Hybrid/Stock이 1.3042 [1.2040,1.4201]이었다. Replay를 Natural EOS 성능으로 대체하지 않는다.

### Q6. Decode ITL/TBT/TPOT를 보호하는가?

QPS2.0의 Natural Main400 및 Held-out192에서 Layer, Stock Token, Hybrid와 Deflection은 NativeStrong 및 정확한 Common-policy Native 대비 ITL p99/TPOT p95의 5% CI Gate와 동결 Absolute Decode Attainment Gate를 통과했다. 이는 모든 개별 ITL이 34ms 이하라는 뜻은 아니다. Max ITL과 모든-ITL 요청 통과율을 별도로 보고한다. Equal-output Replay의 Layer/Stock/Hybrid도 Gate를 통과했다. Deflection Replay의 Common-policy Native 마지막 Start는 예산 때문에 미실행이므로 그 Replay Gate는 미확정이다. 이 판정을 다른 QPS 전체로 확장하지 않는다.

### Q7. SLO-goodput이 증가하는가?

Layer/Hybrid의 Strong Stock Token 대비 추가 Goodput은 확인하지 못했다. Natural Held-out192의 Layer/Stock SLO-goodput 비율은 0.7537 [0.6358,0.8741], Hybrid/Stock은 0.7666 [0.6051,1.0342]이다. 별도 Equal-output Replay에서는 각각 0.8574 [0.8004,0.9224]와 0.7880 [0.7381,0.8332]였다. 이 값은 Finite-cohort Output Throughput이다. Sustainable Capacity가 아니다.

### Q8. 실행 가능한 Decode-safe Window가 검증됐는가?

Kernel Activity는 일부 CPC에서 5% Observer Gate를 통과했다. 예를 들어 NativeStrong QPS3.2의 약20초 ROI에서 P Kernel Busy는 99.55%, D는 48.42%였다. 같은 ROI에서 P Waiting 평균은 27.31 Requests였다. 그러나 세 Clean/Profile 실행의 모든-ITL 요청 통과율은 90.75%/93.00%/96.50%였다. 물리적 Kernel 비활성과 P/D 불균형은 관측했지만, 준비된 Partition의 Tail Cost와 다음 Deadline을 함께 보호하는 실행 가능 Window는 검증하지 못했다. 일부 다른 Capture는 Observer Gate 실패로 Diagnostic에 남긴다.

### Q9. Dynamic 2D Selector가 Best Static보다 추가 이득을 만드는가?

아니다. 검증된 Window/Tail Predictor와 Queue·KV·ITL Tail을 갱신하는 적격 Oracle이 없으므로 Online Dynamic Insertion 및 2D Axis Selector는 NOT_RUN이다. Task+Input Output Prior는 Held-out192에서 Output-bin Accuracy 0.4740을 보였다. 이는 Axis 선택 또는 실제 Serving 이득이 아니다. 미래 Natural EOS를 Online Feature로 사용하지 않았다.

### Q10. 성공하지 못한 조건의 원인은 무엇인가?

D의 짧은 Continuation만으로 전체 TTFT를 설명할 수 없었다. Equal-output Held-out Replay에서 Layer−Stock의 P 앞 세 Host 구간은 각각 +669.227/+35.322/+837.684ms였고 P Prepare→Ready는 +212.984ms였다. D의 Submit→Accept/Accept→Prepare/Prepare→First Worker 이득은 각각 −108.048/−491.679/−197.011ms였다. 합계 Mean TTFT 차이는 +958.457ms다.

Hybrid는 Arrival→P Submit에서 +2267.264ms를 보였다. 이 값은 Mean Host 경로이며 TTFT p95 또는 GPU-exclusive 비용이 아니다. 별도 N3 Intervention에서는 P Compiler 및 Credit8→16의 TTFT 개선을 확인하지 못했다. D Cap4/1은 TTFT를 개선했고, D Compiler 단독은 악화됐다. Short Fusion은 그 회귀를 줄였다.

따라서 Compiler 하나 또는 GEMM 하나로 원인을 확정하지 않는다. Operator-exclusive Attention/GEMM 비중, 일부 비재현 Stock Strict 불일치 및 완전한 Safe Window 모델은 미확정이다. 테스트한 후보 밖의 전역 NO-GO를 주장하지 않는다.

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
- [dense_ar_longbench400_v2_compiler_and_admission_root_cause.md](dense_ar_longbench400_v2_compiler_and_admission_root_cause.md)
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
