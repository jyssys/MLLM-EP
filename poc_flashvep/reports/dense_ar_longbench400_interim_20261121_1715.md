# LongBench400 중간보고

기준 시각은 2026-11-21 17:15 KST다. 최종 연구 판정이 아니다.
사용자는 종료 시각을 2026-11-22 14:00 KST로 연장했다.
신규 대형 실험의 Dispatch 마감은 12:00 KST다. 마지막 2시간은 검증과 출판에 사용한다.
Dataset, SLO와 Measurement Contract는 변경하지 않는다.

## 실행과 연결 단절

캠페인은 2026-11-21 00:02:26 KST에 시작했다.
연결 단절 전 마지막 확인 가능한 제어 명령은 05:04:57 KST다.
정확한 응답 스트림 단절 시각과 네트워크 원인은 서버 로그만으로 확인할 수 없다.
이 오류를 GPU 또는 서버 장애로 해석하지 않는다.

독립 프로세스의 Queue는 연결 단절 뒤에도 실행됐다.
마지막 예약 GPU 실험은 12:32:14 KST에 완료됐다.
시작부터 약 12시간 30분 동안 예약된 실험을 실행했다.
개별 실행 Receipt는 완료 1,183개와 실패 6개다.
완료 개수에는 정합성 검사, 고립 계측과 Online 실험이 함께 포함된다.
완료 개수를 연구 성공 개수로 해석하지 않는다.

자동 분석 감시는 계속 실행됐다. 새 가설 선정과 후속 코드 작성은 이어지지 못했다.
예약 실험이 끝난 뒤 승인된 Burner가 GPU0–7을 사용했다.
17:02 KST 검사에서 모든 GPU의 사용률은 100%였다.
다른 사용자 Process는 종료하지 않았다.
새 종료 시각을 적용해 Resource Guardian과 분석 감시를 다시 시작했다.

## Dataset

`LongBench-Balanced400-v1`은 원문 요청 400개로 동결됐다.
Input 구간마다 100개를 확보했다. 원문 변경, Padding과 요청 복제는 사용하지 않았다.
Input 평균은 13,839.6 Tokens다. Natural Output 평균은 27.42 Tokens다.
이 값은 BI0 Concurrency1 Reference에서 측정했다.
다른 BI0 Batch Geometry의 실제 Output 길이는 별도로 기록한다.

| Input Tokens | Output 1–8 | Output 9–24 | Output 25–49 | Output 50–99 |
|---|---:|---:|---:|---:|
| 3000–5999 | 25 | 25 | 25 | 25 |
| 6000–9999 | 25 | 25 | 25 | 25 |
| 10000–19999 | 25 | 25 | 25 | 25 |
| 20000–36000 | 52 | 29 | 3 | 16 |

긴 Input의 두 Output 구간에서 후보가 부족했다. 16개 Cell의 완전 균형은 달성하지 못했다.
미달분은 동결 전에 정한 재배분 규칙으로 처리했다.
13개 QA/요약 Task의 선언된 Natural 후보 집합에서 확보한 결과다.
모든 LongBench Task 전체에서 가능한 최대 균형이라고 주장하지 않는다.
선정 집합은 일반 Production 분포를 대표하지 않는다.

Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.

## Genuine 2D와 Continuous 정합성

P는 Prefix의 Lower Layers를 계산한다.
D는 Prefix의 Upper Layers와 Suffix의 Full36 Layers를 계산한다.
단순히 두 Pure Axis 중 하나를 선택하는 구현과 다르다.

동일 Geometry의 N8 검사에서는 모든 유효 KV, 마지막 Hidden과 Greedy16이 Reference와 일치했다.
26개 Action에서 각각 N8을 실행했다. 총 208개 실행이 통과했다.
208개 실행은 독립 원본 Prompt 208개나 독립 Server Start 208회가 아니다.

Layer4, Token4와 Hybrid4/s≈L/8은 Natural400 Continuous 검사를 완료했다.
모든 Request에서 Imported Prefix KV Digest와 Coverage/Lifetime 검사를 통과했다.
Mixed Batch는 Layer 344개, Hybrid 537개였다.
Task Quality는 동결한 2 percentage point 비열등성 기준을 통과했다.
다른 BI0 Geometry의 Greedy Output은 모든 Request에서 같지 않았다.
따라서 Natural400 전체의 Strict Token Match라고 보고하지 않는다.

원래 모델과 기본 Compiler를 유지하는 Stock Token 경로도 구현했다.
Cut2/4/8/12는 N8/N32 검사를 통과했다. Cut8은 별도 Full400 Stress도 통과했다.
Cut1은 Strict Gate를 통과하지 못했다. 현재 성능 적격 집합에서 제외한다.
이 제외를 일반적인 Token 계산 오류의 확정으로 해석하지 않는다.

## Layer/Token 승자 지도

Input4 × Controlled Output4 × Cut4의 64개 Cell 반복 실행을 완료했다.
각 Cell은 실제 Batch1/1/1, Source Credit4와 동일 Aggregate GEMM Rows를 사용한다.
기본 반복은 독립 Start 3회다. 사전에 정한 세 핵심 Cell은 5회다.
Input 구간 안에서는 Output별 QPS와 Arrival Trace를 동일하게 유지했다.

현재 판정은 Layer 22개, Token 16개, 불확실 21개다.
나머지 5개는 지연 방향이 있어도 Decode Protection Gate를 통과하지 못했다.
이 5개를 구현 정합성 실패와 혼동하지 않는다.
CI는 Start 단위의 Pointwise Bootstrap이다. 64개 비교의 Familywise 오류를 제어하지 않는다.

| 조건 | Layer/Token TTFT p95 비율 | 95% CI | Start | 해석 |
|---|---:|---:|---:|---|
| Input 3K–6K, Output8, Last2 | 0.944 | [0.924, 0.965] | 3 | Layer의 Axis 우위 |
| Input 3K–6K, Output8, Last4 | 0.924 | [0.835, 1.034] | 5 | 불확실 |
| Input 3K–6K, Output8, Last8 | 1.139 | [1.083, 1.183] | 5 | Token의 Axis 우위 |
| Input 20K–36K, Output16, Last8 | 1.140 | [1.107, 1.177] | 5 | Token의 Axis 우위 |

짧은 Input의 QPS는 0.910229다. 마지막 행의 QPS는 0.763308이다.
이 표는 C0 저부하 Mechanism Control이다. 실제 Continuous 성능으로 재사용하지 않는다.
첫 행의 Layer/Native TTFT 비율은 1.003이며 CI는 [0.970, 1.032]다.
따라서 이 Layer/Token 승리는 Native 대비 Serving 이득을 입증하지 않는다.
작은 Cut이면 항상 Layer가 유리하다는 가설도 지지하지 않는다.

## Agg/Native Motivation

다음 값은 실제 Natural400을 사용한 독립 Start 3회의 평균이다.
모든 Baseline은 B8192/S1024와 기본 Continuous Scheduler를 사용한다.
각 값은 Start별 통계의 평균이다. 아래 표는 CI를 생략한 중간 요약이다.

| QPS | Method | TTFT p95 (s) | ITL p99 (ms) | Raw Tokens/s | SLO-goodput (Tokens/s) |
|---:|---|---:|---:|---:|---:|
| 1.8 | Agg DP2 | 1.554 | 278.15 | 50.28 | 28.77 |
| 1.8 | Native PD | 4.886 | 15.91 | 50.09 | 44.47 |
| 2.8 | Agg DP2 | 2.109 | 341.81 | 77.79 | 21.28 |
| 2.8 | Native PD | 36.683 | 17.33 | 62.01 | 6.53 |

Decode 격리와 TTFT의 Trade-off는 재현됐다.
QPS2.8에서는 Native의 Queue와 TTFT가 크게 증가했다.
이는 Drain을 포함한 Finite-cohort 결과다. Sustainable Capacity로 보고하지 않는다.

## 실제 Continuous Offloading

다음은 Discovery200, QPS1.8의 단일 Start다.
네 방법은 Mode0, Source Credit4와 B8192/S1024를 사용한다.
실제 Continuous Batch는 Method마다 달랐다. Batch-matched C0 결과가 아니다.

| Method | TTFT p95 (s) | ITL p99 (ms) | SLO-goodput (Tokens/s) |
|---|---:|---:|---:|
| Matched Native | 3.879 | 16.66 | 55.00 |
| Layer4 | 2.739 | 35.59 | 31.20 |
| Token4 | 2.475 | 35.20 | 34.92 |
| Hybrid4/s≈L/8 | 1.759 | 48.59 | 21.93 |

Hybrid의 TTFT는 낮았다. 그러나 Decode Tail과 SLO-goodput은 악화됐다.
이 결과는 Hybrid Serving GO가 아니다.
아직 Best-tuned Static 및 기본 Compiler를 유지한 강한 Token과의 최종 비교도 아니다.

## 원인 분석과 다음 실험

긴 Input의 Last4 고립 계측에서 P Kernel Union 차이는 약 30.80ms였다.
Attention Union 차이는 약 29.77ms였다.
두 P 계측은 Observer Gate를 통과했다.
단일 Capture 비교이며, Natural Online TTFT의 기여율을 입증하지 않는다.

Continuous Native 계측은 ITL p99를 약 10% 교란했다.
이때 관측한 P Busy와 D Kernel Inactivity는 진단값으로만 유지한다.
현재 Decode-safe Window는 검증되지 않았다.
검증되지 않은 Active-decode Insertion은 실행하지 않는다.

연장 예산의 우선순위는 다음과 같다.

1. 강한 기본-Compiler Token의 Cut별 Continuous 성능을 측정한다.
2. 같은 Stock Model에서 Compile만 변경해 Host/GPU 경로 차이를 검사한다.
3. API 추적을 제거한 낮은 간섭의 Kernel 계측을 검사한다.
4. Decode 보호가 가능한 Admission과 Strong Static/Hybrid 대조군을 확인한다.
5. Held-out 반복, Figure, 보고서, Commit/Push와 Resource Closeout을 완료한다.

Dataset이나 SLO를 바꿔 성공 조건을 만들지 않는다.
정합성 성공, Axis 우위와 실제 Serving 이득은 계속 분리한다.

## 근거

- [Dataset 계약](dense_ar_longbench400_dataset_contract.md)
- [Hybrid 정합성 검사](dense_ar_layer_token_2d_hybrid_correctness.md)
- [64-cell 반복 판정](dense_ar_longbench400_2d_20261121/analysis/axis_matrix_heldout_confirmation_r2.json)
- [Stock Token Gate](dense_ar_longbench400_2d_20261121/state/C3_stock_token_gate_v1.json)
- [예산 연장 기록](dense_ar_longbench400_2d_20261121/state/budget_extension_20261121_1710.json)
