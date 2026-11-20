# BI0 Axis Selector와 Conditional Oracle 계약

## 분석 대상

`OFFLINE_REPLAY`와 `OFFLINE_ORACLE`을 실제 Online Serving과 구분한다. 후보는 Native와 Last1/2/4/6/8/12의 Layer/Token Continuation이다. Token은 Completed-prefix KV를 전달한 뒤 Full36 Suffix를 계산한다. 모든 Streaming Token Pipeline을 대표하지 않는다.

실제 요청20개로 Host Service Cost를 보정한다. 서로 다른15개 요청으로 오차를 검사한다. 각 Task의 Input Length 순위로 Split을 먼저 동결했다. 후보 성능을 보고 유리한 요청으로 바꾸지 않았다.

Cost Model은 `[1, L/8192, (L/8192)^2]`의 비음수 계수를 사용한다. 각 Action의 P Submit→Ready와 Ready→First Worker Output을 따로 보정한다. P Submit→Ready는 Source IPC와 Admission 비용을 포함한다. Serial Cost 실행에는 다른 요청의 P Queue가 없다. P Prepare→Ready도 별도 보존한다.

Source Submission 비용을 제외하면 작은 요청당 차이가 Queue에서 누적될 수 있다. 따라서 Model v2는 P Submit→Ready를 Replay의 Source Service로 사용한다. 이 변경은 Calibration Model을 Fit하기 전에 동결했다. 두 시간은 Host Enclosure다. GPU-exclusive Kernel Time이 아니다. Validation MAE, p95/p99 Absolute Error와 Underprediction을 보고한다.

## 상태 갱신

Replay는 FIFO P1/D1과 Source Credit4를 갱신한다. Action이 바뀌면 이후 P/D 시작 시각과 Queue도 바뀐다. Source Credit은 D First Output에 반환되는 실제 Controlled 계약을 따른다. Request별 Frozen-state 최소 비용을 더한 값은 Serving Gain으로 사용하지 않는다.

Action별 Trajectory의 KV Ownership도 별도로 검사한다. Source는 전체36 Layer를 예약하는 보수적 상한을 사용한다. Block128로 반올림한다. D는 Input과 Output 및 Lookahead1Token을 포함한다. Source Credit4와 D1 계약에서 GPU당42GiB를 넘지 않는지 확인한다. 이는 `STRUCTURAL_ACCOUNTING`이다. Fragmentation을 재현한 Allocator Model이나 실측 Peak Memory가 아니다. Hidden Buffer의 별도 비용도 KV Budget과 구분한다.

Natural EOS의 실제 종료 길이를 Online에서 미리 알 수는 없다. Conditional Validation은 관측된 종료 길이를 사용한다. Oracle은 고정된 C1 Reference Output Length와 미래 Arrival을 사용한다. 이 정보는 Online Predictor 입력으로 허용하지 않는다.

Async Scheduling은 마지막으로 보이는 Token보다 더 많은 Query를 실행할 수 있다. 추가 Lookahead Work는 일부 First-output Service에 이미 포함된다. 이를 무조건 Output 종료 뒤에 추가하면 중복 계산할 수 있다. 반대로 마지막 Client Output을 Device-wide 완료로 해석하지 않는다.

## 비교 기준

Discovery Seed는2026112061이다. Held-out Seed는2026112062/63이다. QPS는1.2/1.65/2.35다. 각 QPS의 Fixed Action과 Migration Fraction은 Discovery에서 먼저 동결한다. Held-out Cell의 사후 Best Fixed는 별도 Hindsight Control이다.

비교는 Best Fixed Layer, Best Fixed Token, Best Fixed Overall, Dynamic-size Layer, Dynamic-size Token, Mixed-axis Oracle로 나눈다. Fixed 후보는 Cut과 Fraction0/0.25/0.5/0.75/1을 검토한다. Finite Beam Width는24다. 이 탐색을 전역 최적 상한이라고 부르지 않는다.

## 엄격한 성능 제한

현재 Replay는 Client ITL Tail을 모델링하지 않는다. 따라서 결과는 `TTFT-qualified Output Token Proxy`다. Frozen Joint SLO-goodput이 아니다. 정답 품질의 Action-dependent 변화도 모델링하지 않는다. Ordinary Continuous Batching과 Active-decode Gap 삽입도 포함하지 않는다.

이 Proxy에서5% Headroom이 보여도 Online Axis GO를 선언하지 않는다. Best Window-aware Fixed-axis 대비 Full Joint SLO의 추가 Headroom을 별도로 검증해야 한다. Prediction Error와 Queue 예측이 Held-out에서 실패하면 Oracle 결과는 탐색적 분석으로만 남긴다.

실제 완료한 Replay는 아래에 기록했다. 원래 `analysis/BI0_conditional_axis_oracle_v1.json`과 후속 감사`v2`를 모두 보존했다. 설계 문서의 존재만으로 Online Scheduler가 구현됐다고 판단하지 않는다.

## 완료한 분석과 판정

`OFFLINE_ORACLE`과 `OFFLINE_REPLAY`를 완료했다. QPS1.2/1.65/2.35와 Seed61/62/63의9개 Cell을 검사했다. Calibration20으로 Fit했다. 다른15개 요청으로 Service 오차를 검사했다. Full200 Queue Validation에는 Calibration20도 포함한다. 이 검사를 완전히 새로운200개 Prompt의 검증이라고 부르지 않는다.

Discovery의13개 Fixed Action에서 TTFT 예측 MAE는284.33–1897.95ms였다. TTFT p95는 실제값보다6.21–40.26% 낮게 예측했다. 예를 들어 Layer8은8172.84ms를5824.10ms로 예측했다. Token8은6057.29ms를5681.22ms로 예측했다. Model은 실제 Discovery의 Best Cut8과 달리 일부 Cell에서 Token12를 고정 후보로 선택했다.

현재 비용·Queue Model은 개선 이득을 정밀하게 검증할 정도로 보정되지 않았다. Client ITL Tail과 Task-quality 변화도 모델링하지 않는다. 따라서 실제 Joint SLO와 Deadline Admission에 사용하지 않았다.

## 탐색에서 찾은 추가 Headroom

Mixed-axis 후보는 Layer-only 또는 Token-only의 알려진 실행 경로도 포함할 수 있다. 원래 Beam24는 한 Cell에서 이 경로를 버렸다. 그 결과 약−0.02%의 잘못된 추가 Proxy가 나왔다. 원래 `v1`은 보존했다. `v2`는 이미 계산한 유효 Subset 경로를 Mixed 후보에 포함한다. Model Fit이나 Fixed 선택을 바꾸지 않았다.

수정한9개 Cell에서 Mixed-axis의 추가 Proxy는 Best Specialized-axis 대비0%였다. Size/Axis 탐색의 Best-per-cell Hindsight Fixed 대비 개선은 최대0.0248%였다. 이는 전역 최적 Headroom이0이라는 증명이 아니다. 유한 탐색과 현재 Model이5% 추가 이득을 찾지 못했다는 뜻이다.

QPS2.35/Seed63에서 Discovery-frozen Fixed 대비 Oracle Proxy는4.0279배였다. 그러나 같은 Cell의 Hindsight Fixed가 같은 Proxy를 달성했다. 이를 Dynamic Scheduling 이득으로 해석하지 않는다. 강한 고정 대조군과 사후 고정 재선정의 차이를 분리해야 한다.

최종 Oracle 판정은 `HOLD_NOT_FULL_SLO_VALIDATED`다. 추가 Online Axis Controller를 구현하지 않았다. 근거는 `analysis/BI0_conditional_axis_oracle_v2.json`, `BI0_tandem_predictive_validation_v1.json`과 `BI0_oracle_dominance_audit_v1.json`이다.

## 새 Trace의 동일 Output 진단

Frozen Model을 수정하지 않고 Seed2026112062의 동일 Output Control에 적용했다. 각 Arm의 Output은4723Tokens다. Native/Layer8/Token8의 실제 TTFT p95는17233.39/5784.64/4773.52ms였다. 예측값은14425.41/5543.89/4603.54ms였다. Request TTFT MAE는1449.19/412.27/192.30ms였다.

이 검사는 각 Arm1개 실제 Start의 `OFFLINE_REPLAY` 검증이다. Future EOS 차이를 제거한 뒤에도 Native의 Queue 예측 오차가 남았다. Model을 이 결과에 맞춰 다시 Fit하지 않았다. 근거는 `analysis/BI0_equal_output_tandem_validation_v1.json`이다.
