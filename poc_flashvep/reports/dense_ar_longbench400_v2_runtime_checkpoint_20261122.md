# LongBench400-v2 Runtime Checkpoint

상태: `ACTIVE`. 이 Commit은 최종 연구 완료를 의미하지 않는다.
기준 Commit은 `5ebed239073b4fa3709361038cbec026d1c32b78`이다.
실험 결과와 실행 Config는 `dense_ar_longbench400_v2_20261122`에 저장한다.

## 완료한 검사

- 원문 400개를 동결했다. Input 구간별 Count는 각각 100개다.
- Output 구간별 Count는 121, 106, 92, 81개다.
- 마지막 Input 구간의 4개 Cell은 46, 31, 17, 6개다.
- 400개 선정 후에는 요청을 교체하지 않았다.
- 독립 Compiled C1 Reference의 실제 Output 길이를 별도로 측정했다.
- Layer 4개와 Genuine Hybrid 4개가 N8 Strict 및 N32/N400 검사를 통과했다.
- NativeStrong, Common-policy Native, Agg 및 Whole-prefill Deflection도 N400 검사를 통과했다.
- Strong Stock Token의 Cut8, Cut12 및 Equivalent Cut11.5가 검사를 통과했다.

이 목록은 `CORRECTNESS_ONLY` 결과다. 성능 우위를 의미하지 않는다.

## 보존한 실패와 수정

Partial Prefix Compiler의 초기 ABI 오류와 KV·Hidden 불일치를 보존했다.
새 경로는 독립 Config/Cache와 원래 CUDA Prompt 연산을 사용한다.
D Decode는 기본 Compiled 경로를 유지한다.
Compiler On/Off 성능은 같은 Source에서 별도로 비교한다.

Agg의 정상 Preemption 이후 관측 Cursor 오류를 수정했다.
Request별 Preemption 증거가 있을 때만 관측 Buffer를 초기화한다.
모델 또는 Scheduler를 변경하지 않았다.
새로운 Agg N400 검사는 통과했다.

Stock Token의 오래된 v1 Suffix Map은 폐기하지 않고 무효 Config로 보존했다.
새 Prompt 길이와 Cut으로 Map을 다시 생성했다.
Cut2, Cut4 및 일부 추가 Stock Token 후보는 여전히 N8 Strict 검사를 통과하지 못했다.
이 후보들은 Primary 성능 기준선에 포함하지 않는다.

추가 N1 검사에서는 Attention 입력·출력과 전체 KV·Hidden이 일치했다.
N1은 추가 Attention 관측을 제거해도 통과했다.
같은 GPU Pair의 N8도 Numerical Observer를 추가하면 통과했다.
따라서 Warm-up 및 관측의 영향을 분리하는 검사를 진행한다.
현재 원인은 `ROOT_CAUSE_NOT_IDENTIFIED`다.
동기화만 추가하는 별도 Counterfactual도 대기 중이다.
관측 경로의 통과를 Clean 경로의 수정 완료로 간주하지 않는다.

## 성능 실험 계약

C0 계획은 720개 독립 실행으로 구성한다.
Input 4구간, Output 8/16/64/128/192, Cut2/4/8/12를 검사한다.
Discovery는 Cell별 N1이다. Held-out은 Cell별 N3다.
실제 Batch는 1/1/1이다.
실제 Model Visit 기록으로 P/D의 유효 Prompt Token-Layer Rows도 검사한다.
정적인 Partition 설정만으로 실제 실행량이 같다고 주장하지 않는다.

C0 Token은 Mode0 메커니즘 대조군이다.
C3의 Compiled Stock Token과 혼동하지 않는다.
Point Winner 색은 Start-mean TTFT p95 비율을 따른다.
CI와 Decode Gate는 별도로 표시한다.

C3의 기존 QPS Grid는 0.8/1.4/2.0/2.6/3.2/4.0/4.8/6.0이다.
64개 Pilot의 실제 도착률은 Offered QPS의 약 0.728배였다.
후속 Native 208개 실행은 QPS2.6과 4.8에서 Backlog를 보였다.
기존 Grid와 실행 결과는 모두 유지한다.
새 Axis의 Clean 성능을 보기 전에 QPS2.0의 튜닝 및 N3 Confirmation을 추가로 동결했다.
설정 근거는 `configs/c3_confirmation_knee_refinement_v3.json`에 기록했다.

모든 C3 최종 후보는 Discovery에서 선정한 뒤 Held-out 전에 고정한다.
NativeStrong과 동일 D 정책 Native를 모두 Decode 대조군으로 사용한다.
ITL p99 및 TPOT p95 비율의 Paired 95% CI 상한은 각각 1.05 이하여야 한다.
Absolute ITL 34ms, TPOT 33ms 및 TTFT 5,000ms 기준은 변경하지 않는다.

## 아직 확정하지 않은 결과

- C3 Main 및 Held-out Serving 우위는 미확정이다.
- 강한 Stock Token 대비 Hybrid의 순이익은 미확정이다.
- Online GPU-exclusive Operator 비중은 미확정이다.
- Decode-safe Executable Window는 미확정이다.
- Active-decode Dynamic Insertion은 검증 전까지 `NOT_RUN`이다.

Output Prior는 Discovery208만 사용해 동결했다.
Task 및 Input Metadata를 사용한다. Held-out의 실제 EOS를 Feature로 사용하지 않는다.
별도의 C0 Fixed-output Rule도 Discovery Cell만 사용해 동결했다.
두 예측 결과를 실제 Dynamic Serving 이득으로 표현하지 않는다.

## 자원과 재현성

네 개 H100 Pair를 사용한다. 각 GPU의 SM 수는 132다.
Primary 비교는 같은 QPS와 Start 안에서 동일 Pair를 사용한다.
Actual Batch, B/S, Source Credit, D Cap 및 Hidden Pool은 서로 다른 설정이다.
Layer/Hybrid의 Source 및 Destination Hidden Pool은 각각 16 Slots다.
Current C3 Source Credit 기본값은 8이다. 16도 공통 튜닝 후보에 포함한다.
KV Budget은 GPU당 42GiB다.

사용자 Checkout과 Index는 유지한다.
기존 v1 코드 및 보고서는 수정하지 않는다.
실패한 실행과 미실행 Config도 보존한다.
각 Pair의 실험이 끝나면 승인된 Burner를 복원한다.
최종 종료 시에는 GPU0–7의 Burner 상태와 Remote SHA를 다시 확인한다.
