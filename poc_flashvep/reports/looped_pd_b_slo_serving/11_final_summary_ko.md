# B 후속 PoC 최종 검토

## 1. 결론

현재 구현의 B를 lossless serving 최적화로 채택할 근거는 확보하지 못했다.
두 모델 모두 최종적으로 출력 정합성 기준을 통과하지 못했다.
작은 재현 실험과 일부 확대 실험은 수정 효과를 보였다.
그러나 새로운 seed와 더 큰 요청 집합에서 잔여 불일치가 나왔다.

과거 B의 SLO 개선을 잘못된 결과로 폐기하는 것은 아니다.
그 결과는 특정 workload와 SLO에서 측정한 제한적인 개선이다.
이번 목표인 실제 입력 분포의 지속 가능한 open-loop serving capacity와는 다르다.
과거 보고서의 `CONDITIONAL` 판정은 그대로 보존한다.

| 대상 | 현재 판단 | 근거 |
|---|---|---|
| Ouro, 기존 runtime, 2GPU B | NO_GO_CORRECTNESS | B와 PD_G1 생성 결과185/200 일치 |
| Huginn, 기존 runtime, 2GPU B | NO_GO_CORRECTNESS |193/200 일치 |
| Ouro, 수정 runtime, 2GPU B | NO_GO_CORRECTNESS |200/200 검증 후 새400개 요청의 두 번째 seed에서399/400 |
| Huginn, 수정 runtime | NO_GO_CORRECTNESS |200개 검증에서199/200, 대조군 반복198/200 |
| 4/8GPU B | NOT_MEASURED | 모델별 정합성 gate 때문에 B 확대를 수행하지 않음 |
| 공식 HF와의 lossless 동등성 | 미해결 | 작은 진단에서 Ouro7/8, Huginn6/8 |
| 새로운 loop-specific 원리 | 입증되지 않음 | Generic immutable-layer streaming과 별도 실측 비교 미완료 |

이 표의 분모는 요청별 생성 토큰열 수다. GSM8K/MATH 정답률이 아니다.
4/8GPU의 미측정은 해당 GPU 구성에서 실패를 실측했다는 뜻이 아니다.
강한 동일 GPU budget baseline 대비 B의 lossless speedup과 accepted capacity는 없다.

## 2. 무엇이 문제였나

### 2.1 Prompt KV 전송 오류만으로 설명되지 않는다

Ouro의 실제 실패 사례를8개 요청으로 재현했다.
B를 사용하지 않은 동일 PD_G1의 반복 실행에서도7/8만 일치했다.
따라서 B가 있어야만 발생하는 전송 오류로 단정할 수 없다.

실제 요청3의 출력 index125 직전 상태를 저장했다.
두 실행은 prompt와 이전 생성 토큰 이력이 같았다.
전달된 prompt KV도 같았다.
하지만 decode 중 생성된 R3 과거 KV는551,940개 BF16 원소가 달랐다.
비교한 전체 KV 원소 수는17,989,632개다.

해당 token의 최종 logits 최대 절대 차이는0.125였다.
한 실행의 최고 점수 차이는0.125였지만 다른 실행에서는 동점이었다.
Greedy argmax가 token4533과 token30으로 갈렸다.
그 뒤에는 서로 다른 token 이력을 사용하므로 차이가 더 커졌다.

배치에 따른 BF16 계산 경로 차이가 유력한 원인이다.
다만 모든 실패의 최초 연산을 특정한 것은 아니다.
남은 불일치를 모두 무해한 반올림 오차라고 단정하지 않는다.
모든 KV/state 오류를 배제한 것도 아니다.

### 2.2 무엇을 수정했나

선형연산의 작은 행렬 크기를 일관되게 맞췄다.
Decode recurrent graph는 기존 masked fixed-capacity 경로를 재사용했다.
Huginn의 attention-bearing coda에도 기존 active-row KV guard를 사용했다.
모델 Weight, BF16 tensor precision, R4/R32 depth는 바꾸지 않았다.
새 CUDA kernel이나 hidden-state 근사는 사용하지 않았다.

Padding은 실제 GPU 작업을 늘릴 수 있다.
따라서 수정 runtime은 B와 모든 대조군에 동일하게 적용했다.
그 비용을 serving 시간에서 제외하지 않았다.

수정 후보는 작은 KV 비교에서 차이를0개로 줄였다.
Ouro8/32/200개 요청의 B·대조군·대조군 반복은 모두 완전히 일치했다.
그러나 독립 serving400개 요청의 두 번째 seed에서 sample233이 갈렸다.
최초 차이는 출력 index4였다. 두 출력 길이는 모두128이었다.
출력 길이가 같다는 사실은 token과 계산 결과의 동등성을 보장하지 않는다.

Huginn도33개 요청은 완전히 일치했지만200개에서 기준을 통과하지 못했다.
정합성 기준을 완화하지 않았다. 이후 B의 추가 성능 확대를 중단했다.

### 2.3 수정이 공식 모델 출력을 복원한 것은 아니다

수정된 Ouro 대조군과 기존 대조군은142/200개 생성 결과가 일치했다.
기존 runtime 자체도 반복 실행에 민감했다.
따라서58개 차이를 모두 수정 때문이라고 분리할 수는 없다.
그러나 수정 결과를 기존 출력의 정확한 복원이라고 부를 수도 없다.

공식 HF 대비 정합성은 별도 미해결 항목이다.
전체 benchmark의 답변 정확도는 측정하지 않았다.
작은8문제 진단을 품질 재현으로 해석하지 않는다.

## 3. 왜 예전에는 B의 SLO 개선이 컸나

### 3.1 큰 숫자는 주로 특정 최대-ITL 조건에서 나왔다

과거 Huginn 대표 조건은 입력512, concurrency4, 출력 cap32였다.
SLO는 TTFT<=5초, 모든 관측 ITL<=150ms, E2E<=30초였다.
24개 paired prompt를3번 실행했다.72개는72종의 서로 다른 prompt가 아니다.
입력은 반복된 synthetic text였다.

| 과거 비교군 | SLO 통과 요청 | B 통과 요청 | B의 goodput 비율 |
|---|---:|---:|---:|
| DP |18/72|70/72|3.335×|
| DP_Mixed |9/72|70/72|6.792×|
| PD_G1 |56/72|70/72|1.310× [95% CI1.125–1.580]|

이 조건의 가장 강한 goodput 비교군은 PD_G1이었다.
DP_Mixed 대비6.792×는 GPU 처리 자체가6.792배 빨라졌다는 뜻이 아니다.
단 한 번의 긴 ITL도 요청 전체를 SLO 실패로 만들었다.
그 때문에 짧은 interruption 감소가 통과 요청 수를 크게 바꿀 수 있다.

측정값의 산술 분해는 `1.310443 = 1.250000 × 1.048355`다.
1.25는 통과 요청 수70/56의 비율이다.
1.048355는 두 방법의 전체 cohort 시간 비율이다.
이 분해는 실측 count와 시간의 계산이다. 별도의 GPU speedup 실험은 아니다.

### 3.2 같은 과거 trace를 이번 SLO로 다시 채점했다

새 GPU 실험을 한 것이 아니다.
이전 출력과 timestamp를 그대로 사용한 CPU `TRACE_RESCORING`이다.
이번 Huginn 기준은 TTFT<=1454ms와 요청 평균 TPOT<=119ms다.

| 과거 trace의 방법 | 과거 SLO 통과 | 이번 SLO 통과 | 이번 조건의 cohort goodput req/s |
|---|---:|---:|---:|
| DP |18/72|72/72|1.256845|
| DP_Mixed |9/72|72/72|1.234220|
| PD_G1 |56/72|72/72|1.028140|
| B |70/72|72/72|1.077855|

동일 trace에서도 B/PD_G1 비율은1.310×에서1.048×로 줄어든다.
DP_Mixed 대비는0.873×, DP 대비는0.858×다.
최대 ITL을 줄이는 효과와 평균 TPOT를 개선하는 효과는 같지 않다.
SLO 정의에 따라 가장 강한 baseline도 달라진다.
이 값은 과거 cohort 기준의 재채점이며 이번 open-loop capacity는 아니다.

### 3.3 SLO 변경만이 전부는 아니다

이번 실제 W1을 과거 SLO로 재채점해도 Huginn B가 우세하지 않았다.
PD_G1은177/200, B는166/200이 통과했다.
동일 arrival-window 분모에서 비율은0.938×다.
이 B 실행은 출력193/200만 일치했으므로 lossless 성능으로 채택하지 않는다.

Workload도 달라졌다.
이번에는200개 real-text ShareGPT prompt, 다양한 입력 길이, Poisson arrival, cap128을 사용했다.
긴 decode와 queue, batch 구성, 중간에 들어오는 prefill이 더 오래 상호작용한다.
입력 내용·길이·arrival·출력 길이 각각의 인과 기여율은 분리 실험하지 않았다.
어느 하나가 전체 차이의 원인이라고 단정하지 않는다.

이전 결과도 일반적인 E2E 우세는 아니었다.
과거 Huginn P512/C4/cap32의 B p95 E2E는4065.456ms였다.
가장 빠른 DP는3295.142ms였다. B speedup은0.811×였다.
Ouro도 cap32에서는1.090×였지만 cap128에서는 DP_Mixed 대비0.981×였다.
즉, 과거 신호부터 제한된 operating region의 `CONDITIONAL` 결과였다.

## 4. 이번 동일 출력 조건의 실제 성능

수정 runtime의 Ouro,2GPU,1req/s 결과다.
각 seed는400회 도착이며200개 원문을 두 번 사용했다. Prefix caching은 사용하지 않았다.

| Seed | DP_Mixed joint SLO | PD_G1 joint SLO | B joint SLO | B/PD_G1 출력 일치 |
|---|---:|---:|---:|---:|
|2026106001|43.75%|87.25%|80.25%|400/400|
|2026107001|53.25%|74.50%|68.25%|399/400|

모든 방법이90% 기준에 미달했다.
출력이 완전히 같은 첫 seed에서 B의 G1 대비 SLO goodput은0.920×였다.
독립적으로 통과한 start가 하나라서 start 간95% CI는 측정하지 못했다.
두 번째 seed는 출력 정합성이 실패했으므로 equivalent-work speedup으로 채택하지 않는다.

첫 seed의 G1/B p95 TTFT는876.059/1082.067ms였다.
P95 TPOT는30.541/30.872ms, p95 E2E는4419.234/4623.385ms였다.
이 지점에서 B는 더 빠르지 않았다.

수정 runtime은 기존 최적화 runtime과 비용 및 출력이 다를 수 있다.
수정 전·후 latency를 같은 작업의 speedup으로 합치지 않았다.
수정된 약한 대조군을 이겼다는 이유로 기존 최적화 engine보다 낫다고 주장하지 않는다.

## 5. E2E 구성요소와 통신 비용

다음은 host에서 관측한 요청 phase 비중이다.
분모는 유효 요청들의 E2E 시간 합이다. Cohort makespan이나 GPU utilization이 아니다.
각 phase에는 내부 scheduling, backpressure, synchronization이 포함된다.
GPU Attention/FFN/dispatch 순수 실행시간의 비중으로 읽으면 안 된다.

| Phase, % | 기존 Ouro G1/B | 기존 Huginn G1/B | 수정 Ouro G1/B, 첫 seed |
|---|---:|---:|---:|
| Coordinator queue |0.0050/0.0053|9.9914/10.5252|0.0039/0.0037|
| Reservation/control |0.2857/0.2865|0.0436/0.0439|0.3680/0.3946|
| P prefill + backpressure |9.5180/7.9439|3.8288/3.5228|10.7847/11.4444|
| Exposed handoff |0.5509/0.2827|0.0900/0.0252|0.6044/0.6060|
| D resume–first token |0.8574/1.7244|0.0183/0.0182|0.5483/0.7662|
| Remaining decode |88.7521/89.7286|86.0217/85.8580|87.6633/86.7603|
| Harness residual |0.0309/0.0285|0.0062/0.0068|0.0274/0.0249|

기존 두 모델의 B 열은 출력 차이가 있는 진단값이다.
수정 Ouro 첫 seed는400/400 일치한 값이다.
원본 소수점 전체와 요청 수는 `e2e_component_percent.csv`에 있다.

기존 Huginn의 평균 exposed handoff는11.914ms에서3.386ms로 줄었다.
하지만 G1에서 이 구간은 요청 E2E 합의0.0900%였다.
이 구간만 제거하는 상한으로 큰 E2E 개선을 설명할 수는 없다.
간접적인 queue 또는 prefill-backpressure 효과까지 없다는 뜻은 아니다.

수정 Ouro 첫 seed에서는 평균 exposed handoff가21.029ms에서21.865ms가 됐다.
이 경우에는 handoff 자체도 개선되지 않았다.

통신 payload도 줄지 않았다.
기존 Ouro는 두 방법 모두131,400,237,056bytes를 전송했지만 fragment는2058→3242였다.
기존 Huginn은460,435,430,400bytes로 같았고 fragment는7183→11114였다.
Fragment 증가에 따른 descriptor/event/polling 비용이 가능한 원인이다.
전체 slowdown을 그 비용만으로 설명했다고 주장하지 않는다.
겹치는 transfer 관측시간을 더해서 E2E 비중으로 사용하지 않았다.

## 6. 완료 범위와 미측정 항목

184개 실행 기록을 보존했다.167개 COMPLETE와17개 FAILED가 있다.
FAILED에는 모델 정합성 결과가 아닌 초기 진단 launcher/import 실패도 포함된다.
저장 evidence 검증의 무결성 오류는0개다.
원본 worker/common source, 기존 A/B/C source와 private RLT patch pin은 보존했다.

W1은 두 모델의200개 분포와 독립 반복을 측정했다.
W2/W3/W4와4/8GPU의 DP_Mixed/G1 대조군 결과도 CSV에 남겼다.
다만 B는 정합성 gate 이후 확대하지 않았다.
4/8GPU control은 주로100개 요청·단일 start라서 accepted scaling 개선으로 판단하지 않는다.

Fine-grained GPU Attention/FFN/communication critical-path 비중은 NOT_MEASURED다.
Native heavy profiler와 상세 DMA/event timeline도 최종적으로 확보하지 못했다.
관측한 host phase를 해당 GPU 수치로 바꿔 적지 않았다.
전체 GSM8K/MATH 답변 정확도와 공식 모델 품질 재현도 NOT_MEASURED다.
GenericLayer의 별도 live 구현은 NOT_MEASURED다.
Open-loop rate grid는 유한하며 무한 시간의 최대 capacity를 증명하지 않는다.

## 7. 시간 및 게시 상태

실험 시계는2026-10-29 21:29:30 KST에 시작했다.
자동 closeout은13.954시간 경과 시점에 결과를 정리했고 로컬 commit을 생성했다.
GPU 실험은 종료됐으며0–7 GPU의 버너는 복구됐다.
이후 GitHub가120MB 입력 JSON 때문에 최초 push를 거부했다.

Capacity 중단 후2026-10-30 13:04 KST에 사용자의 요청으로 보고·게시를 재개했다.
추가 GPU 실험은 수행하지 않는다.
따라서 GPU 실험과 최초 정리는15시간 이내였지만 최종 게시까지 포함한 원래15시간 상한은 지키지 못했다.
이 초과를 원래 예산 내 완료로 보고하지 않는다.

원본 `W1_long.json`은 삭제하거나 수정하지 않았다.
126,308,939bytes의 원본과 SHA-256은 유지한다.
GitHub에는 동일 bytes로 복원되는19,381,397bytes의 gzip 파일을 게시한다.
`configs/dataset_storage.json`에 원본·압축 파일의 hash와 검증 결과를 남겼다.
최초 실패 commit도 로컬 archive ref에 보존한다. 원격 강제 push는 사용하지 않는다.

## 8. 다음 단계

바로 B의 더 큰 capacity sweep을 반복하는 것은 권하지 않는다.
먼저 잔여 Ouro sample233과 Huginn 실패 요청의 최초 차이를 추적해야 한다.
Prefill/prelude/core/coda의 배치 크기·순서·KV pointer 경계 테스트가 필요하다.
Checkpoint, BF16, fixed depth와 엄격한 출력 기준은 유지해야 한다.

정합성이 확보되면 같은 workload에서 SLO 정의만 바꾸는 비교와 arrival만 바꾸는 비교를 분리해야 한다.
이를 통해 threshold 효과와 실제 execution 효과를 구분할 수 있다.
현재 결과만으로 B의 일반적인 우월성이나 완전한 불가능성을 주장하지 않는다.

## 9. 읽을 파일

- `09_final_decision.md`: 전체 실측 표와 원래 실행/수정 실행의 판정.
- `10_why_previous_gain_did_not_transfer.md`: 과거 신호와 이번 결과의 상세 영어 비교.
- `historical_vs_current_slo_rescoring.csv`: 기존 trace의 두 가지 SLO 재채점.
- `04_latency_percentiles.csv`: 모델·구성별 TTFT/TPOT/ITL/E2E mean과 P50/P95/P99.
- `06_transfer_and_critical_path.csv`: 전송량, fragment, host-observed stall, HBM.
- `e2e_component_percent.csv`: 요청 phase 비중과 분모·표본 수.
- `repair_epoch_report.md`: 수정 후보의 단계별 정합성과 serving 재실패.
- `b_repair_diagnosis.md`: 실제 logits/KV 재현과 ablation.
- `reproduce.md`, `commands.jsonl`, `configs/`, `raw/`: 실행 방법과 원본 evidence.
