# Dense AR Layer/Token/Dynamic 심층 PoC 실행 계약

## 범위와 판정

시작은 2026-11-16 16:33:39 UTC다.
연구 예산은 약 10시간이다.
TP2 원인 진단, 실제 Token-axis 이전, Layer-vs-Token 비교를 먼저 수행한다.
그다음 Safe-budget 변동성과 Dynamic-vs-Static의 독립적 이득을 검사한다.
유효한 중단 근거가 있으면 시간을 채우기 위한 반복은 하지 않는다.

Base Commit은 `f0e80395aef488a7b3c6148d4155784f87291df2`다.
Publication Branch는 `flashvep/dense-ar-layer-token-dynamic-headroom-20261117`다.
사용자 Checkout과 Index는 변경하지 않는다.
새 Source는 `poc_flashvep/dense_ar_layer_token_pd/`다.
Evidence는 `dense_ar_layer_token_dynamic_headroom_20261117/`다.

## 고정 환경

Model은 Qwen/Qwen3-8B다.
Revision은 `b968826d9c46dd6066d109eabc6255188de91218`다.
Original BF16, Exact Causal Attention과 Full36 Decode를 유지한다.
Greedy Sampling을 사용한다.
Thinking은 비활성화한다.
GPU0/1의 H100 SXM 80GB 두 대를 공정 비교에 사용한다.
vLLM 0.26.0, PyTorch 2.11.0+cu129와 NIXL 1.4.1을 재사용한다.
GPU당 KV Budget은 42 GiB다.
Prefix Caching은 비활성화한다.
기존 정합성 보정과 ACK/Lifetime 계약을 유지한다.
실행마다 실제 Runtime, GPU UUID, Source ZIP, Model Revision과 Dataset Hash를 기록한다.

## TP2 진단

기존 N8의 7개 Output 불일치는 재현 가능한 Red Signal이다.
동일 Prompt와 Output 분기 전의 같은 Token Prefix만 Tensor 비교에 사용한다.
Embedding부터 Logits까지 논리적인 TP Shard를 결합해 비교한다.
최초 Tensor 차이와 최초 Argmax 차이를 구분한다.
동일 입력의 Primitive Replay로 수치 원인을 검사한다.
FP32는 진단에만 사용할 수 있다.
TP1 Strict Parity와 Within-TP2 정합성은 별도 Gate다.
Strict Parity에 실패한 TP2를 적격 Primary 성능으로 표시하지 않는다.

초기 가설은 다음과 같다.

1. Row-parallel Projection의 BF16 부분합 반올림이 TP1과 다를 수 있다.
2. Column-parallel GEMM Shape가 최초 차이를 만들 수 있다.
3. Norm 경로 또는 Collective가 차이를 확대할 수 있다.
4. 실제 Weight Shard나 Rank 상태가 잘못됐을 수 있다.

각 가설은 Tensor 비교와 동일 입력 Replay로 검사한다.
사전 가설을 확인된 원인으로 표시하지 않는다.

## 두 이전 축과 정합성

Layer-axis는 P의 Prefix Layer 뒤에서 Hidden/Residual과 Partial KV를 이전한다.
D는 남은 Layer를 완료한 뒤 Full36 Decode를 수행한다.
후보는 Last0/2/4/8/12다.
Token-axis는 P가 Prefix Token의 Full36을 완료한 뒤 그 KV를 이전한다.
D는 정확한 Position에서 Suffix Token의 Full36을 수행한다.
P의 Prefix Sample은 최종 출력으로 사용하지 않는다.
초기 Suffix 후보는 512/1024/2048/4096 Tokens다.
P-local Chunked Prefill은 별도 기준선이다.

N8, N32와 N128을 순서대로 검사한다.
Output, 전체 Prompt KV, Final Hidden, Layer/Token 방문, ACK와 Credit 복원을 확인한다.
상세 Tensor 복사는 `CORRECTNESS_ONLY`다.
계측 실행의 시간을 Clean Serving 성능으로 사용하지 않는다.

## Serving과 통계

초기 Input은 8000 Tokens다.
Output은 16/64를 먼저 검사한다.
같은 Prompt와 Arrival Trace를 모든 Arm에 사용한다.
기존 Frozen SLO를 변경하지 않는다.
Best-vs-Best와 같은 이전 Work 비교를 구분한다.
독립 Server Start가 Bootstrap 재표본 단위다.
CI와 반복 수를 모두 보존한다.
과부하 Finite Cohort를 Sustainable Capacity로 표시하지 않는다.

Dynamic의 추가 기여는 튜닝된 Static과 비교한다.
같은 이전량, Cut Histogram과 실제 Group Geometry를 검사한다.
Frozen-state Counterfactual과 State-updating Replay를 구분한다.
Offline Oracle을 실제 Online Speedup으로 표시하지 않는다.
미래 Arrival과 실제 미래 완료시각은 Online 정책에서 사용하지 않는다.

## 자원과 보존

성능 측정은 공유 Quiet-lane Lock 안에서 실행한다.
승인된 Burner만 종료한다.
알 수 없는 GPU 소유자는 변경하지 않는다.
각 실행에 Timeout과 소유 Session Cleanup을 둔다.
실패한 실행도 보존한다.
GPU 실험을 하지 않는 동안과 최종 종료 시에는 승인된 Burner를 복원한다.
실제 GPU0~7의 활동을 확인한다.
새 Source와 Evidence만 Private Index로 Commit/Push한다.
Remote SHA와 기존 사용자·외부 Runtime 보존을 확인한다.

## 증거 수준

`MEASURED`, `CORRECTNESS_ONLY`, `TRACE_ANALYSIS`, `OFFLINE_ORACLE`을 구분한다.
`STRUCTURAL_ACCOUNTING`을 GPU 시간 절감으로 바꾸어 표현하지 않는다.
미확정 원인은 `NOT_IDENTIFIED`로 남긴다.
미측정은 `NOT_MEASURED`, 미실행은 `NOT_RUN`으로 남긴다.
실패는 `FAILED` 또는 `BLOCKED_CORRECTNESS`로 보존한다.
