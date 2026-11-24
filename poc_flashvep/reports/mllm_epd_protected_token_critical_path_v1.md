# Protected Token Split — Request 경로와 SLO 실패 원인

증거는 실제 E/P/D Online 결과의 `TRACE_ANALYSIS`다. 전체 Task Quality는 PENDING이다.
동일 Request ID와 고정 Arrival의 독립 Start3회를 연결했다. Natural과 Replay는 별도다.
Host 경계를 중복 없이 합산했다. 이 합계는 Mean TTFT를 설명하며 p95의 가법 분해가 아니다.
P/E HTTP 구간에는 GPU Compute, Queue, Scheduler와 Transfer가 함께 들어 있다.
이 보고서는 그 구간을 GPU-exclusive 계산시간으로 바꾸지 않는다.

## 같은 요청의 평균 경로 차이

아래는 Equal-output Replay다. 양수는 후보가 Native보다 느리다는 뜻이다.
NativeStrong은 Stock Compiler다. NativeCommon은 후보와 공통 LLM Compiler 설정을 사용했다.
두 Native 모두 원래 Admission을 유지했다. Candidate의 추가 Suffix 보호 정책은 같은 정책이 아니다.

| Subset | Candidate | Native | E HTTP Δms | P HTTP Δms | D Coordinator Wait Δms | D Submit→First Δms | Mean TTFT Δms [95% CI] |
|---|---|---|---:|---:|---:|---:|---|
| Main100 | Static10 | NativeStrong | -6.794 | -226.179 | 971.956 | 41.716 | 780.854 [639.307, 929.115] |
| Main100 | Static10 | NativeCommon | 221.297 | -305.329 | 971.956 | 44.822 | 933.063 [817.498, 1020.626] |
| Main100 | Idle10 | NativeStrong | -245.297 | -59.312 | 145.568 | 12.109 | -146.817 [-239.963, -68.629] |
| Main100 | Idle10 | NativeCommon | -17.205 | -138.462 | 145.568 | 15.216 | 5.393 [-181.896, 145.673] |
| Heldout50 | Static10 | NativeStrong | 2.423 | -145.820 | 2790.985 | 33.492 | 2680.954 [2574.822, 2773.657] |
| Heldout50 | Static10 | NativeCommon | 27.544 | -155.975 | 2790.985 | 36.098 | 2698.824 [2567.144, 2803.053] |
| Heldout50 | Idle10 | NativeStrong | -31.501 | -35.870 | 161.002 | 7.481 | 100.894 [56.185, 146.491] |
| Heldout50 | Idle10 | NativeCommon | -6.381 | -46.025 | 161.002 | 10.087 | 118.764 [86.149, 145.562] |

## 확인한 범위

정적 Token은 기존 D Stream이 끝난 뒤 Suffix를 시작했다. 이를 위해 실제 Coordinator 대기가 발생했다.
D에서 Prefill을 실행하지 않는 Native는 이 추가 Suffix 보호 대기가 없다.
이 차이는 Partition과 Admission의 결합 효과다. 개별 원인의 인과적 비율은 이 표만으로 확정하지 않는다.
낮은 ITL은 긴 Suffix Admission 대기와 동시에 나타날 수 있다. 낮은 ITL만으로 Serving 이득을 판단하지 않는다.
현재 상태 Idle-fallback도 계획 후 실제 D 시작 시점까지 상태가 바뀔 수 있다.
계획 시 HTTP Idle은 GPU의 미래 Decode-safe Budget이 아니다.

## SLO와 Offline 진단

TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token을 유지했다.
8개 실패 조합은 원래 N3 분석에 있다. 이번 분석은 Gate 제거 Counterfactual도 계산했다.
Shapley-style Attribution은 실패 요청의 Token 수를 실패한 Gate에 균등 배분한다.
이는 Metric 분해다. 실제 GPU 또는 Host의 인과적 기여율이 아니다.
Zero-D-Gate Oracle은 관측된 Coordinator 대기만 제거한다. 다른 Queue와 Decode 경쟁은 다시 계산하지 않는다.
따라서 Oracle Goodput과 TTFT는 실제 Online Speedup이 아니다.
같은 Output 수의 Replay도 Answer String이 같다는 보장은 없다. Natural Output 변화도 별도로 보존했다.

## 재현 자료

Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`다.
모델 Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다. GPU4/5/6을 E/P/D로 사용했다.
B8192/S1024, KV42GiB, Block128, Original BF16, BI0, 원본 Vision과 Processor를 유지했다.
각 Stage 차이와 CI는 `analysis/token_root_cause_C37_*_v0.csv`에 있다.
PNG/PDF와 대응 CSV는 `figures/token_root_cause_C37_*_v0`에 있다.
모든 Raw/Arrival 검사는 앞선 N3 Recalculation을 통과했다. 기존 실패나 Frozen Manifest를 변경하지 않았다.
