# P Admission·Source Credit의 Controlled Intervention

상태: FINAL.

작성 시각: 2026-11-23T12:10:23.364370+00:00.

기준 Commit: `0ae6cd973a4544c6dce13387882479b94858e0d0`. 연구 Branch: `flashvep/dense-ar-p-critical-path-20261123`.

Model: Qwen/Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`.
BI0, Original BF16, Exact Causal Attention, Greedy, Thinking Disabled를 유지합니다.
Full36 Decode와 Prefix Cache Disabled를 유지합니다.
B8192/S1024, GPU당 KV42GiB, KV Block128을 유지합니다.

GPU Pair는 H100 SXM80GB×2입니다. 실제 GPU UUID는 Environment와 각 Raw Runtime 파일에 보존합니다.
Primary QPS는2입니다. C3의 Actual Batch는 동적입니다. B/S를 Batch 수로 표현하지 않습니다.
Client 시각은 Coordinator 전달 시각입니다. HTTP Serving Latency를 측정한 값은 아닙니다.

LongBench400-v2는 변경하지 않습니다. 성능 확인은 동결한 Heldout192를 사용합니다.
Full400 SHA256: `0845a0e44f001acf55dfd16407dc05e80dc56103590a8be3ac772473c1eb6cd9`.
Heldout192 SHA256: `c00eb4dd16deaa319100b47f9b22cc658e8003bdda08b56611ce544dac5e35f7`.
Discovery208과 Heldout192를 구분합니다. Natural400 정합성 검사는 성능 측정이 아닙니다.
Natural EOS와 요청별 독립 C1 길이를 사용한 Equal-output Replay를 구분합니다.
Fixed-output 표의 Quality는 적격 Natural400 구성 검사를 계승합니다. EOS 이후 강제 출력은 새 Natural 정답으로 평가하지 않습니다.
독립 단위는 Server Start입니다. Request와 Token을 독립 Start로 사용하지 않습니다.

동결 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, TPOT≤33ms/token입니다.
NativeStrong과 Common-D-policy Native에 대한 ITL p99·TPOT p95의 상대5% Gate를 별도로 검사합니다.
각 상대 비율의 Paired95% CI 상한은1.05 이하여야 합니다. Absolute Decode Attainment는 각 Start에서95% 이상이어야 합니다.

## Safe Retirement

원래 Credit은 첫 Client Token에서 반환합니다.
새 경로는 Native Request 제거, 필요한 GPU Fence, Hidden READ ACK와 Owner 해제를 확인합니다.
조건이 충족되면 Logical Source Credit을 먼저 반환합니다.
첫 Client Token의 기존 반환 경로도 유지합니다. 반환은 Idempotent입니다.
KV Block을 수동 해제하지 않습니다. Native Allocator가 물리적 재사용을 관리합니다.
Credit 반환 전에 모든 물리적 KV Block이 Free Pool에 들어갔다고 주장하지 않습니다.

같은 Capsule에서 False/True Switch만 바꿉니다.
Source Credit16/8과 D Cap2는 유지합니다. Actual Batch는 별도로 기록합니다.
Queue가 Coordinator에서 P Worker로 이동한 것만으로 성공을 판정하지 않습니다.
Primary P 지표는 Arrival→P-ready입니다. 사전 목표는5% 감소입니다.
Normal Completion을 검증합니다. Cancellation과 Crash Recovery는 NOT_RUN입니다.

| Family | Workload | 비교 | Paired Starts | TTFT p95 Ratio [95% CI] | P Path Ratio [95% CI] | Goodput Ratio [95% CI] | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Credit Retirement | EqualReplay | OptimizedLayer/OriginalLayer | 3 | 0.9846 [0.9658, 1.0037], N=3 | 0.9839 [0.9764, 0.9876], N=3 | 1.0096 [1.0022, 1.0169], N=3 | True |
| Credit Retirement | EqualReplay | OptimizedHybrid/OriginalHybrid | 3 | 1.0169 [0.9774, 1.0481], N=3 | 0.9860 [0.9499, 1.0029], N=3 | 0.9761 [0.9624, 0.9876], N=3 | True |
| Credit Retirement | EqualReplay | OriginalLayer/StockToken | 3 | 1.2295 [1.1831, 1.2635], N=3 | 4.4621 [3.1256, 5.4139], N=3 | 0.8509 [0.7158, 0.9343], N=3 | True |
| Credit Retirement | EqualReplay | OptimizedLayer/StockToken | 3 | 1.2106 [1.1732, 1.2505], N=3 | 4.3904 [3.0659, 5.3408], N=3 | 0.8591 [0.7174, 0.9493], N=3 | True |
| Credit Retirement | EqualReplay | OriginalHybrid/StockToken | 3 | 1.4174 [1.2440, 1.5599], N=3 | 6.2206 [4.8023, 7.3561], N=3 | 0.6998 [0.3961, 0.9072], N=3 | True |
| Credit Retirement | EqualReplay | OptimizedHybrid/StockToken | 3 | 1.4414 [1.2399, 1.6228], N=3 | 6.1337 [4.5621, 7.3783], N=3 | 0.6831 [0.3855, 0.8912], N=3 | True |
| Credit Retirement | Natural | OptimizedLayer/OriginalLayer | 3 | 1.0119 [0.9771, 1.0740], N=3 | 0.9439 [0.9347, 0.9726], N=3 | 1.1575 [1.0327, 1.7279], N=3 | True |
| Credit Retirement | Natural | OptimizedHybrid/OriginalHybrid | 3 | 1.1512 [0.9388, 1.4430], N=3 | 1.0047 [0.6711, 1.4453], N=3 | 1.0834 [0.7827, 1.5577], N=3 | True |
| Credit Retirement | Natural | OriginalLayer/StockToken | 3 | 0.7897 [0.6023, 0.9441], N=3 | 2.6281 [1.5223, 3.5274], N=3 | 0.7178 [0.4137, 0.9008], N=3 | True |
| Credit Retirement | Natural | OptimizedLayer/StockToken | 3 | 0.7991 [0.6333, 0.9248], N=3 | 2.4807 [1.4704, 3.2968], N=3 | 0.8309 [0.6614, 0.9303], N=3 | True |
| Credit Retirement | Natural | OriginalHybrid/StockToken | 3 | 0.7806 [0.5136, 1.0412], N=3 | 3.5157 [1.5786, 5.0194], N=3 | 0.6643 [0.3513, 0.9424], N=3 | True |
| Credit Retirement | Natural | OptimizedHybrid/StockToken | 3 | 0.8986 [0.7216, 1.0477], N=3 | 3.5322 [2.0473, 4.7415], N=3 | 0.7197 [0.3842, 0.9471], N=3 | True |
| Partition Geometry | EqualReplay | OptimizedLayer/OriginalLayer | 3 | 1.0772 [1.0289, 1.1298], N=3 | 0.6057 [0.4528, 0.7820], N=3 | 0.8933 [0.7928, 1.0141], N=3 | True |
| Partition Geometry | EqualReplay | OptimizedHybrid/OriginalHybrid | 3 | 1.0572 [1.0219, 1.0843], N=3 | 1.1061 [1.0192, 1.2564], N=3 | 0.8043 [0.6097, 0.9668], N=3 | True |
| Partition Geometry | EqualReplay | OriginalLayer/StockToken | 3 | 1.1722 [1.1425, 1.2036], N=3 | 3.2358 [3.0315, 3.3787], N=3 | 0.8461 [0.7178, 0.9503], N=3 | True |
| Partition Geometry | EqualReplay | OptimizedLayer/StockToken | 3 | 1.2627 [1.2221, 1.2972], N=3 | 1.9598 [1.4227, 2.5887], N=3 | 0.7558 [0.6845, 0.8585], N=3 | True |
| Partition Geometry | EqualReplay | OriginalHybrid/StockToken | 3 | 1.2080 [1.1546, 1.2491], N=3 | 3.8907 [3.4019, 4.2946], N=3 | 0.7366 [0.6156, 0.8574], N=3 | True |
| Partition Geometry | EqualReplay | OptimizedHybrid/StockToken | 3 | 1.2770 [1.2165, 1.3408], N=3 | 4.3036 [4.1041, 4.4823], N=3 | 0.5925 [0.4340, 0.7876], N=3 | True |
| Partition Geometry | Natural | OptimizedLayer/OriginalLayer | 3 | 1.2151 [1.1574, 1.2698], N=3 | 0.6252 [0.5172, 0.7395], N=3 | 0.9251 [0.8601, 0.9813], N=3 | True |
| Partition Geometry | Natural | OptimizedHybrid/OriginalHybrid | 3 | 1.1426 [1.0727, 1.2253], N=3 | 1.1436 [0.8605, 1.4506], N=3 | 0.8802 [0.7459, 1.0683], N=3 | True |
| Partition Geometry | Natural | OriginalLayer/StockToken | 3 | 0.8772 [0.6029, 1.5427], N=3 | 2.8097 [2.4145, 3.2092], N=3 | 0.8123 [0.6756, 0.9354], N=3 | True |
| Partition Geometry | Natural | OptimizedLayer/StockToken | 3 | 1.0658 [0.7655, 1.7806], N=3 | 1.7565 [1.6141, 1.9113], N=3 | 0.7514 [0.5818, 0.9180], N=3 | True |
| Partition Geometry | Natural | OriginalHybrid/StockToken | 3 | 1.0595 [0.7880, 1.7562], N=3 | 3.8109 [3.2106, 4.3225], N=3 | 0.7570 [0.6146, 0.8846], N=3 | True |
| Partition Geometry | Natural | OptimizedHybrid/StockToken | 3 | 1.2106 [0.9651, 1.8816], N=3 | 4.3580 [3.3801, 5.7646], N=3 | 0.6663 [0.6375, 0.7067], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OptimizedLayer/OriginalLayer | 3 | 1.1069 [1.0906, 1.1190], N=3 | 0.6531 [0.5633, 0.7688], N=3 | 0.9567 [0.9156, 0.9769], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OptimizedHybrid/OriginalHybrid | 3 | 1.0913 [1.0823, 1.1060], N=3 | 1.1572 [1.1094, 1.1835], N=3 | 0.9596 [0.8139, 1.0981], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OriginalLayer/StockToken | 3 | 1.1441 [1.1137, 1.1737], N=3 | 3.7668 [2.6908, 4.7912], N=3 | 0.7571 [0.6501, 0.8743], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OptimizedLayer/StockToken | 3 | 1.2664 [1.2207, 1.3020], N=3 | 2.4602 [1.7075, 3.2604], N=3 | 0.7243 [0.5954, 0.8532], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OriginalHybrid/StockToken | 3 | 1.1759 [1.1240, 1.2146], N=3 | 4.4127 [3.4506, 5.4929], N=3 | 0.6756 [0.6261, 0.7324], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OptimizedHybrid/StockToken | 3 | 1.2833 [1.2430, 1.3146], N=3 | 5.1064 [4.0308, 6.3883], N=3 | 0.6482 [0.5223, 0.8040], N=3 | True |
| Warm-regime Coordinator | EqualReplay | OriginalHybrid/HybridGCControl | 3 | 0.9856 [0.9811, 0.9910], N=3 | 0.9486 [0.9321, 0.9743], N=3 | 1.0413 [1.0117, 1.0811], N=3 | True |
| Warm-regime Coordinator | Natural | OptimizedLayer/OriginalLayer | 3 | 1.0828 [1.0543, 1.1123], N=3 | 0.6209 [0.5079, 0.7435], N=3 | 0.8456 [0.8100, 0.8991], N=3 | True |
| Warm-regime Coordinator | Natural | OptimizedHybrid/OriginalHybrid | 3 | 1.3037 [1.1452, 1.5205], N=3 | 1.5859 [1.3159, 2.0373], N=3 | 0.7280 [0.4709, 0.9109], N=3 | True |
| Warm-regime Coordinator | Natural | OriginalLayer/StockToken | 3 | 0.9013 [0.6682, 1.4047], N=3 | 2.8487 [2.3679, 3.0863], N=3 | 0.7209 [0.6177, 0.8116], N=3 | True |
| Warm-regime Coordinator | Natural | OptimizedLayer/StockToken | 3 | 0.9760 [0.7271, 1.4921], N=3 | 1.7687 [1.4551, 2.0167], N=3 | 0.6096 [0.5377, 0.6938], N=3 | True |
| Warm-regime Coordinator | Natural | OriginalHybrid/StockToken | 3 | 0.9037 [0.6810, 1.4181], N=3 | 3.1952 [2.9662, 3.3951], N=3 | 0.6399 [0.5472, 0.7275], N=3 | True |
| Warm-regime Coordinator | Natural | OptimizedHybrid/StockToken | 3 | 1.1781 [0.7804, 2.1609], N=3 | 5.0673 [4.1678, 6.0417], N=3 | 0.4659 [0.2585, 0.6628], N=3 | True |
| Output192 D cap2→4 | Fixed192 | OptimizedLayer/OriginalLayer | 1 | 1.0252 [CI 미확정], N=1 | 1.0169 [CI 미확정], N=1 | 0.9843 [CI 미확정], N=1 | True |
| Output192 D cap2→4 | Fixed192 | OptimizedHybrid/OriginalHybrid | 1 | 0.9122 [CI 미확정], N=1 | 0.9163 [CI 미확정], N=1 | 1.0620 [CI 미확정], N=1 | True |
| Output192 D cap2→4 | Fixed192 | OriginalLayer/StockToken | 1 | 3.0617 [CI 미확정], N=1 | 5.8187 [CI 미확정], N=1 | 0.3937 [CI 미확정], N=1 | True |
| Output192 D cap2→4 | Fixed192 | OptimizedLayer/StockToken | 1 | 3.1387 [CI 미확정], N=1 | 5.9170 [CI 미확정], N=1 | 0.3876 [CI 미확정], N=1 | True |
| Output192 D cap2→4 | Fixed192 | OriginalHybrid/StockToken | 1 | 3.0929 [CI 미확정], N=1 | 6.4922 [CI 미확정], N=1 | 0.5461 [CI 미확정], N=1 | True |
| Output192 D cap2→4 | Fixed192 | OptimizedHybrid/StockToken | 1 | 2.8212 [CI 미확정], N=1 | 5.9490 [CI 미확정], N=1 | 0.5800 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OptimizedLayer/OriginalLayer | 1 | 0.5599 [CI 미확정], N=1 | 0.4489 [CI 미확정], N=1 | 1.0157 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OptimizedHybrid/OriginalHybrid | 1 | 0.8852 [CI 미확정], N=1 | 0.7783 [CI 미확정], N=1 | 1.0000 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OriginalLayer/StockToken | 1 | 2.1594 [CI 미확정], N=1 | 2.9705 [CI 미확정], N=1 | 0.9841 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OptimizedLayer/StockToken | 1 | 1.2090 [CI 미확정], N=1 | 1.3334 [CI 미확정], N=1 | 0.9995 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OriginalHybrid/StockToken | 1 | 1.2641 [CI 미확정], N=1 | 1.4628 [CI 미확정], N=1 | 0.9997 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed16 | OptimizedHybrid/StockToken | 1 | 1.1190 [CI 미확정], N=1 | 1.1385 [CI 미확정], N=1 | 0.9997 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OptimizedLayer/OriginalLayer | 1 | 2.2402 [CI 미확정], N=1 | 1.7841 [CI 미확정], N=1 | 0.2516 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OptimizedHybrid/OriginalHybrid | 1 | 1.2617 [CI 미확정], N=1 | 1.3224 [CI 미확정], N=1 | 0.7182 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OriginalLayer/StockToken | 1 | 1.6312 [CI 미확정], N=1 | 2.9382 [CI 미확정], N=1 | 0.5047 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OptimizedLayer/StockToken | 1 | 3.6541 [CI 미확정], N=1 | 5.2419 [CI 미확정], N=1 | 0.1270 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OriginalHybrid/StockToken | 1 | 3.4077 [CI 미확정], N=1 | 8.1772 [CI 미확정], N=1 | 0.1650 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed64 | OptimizedHybrid/StockToken | 1 | 4.2994 [CI 미확정], N=1 | 10.8136 [CI 미확정], N=1 | 0.1185 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OptimizedLayer/OriginalLayer | 1 | 1.1412 [CI 미확정], N=1 | 1.1729 [CI 미확정], N=1 | 0.9359 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OptimizedHybrid/OriginalHybrid | 1 | 0.9913 [CI 미확정], N=1 | 0.9873 [CI 미확정], N=1 | 0.8918 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OriginalLayer/StockToken | 1 | 3.5044 [CI 미확정], N=1 | 19.8365 [CI 미확정], N=1 | 0.6314 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OptimizedLayer/StockToken | 1 | 3.9994 [CI 미확정], N=1 | 23.2661 [CI 미확정], N=1 | 0.5910 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OriginalHybrid/StockToken | 1 | 4.0370 [CI 미확정], N=1 | 27.4912 [CI 미확정], N=1 | 0.5883 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed128 | OptimizedHybrid/StockToken | 1 | 4.0021 [CI 미확정], N=1 | 27.1407 [CI 미확정], N=1 | 0.5247 [CI 미확정], N=1 | True |
| Fixed-output Stress | Fixed192 | OptimizedLayer/OriginalLayer | 3 | 1.0856 [1.0706, 1.0981], N=3 | 1.0882 [1.0654, 1.1073], N=3 | 0.7828 [0.6886, 0.9165], N=3 | True |
| Fixed-output Stress | Fixed192 | OptimizedHybrid/OriginalHybrid | 3 | 1.0201 [0.9792, 1.0527], N=3 | 1.0311 [0.9955, 1.0619], N=3 | 0.9937 [0.9689, 1.0224], N=3 | True |
| Fixed-output Stress | Fixed192 | OriginalLayer/StockToken | 3 | 2.9005 [2.8239, 2.9806], N=3 | 5.9255 [5.5141, 6.4440], N=3 | 0.4251 [0.3362, 0.5406], N=3 | True |
| Fixed-output Stress | Fixed192 | OptimizedLayer/StockToken | 3 | 3.1487 [3.0846, 3.1991], N=3 | 6.4484 [6.0552, 7.0205], N=3 | 0.3328 [0.3077, 0.3720], N=3 | True |
| Fixed-output Stress | Fixed192 | OriginalHybrid/StockToken | 3 | 2.9720 [2.8737, 3.1114], N=3 | 6.8415 [6.4120, 7.3383], N=3 | 0.4653 [0.4155, 0.5314], N=3 | True |
| Fixed-output Stress | Fixed192 | OptimizedHybrid/StockToken | 3 | 3.0319 [3.0085, 3.0542], N=3 | 7.0540 [6.6265, 7.6834], N=3 | 0.4624 [0.4130, 0.5199], N=3 | True |

## Coordinator 전달 지연

Client ITL은 Worker Emission Gap과 Delivery Delay 변화로 나눕니다.
기존1.85초 구간의 정확한 Host 원인은 계측하지 못했습니다.
별도 Diagnostic에서 파일 쓰기와 Python GC가 Token 전달을 지연한 구간을 관측했습니다.
Host Enclosure가 겹치므로 각 시간을 더해서 독점 원인 비중으로 계산하지 않습니다.
파일 쓰기 지연과 GC 지연은 별도의 One-factor 진단으로 검사합니다.
Worker Emission Gap이 긴 경우를 Coordinator 문제로 바꾸어 설명하지 않습니다.

| Diagnostic Run | Write Deferred | Cyclic GC Deferred | Client Max ITL ms | Worker Max Gap ms | GC Max ms |
| --- | --- | --- | --- | --- | --- |
| PC_CoordDiagnostic_Hybrid_ImmediateIO_Replay192 | False | False | 83.856 | 11.524 | 36.666 |
| PC_CoordDiagnostic_Hybrid_DeferredIO_Replay192 | True | False | 53.305 | 53.111 | 35.174 |
| PC_CoordDiagnostic_StockToken_ImmediateIO_Replay192 | False | False | 45.712 | 15.423 | 37.264 |
| PC_CoordDiagnostic_StockToken_DeferredIO_Replay192 | True | False | 15.952 | 15.937 | 38.738 |
| PC_CoordDiagnostic_NativeStrong_ImmediateIO_Replay192 | False | False | 69.648 | 69.511 | 35.992 |
| PC_CoordGC_Hybrid_NormalGC_BufferedReplay192 | True | False | 39.469 | 12.870 | 36.070 |
| PC_CoordGC_Hybrid_DeferredGC_BufferedReplay192 | True | True | 16.454 | 16.515 | NOT_RUN / 진행 중 |
| PC_CoordGC_StockToken_NormalGC_BufferedReplay192 | True | False | 128.998 | 18.408 | 35.555 |
| PC_CoordGC_StockToken_DeferredGC_BufferedReplay192 | True | True | 16.253 | 16.217 | NOT_RUN / 진행 중 |
| PC_CoordGC_NativeStrong_NormalGC_BufferedReplay192 | True | False | 68.149 | 68.184 | 28.460 |
| PC_CoordGC_NativeStrong_DeferredGC_BufferedReplay192 | True | True | 69.939 | 69.945 | NOT_RUN / 진행 중 |

공통 개선은 모든 비교군에 동일하게 적용합니다. Client Timestamp는 보정하지 않습니다.
JSON 직렬화는 원래 호출 시점에 유지합니다. 파일 쓰기만 측정 이후로 옮깁니다.
Coordinator의 Cyclic GC 상태는 측정 후와 예외 종료 시 복원합니다.
Reference Counting과 Worker GC는 변경하지 않습니다.
Hard Kill이 `finally`보다 먼저 발생하면 Buffered Evidence를 잃을 수 있습니다.
이 개선은 계측 경로의 개선입니다. Layer-specific GPU 효율로 주장하지 않습니다.

## 근거 파일

[configs/research_contract.json](dense_ar_p_critical_path_20261123/configs/research_contract.json)

[analysis/p_critical_causal_summary.json](dense_ar_p_critical_path_20261123/analysis/p_critical_causal_summary.json)

[analysis/c3_credit_confirmation_analysis.json](dense_ar_p_critical_path_20261123/analysis/c3_credit_confirmation_analysis.json)

[analysis/p_geometry_confirmation_analysis.json](dense_ar_p_critical_path_20261123/analysis/p_geometry_confirmation_analysis.json)

[analysis/serial_serving_operator_analysis.json](dense_ar_p_critical_path_20261123/analysis/serial_serving_operator_analysis.json)

[analysis/native_relative_accuracy_analysis.json](dense_ar_p_critical_path_20261123/analysis/native_relative_accuracy_analysis.json)

[analysis/coordinator_stock_compiler_artifact_audit.json](dense_ar_p_critical_path_20261123/analysis/coordinator_stock_compiler_artifact_audit.json)

[analysis/figure_manifest.json](dense_ar_p_critical_path_20261123/analysis/figure_manifest.json)

[analysis/historical_stage_recalculation.json](dense_ar_p_critical_path_20261123/analysis/historical_stage_recalculation.json)

[references/notion_archive_audit.json](dense_ar_p_critical_path_20261123/references/notion_archive_audit.json)
