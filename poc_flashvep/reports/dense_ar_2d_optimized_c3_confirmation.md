# Optimized C3의 독립 확인과 Decode 보호

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

## 비교 계약

Strong Stock Token은 원래 Compiled Full-model 경로를 유지합니다.
Credit0은 무제한 Coordinator Window입니다. GPU Memory가 무제한이라는 뜻이 아닙니다.
Common16 Control은 Logical Source Window16과 D Cap2를 사용합니다.
Strong Stock을 이 Control로 대체하지 않습니다.
같은 정책이 같은 Actual GPU Batch나 Hidden Buffer 소유권을 뜻하지 않습니다.

Geometry, Credit Retirement와 Common Coordinator 실험은 서로 다른 Family입니다.
서로 다른 Family의 Start를 합쳐 N을 늘리지 않습니다.
Natural EOS의 실제 Output Work가 다른 경우를 기록합니다.
Equal-output Replay에서는 Request별 Output Token 수를 정확히 맞춥니다.
Natural400 품질 통과만으로 Heldout 성능 통과를 주장하지 않습니다.

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

## Decode Gate

| Family | Workload | Arm | Starts | ITL p99 ms [CI] | TPOT p95 ms [CI] | Relative5% | Absolute95% | Quality | Start별 Max ITL ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Credit Retirement | EqualReplay | NativeStrong | 3 | 20.4253 [20.1228, 20.6776], N=3 | 8.8088 [8.6604, 8.9531], N=3 | False | False | True | [94.84229981899261, 81.32350817322731, 52.67554707825184] |
| Credit Retirement | EqualReplay | NativeCommon2 | 3 | 9.1844 [9.0922, 9.3011], N=3 | 8.1448 [8.1435, 8.1465], N=3 | True | True | True | [35.075874999165535, 34.43382866680622, 189.80525992810726] |
| Credit Retirement | EqualReplay | StockToken | 3 | 9.2247 [8.9268, 9.4921], N=3 | 8.3109 [8.2206, 8.3853], N=3 | True | True | True | [12.04938068985939, 37.56649233400822, 19.200969487428665] |
| Credit Retirement | EqualReplay | OriginalLayer | 3 | 8.4813 [8.4346, 8.5358], N=3 | 7.9645 [7.9447, 7.9889], N=3 | True | True | True | [44.26499456167221, 29.991930350661278, 36.429040133953094] |
| Credit Retirement | EqualReplay | OptimizedLayer | 3 | 8.5172 [8.4831, 8.5580], N=3 | 7.9550 [7.9512, 7.9595], N=3 | True | True | True | [44.450853019952774, 44.3244855850935, 41.10332950949669] |
| Credit Retirement | EqualReplay | OriginalHybrid | 3 | 8.5397 [8.5028, 8.5846], N=3 | 7.9815 [7.9625, 8.0071], N=3 | True | True | True | [47.53568768501282, 50.31631700694561, 61.666106805205345] |
| Credit Retirement | EqualReplay | OptimizedHybrid | 3 | 8.5332 [8.4582, 8.6298], N=3 | 7.9857 [7.9510, 8.0132], N=3 | True | True | True | [45.49791477620602, 1850.9587328881025, 43.12184266746044] |
| Credit Retirement | Natural | NativeStrong | 3 | 19.5079 [16.9364, 21.7751], N=3 | 8.7656 [8.6679, 8.8574], N=3 | False | False | True | [67.86762736737728, 71.02785259485245, 68.51057149469852] |
| Credit Retirement | Natural | NativeCommon2 | 3 | 9.1748 [8.9468, 9.4062], N=3 | 8.2026 [8.1448, 8.2760], N=3 | True | True | True | [40.73134623467922, 39.92745280265808, 42.96753369271755] |
| Credit Retirement | Natural | StockToken | 3 | 9.2575 [8.9924, 9.5030], N=3 | 8.4007 [8.3354, 8.4639], N=3 | False | True | True | [38.7110561132431, 34.20766070485115, 64.03381749987602] |
| Credit Retirement | Natural | OriginalLayer | 3 | 8.5409 [8.5154, 8.5692], N=3 | 7.9747 [7.9264, 8.0234], N=3 | True | True | True | [43.99791359901428, 37.662189453840256, 43.61974634230137] |
| Credit Retirement | Natural | OptimizedLayer | 3 | 8.5345 [8.5193, 8.5530], N=3 | 7.9560 [7.9463, 7.9652], N=3 | True | True | True | [355.1707249134779, 42.46481880545616, 42.530618607997894] |
| Credit Retirement | Natural | OriginalHybrid | 3 | 8.5893 [8.5527, 8.6168], N=3 | 7.9832 [7.9759, 7.9920], N=3 | True | True | True | [54.91405725479126, 64.5323246717453, 42.83948987722397] |
| Credit Retirement | Natural | OptimizedHybrid | 3 | 8.5879 [8.5405, 8.6491], N=3 | 7.9903 [7.9591, 8.0238], N=3 | True | True | True | [80.95981925725937, 43.601132929325104, 59.728531166911125] |
| Partition Geometry | EqualReplay | NativeStrong | 3 | 19.6704 [18.9837, 20.4742], N=3 | 8.8206 [8.7455, 8.9090], N=3 | False | False | True | [63.40130977332592, 73.27312603592873, 78.33965681493282] |
| Partition Geometry | EqualReplay | NativeCommon2 | 3 | 9.0848 [9.0447, 9.1386], N=3 | 8.1501 [8.1362, 8.1607], N=3 | True | True | True | [27.821030467748642, 29.18166294693947, 29.557492583990097] |
| Partition Geometry | EqualReplay | StockToken | 3 | 9.2920 [9.0208, 9.5632], N=3 | 8.2588 [8.1898, 8.3280], N=3 | False | True | True | [39.99166749417782, 89.14894424378872, 42.67824627459049] |
| Partition Geometry | EqualReplay | OriginalLayer | 3 | 8.4977 [8.4541, 8.5468], N=3 | 7.9579 [7.9518, 7.9627], N=3 | True | True | True | [34.142643213272095, 34.975431859493256, 32.61488303542137] |
| Partition Geometry | EqualReplay | OptimizedLayer | 3 | 8.5140 [8.4901, 8.5360], N=3 | 7.9487 [7.9282, 7.9699], N=3 | True | True | True | [41.558846831321716, 36.689337342977524, 35.622457042336464] |
| Partition Geometry | EqualReplay | OriginalHybrid | 3 | 8.4941 [8.4680, 8.5179], N=3 | 7.9502 [7.9406, 7.9623], N=3 | True | True | True | [53.1070102006197, 457.17231929302216, 42.52684488892555] |
| Partition Geometry | EqualReplay | OptimizedHybrid | 3 | 8.4807 [8.4306, 8.5392], N=3 | 7.9371 [7.9144, 7.9611], N=3 | True | True | True | [171.4673526585102, 44.313397258520126, 50.044262781739235] |
| Partition Geometry | Natural | NativeStrong | 3 | 18.2139 [16.0099, 20.0036], N=3 | 8.8585 [8.7892, 8.9451], N=3 | False | False | True | [59.952158480882645, 66.60382822155952, 79.578947275877] |
| Partition Geometry | Natural | NativeCommon2 | 3 | 8.8039 [8.7232, 8.8889], N=3 | 8.1317 [8.0856, 8.1673], N=3 | True | True | True | [34.19223055243492, 42.5126776099205, 42.78155975043774] |
| Partition Geometry | Natural | StockToken | 3 | 9.1219 [8.9681, 9.2525], N=3 | 8.3061 [8.2434, 8.3605], N=3 | False | True | True | [37.096260115504265, 32.59115293622017, 40.79297557473183] |
| Partition Geometry | Natural | OriginalLayer | 3 | 8.5502 [8.5288, 8.5786], N=3 | 7.9616 [7.9554, 7.9688], N=3 | True | True | True | [39.05919753015041, 37.461426109075546, 266.2359457463026] |
| Partition Geometry | Natural | OptimizedLayer | 3 | 8.5285 [8.4830, 8.5782], N=3 | 7.9786 [7.9683, 7.9891], N=3 | True | True | True | [37.4408233910799, 39.07599300146103, 67.27282702922821] |
| Partition Geometry | Natural | OriginalHybrid | 3 | 8.6296 [8.5917, 8.6591], N=3 | 7.9528 [7.9461, 7.9581], N=3 | True | True | True | [49.7359037399292, 51.56787298619747, 38.13457302749157] |
| Partition Geometry | Natural | OptimizedHybrid | 3 | 8.5047 [8.4701, 8.5319], N=3 | 7.9401 [7.9144, 7.9635], N=3 | True | True | True | [54.33879420161247, 56.476496160030365, 166.51160456240177] |
| Warm-regime Coordinator | EqualReplay | NativeStrong | 3 | 19.5204 [19.4093, 19.6245], N=3 | 8.7871 [8.6714, 8.9202], N=3 | False | False | True | [78.05035635828972, 62.39285133779049, 63.78483958542347] |
| Warm-regime Coordinator | EqualReplay | NativeCommon2 | 3 | 9.0428 [9.0070, 9.0874], N=3 | 8.1216 [8.1016, 8.1440], N=3 | True | True | True | [26.10711380839348, 24.49744939804077, 27.57333405315876] |
| Warm-regime Coordinator | EqualReplay | StockToken | 3 | 9.2424 [8.8942, 9.5701], N=3 | 8.3090 [8.1723, 8.4257], N=3 | False | True | True | [17.383234575390816, 15.441028401255608, 13.33954744040966] |
| Warm-regime Coordinator | EqualReplay | OriginalLayer | 3 | 8.4101 [8.3873, 8.4279], N=3 | 7.9547 [7.9373, 7.9738], N=3 | True | True | True | [11.928066611289978, 10.94982773065567, 22.54234068095684] |
| Warm-regime Coordinator | EqualReplay | OptimizedLayer | 3 | 8.4468 [8.4333, 8.4620], N=3 | 7.9411 [7.9200, 7.9682], N=3 | True | True | True | [14.398293569684029, 15.523804351687431, 11.330990120768547] |
| Warm-regime Coordinator | EqualReplay | OriginalHybrid | 3 | 8.4838 [8.4028, 8.5791], N=3 | 7.9629 [7.9412, 7.9885], N=3 | True | True | True | [42.39421337842941, 37.86826692521572, 34.44053418934345] |
| Warm-regime Coordinator | EqualReplay | OptimizedHybrid | 3 | 8.5244 [8.5006, 8.5471], N=3 | 7.9537 [7.9233, 7.9843], N=3 | True | True | True | [11.65643148124218, 38.12321089208126, 37.26768307387829] |
| Warm-regime Coordinator | EqualReplay | HybridGCControl | 3 | 8.5067 [8.4489, 8.5828], N=3 | 7.9467 [7.9393, 7.9558], N=3 | True | True | True | [43.90142112970352, 41.66825674474239, 36.28081455826759] |
| Warm-regime Coordinator | Natural | NativeStrong | 3 | 17.9280 [16.3891, 19.1282], N=3 | 8.7987 [8.7651, 8.8323], N=3 | False | False | True | [68.97606328129768, 64.45509567856789, 67.45134480297565] |
| Warm-regime Coordinator | Natural | NativeCommon2 | 3 | 8.9728 [8.8418, 9.0935], N=3 | 8.1378 [8.1082, 8.1602], N=3 | True | True | True | [27.97403559088707, 26.83514542877674, 26.20377205312252] |
| Warm-regime Coordinator | Natural | StockToken | 3 | 9.0332 [8.7927, 9.3264], N=3 | 8.2794 [8.1673, 8.4223], N=3 | True | True | True | [12.140417471528053, 16.224458813667297, 10.604612529277802] |
| Warm-regime Coordinator | Natural | OriginalLayer | 3 | 8.4745 [8.4642, 8.4832], N=3 | 7.9575 [7.9529, 7.9630], N=3 | True | True | True | [12.855209410190582, 11.633772403001785, 11.966539546847343] |
| Warm-regime Coordinator | Natural | OptimizedLayer | 3 | 8.4825 [8.4536, 8.5180], N=3 | 7.9362 [7.9154, 7.9593], N=3 | True | True | True | [10.284030809998512, 10.206632316112518, 13.492237776517868] |
| Warm-regime Coordinator | Natural | OriginalHybrid | 3 | 8.5476 [8.4703, 8.6200], N=3 | 7.9779 [7.9495, 8.0086], N=3 | True | True | True | [42.31921769678593, 12.099256739020348, 43.162232264876366] |
| Warm-regime Coordinator | Natural | OptimizedHybrid | 3 | 8.4623 [8.4078, 8.5185], N=3 | 7.9422 [7.9197, 7.9597], N=3 | True | True | True | [35.04814952611923, 36.43212281167507, 36.24660708010197] |
| Output192 D cap2→4 | Fixed192 | NativeStrong | 1 | 21.9086 [CI 미확정], N=1 | 11.7687 [CI 미확정], N=1 | False | False | True | [89.8223239928484] |
| Output192 D cap2→4 | Fixed192 | NativeCommon2 | 1 | 9.8109 [CI 미확정], N=1 | 8.9212 [CI 미확정], N=1 | False | True | True | [41.78920201957226] |
| Output192 D cap2→4 | Fixed192 | NativeCommon4 | 1 | 12.4022 [CI 미확정], N=1 | 11.5624 [CI 미확정], N=1 | False | False | True | [55.01743406057358] |
| Output192 D cap2→4 | Fixed192 | StockToken | 1 | 9.9482 [CI 미확정], N=1 | 9.5459 [CI 미확정], N=1 | False | True | True | [35.629475489258766] |
| Output192 D cap2→4 | Fixed192 | OriginalLayer | 1 | 8.5919 [CI 미확정], N=1 | 8.2493 [CI 미확정], N=1 | False | True | True | [329.8900779336691] |
| Output192 D cap2→4 | Fixed192 | OptimizedLayer | 1 | 8.5512 [CI 미확정], N=1 | 8.2461 [CI 미확정], N=1 | False | True | True | [48.81979338824749] |
| Output192 D cap2→4 | Fixed192 | OriginalHybrid | 1 | 8.5345 [CI 미확정], N=1 | 8.2669 [CI 미확정], N=1 | False | True | True | [108.44041220843792] |
| Output192 D cap2→4 | Fixed192 | OptimizedHybrid | 1 | 8.8468 [CI 미확정], N=1 | 8.4920 [CI 미확정], N=1 | False | True | True | [91.49399399757385] |
| Fixed-output Stress | Fixed16 | NativeStrong | 1 | 9.1560 [CI 미확정], N=1 | 7.8979 [CI 미확정], N=1 | False | True | True | [36.425624042749405] |
| Fixed-output Stress | Fixed16 | NativeCommon2 | 1 | 8.8093 [CI 미확정], N=1 | 7.9118 [CI 미확정], N=1 | False | True | True | [22.95980043709278] |
| Fixed-output Stress | Fixed16 | StockToken | 1 | 8.5555 [CI 미확정], N=1 | 7.9129 [CI 미확정], N=1 | False | True | True | [10.137252509593964] |
| Fixed-output Stress | Fixed16 | OriginalLayer | 1 | 8.6704 [CI 미확정], N=1 | 7.9347 [CI 미확정], N=1 | False | True | True | [42.37944632768631] |
| Fixed-output Stress | Fixed16 | OptimizedLayer | 1 | 8.5480 [CI 미확정], N=1 | 8.0079 [CI 미확정], N=1 | False | True | True | [9.402044117450714] |
| Fixed-output Stress | Fixed16 | OriginalHybrid | 1 | 8.5388 [CI 미확정], N=1 | 7.9131 [CI 미확정], N=1 | False | True | True | [9.219126775860786] |
| Fixed-output Stress | Fixed16 | OptimizedHybrid | 1 | 8.6635 [CI 미확정], N=1 | 7.9704 [CI 미확정], N=1 | False | True | True | [11.654647067189217] |
| Fixed-output Stress | Fixed64 | NativeStrong | 1 | 20.2318 [CI 미확정], N=1 | 8.9317 [CI 미확정], N=1 | False | False | True | [59.23233740031719] |
| Fixed-output Stress | Fixed64 | NativeCommon2 | 1 | 8.6185 [CI 미확정], N=1 | 8.1777 [CI 미확정], N=1 | False | True | True | [42.99473389983177] |
| Fixed-output Stress | Fixed64 | StockToken | 1 | 9.8190 [CI 미확정], N=1 | 8.7769 [CI 미확정], N=1 | False | True | True | [29.37348373234272] |
| Fixed-output Stress | Fixed64 | OriginalLayer | 1 | 8.5702 [CI 미확정], N=1 | 8.2174 [CI 미확정], N=1 | False | True | True | [46.40871286392212] |
| Fixed-output Stress | Fixed64 | OptimizedLayer | 1 | 8.5961 [CI 미확정], N=1 | 8.2145 [CI 미확정], N=1 | False | True | True | [42.37276315689087] |
| Fixed-output Stress | Fixed64 | OriginalHybrid | 1 | 8.6433 [CI 미확정], N=1 | 8.1997 [CI 미확정], N=1 | False | True | True | [235.4424148797989] |
| Fixed-output Stress | Fixed64 | OptimizedHybrid | 1 | 8.7451 [CI 미확정], N=1 | 8.2004 [CI 미확정], N=1 | False | True | True | [56.98901228606701] |
| Fixed-output Stress | Fixed128 | NativeStrong | 1 | 22.9003 [CI 미확정], N=1 | 10.2004 [CI 미확정], N=1 | False | False | True | [63.239939510822296] |
| Fixed-output Stress | Fixed128 | NativeCommon2 | 1 | 9.8305 [CI 미확정], N=1 | 8.9156 [CI 미확정], N=1 | False | True | True | [45.60929723083973] |
| Fixed-output Stress | Fixed128 | StockToken | 1 | 10.1654 [CI 미확정], N=1 | 9.4131 [CI 미확정], N=1 | False | True | True | [43.70083101093769] |
| Fixed-output Stress | Fixed128 | OriginalLayer | 1 | 8.6448 [CI 미확정], N=1 | 8.2486 [CI 미확정], N=1 | False | True | True | [220.36474756896496] |
| Fixed-output Stress | Fixed128 | OptimizedLayer | 1 | 8.5754 [CI 미확정], N=1 | 8.2178 [CI 미확정], N=1 | False | True | True | [46.38107493519783] |
| Fixed-output Stress | Fixed128 | OriginalHybrid | 1 | 8.6025 [CI 미확정], N=1 | 8.2266 [CI 미확정], N=1 | False | True | True | [46.43789865076542] |
| Fixed-output Stress | Fixed128 | OptimizedHybrid | 1 | 8.6062 [CI 미확정], N=1 | 8.2429 [CI 미확정], N=1 | False | True | True | [51.70099064707756] |
| Fixed-output Stress | Fixed192 | NativeStrong | 3 | 30.2918 [29.2914, 31.2367], N=3 | 12.1551 [12.0384, 12.2970], N=3 | False | False | True | [122.16385640203953, 120.19401229918003, 105.0635278224945] |
| Fixed-output Stress | Fixed192 | NativeCommon2 | 3 | 9.8152 [9.8058, 9.8264], N=3 | 8.9319 [8.9082, 8.9610], N=3 | True | True | True | [43.01971569657326, 69.99783031642437, 44.627366587519646] |
| Fixed-output Stress | Fixed192 | StockToken | 3 | 10.5877 [10.3134, 10.7992], N=3 | 9.9002 [9.6595, 10.0791], N=3 | False | True | True | [37.63442859053612, 90.46150930225849, 30.24345636367798] |
| Fixed-output Stress | Fixed192 | OriginalLayer | 3 | 8.5944 [8.5825, 8.6077], N=3 | 8.2502 [8.2243, 8.2695], N=3 | True | True | True | [47.417210415005684, 43.975261971354485, 43.13487559556961] |
| Fixed-output Stress | Fixed192 | OptimizedLayer | 3 | 8.5737 [8.5730, 8.5745], N=3 | 8.2629 [8.2616, 8.2644], N=3 | True | True | True | [49.784692004323006, 52.26055905222893, 52.63741686940193] |
| Fixed-output Stress | Fixed192 | OriginalHybrid | 3 | 8.6018 [8.5761, 8.6217], N=3 | 8.2522 [8.2486, 8.2549], N=3 | True | True | True | [95.14244459569454, 86.80446818470955, 87.53248117864132] |
| Fixed-output Stress | Fixed192 | OptimizedHybrid | 3 | 8.6177 [8.5458, 8.6947], N=3 | 8.2590 [8.2527, 8.2660], N=3 | True | True | True | [91.054642572999, 426.9907344132662, 103.07363234460354] |

Output16/64/128/192 Stress는 원문을 바꾸지 않습니다.
실제 Natural Output을 미리 안다고 가정하지 않습니다. Fixed-output 결과는 별도 Control입니다.
CI가 없는 Point를 반복 재현된 우위로 표현하지 않습니다.
실측하지 않은 QPS와 Output Cell은 보간하지 않습니다.

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

## 출력 동일성과 Task 품질

Token과 Layer 모두 Native와 다른 답변을 생성할 수 있습니다.
분할 자체가 필요한 계산을 생략하는 근사 기법이라는 뜻은 아닙니다.
BI0의 Batch·Shape·Compiler 변화가 수치 차이를 만들 수 있습니다.
그러나 확인되지 않은 구현 오류를 수치 차이로 단정하지 않습니다.

아래 표의 NativeStrong은 정답 Teacher가 아니라 같은 Arrival Start의 비교군입니다.
Output Token 전체 일치와 Task 점수를 별도로 검사합니다.
Task 점수 하락은 개별 요청에서 측정된 품질 저하입니다.
전체 평균 저하는 별도의 CI로 판단합니다. 낮아진 요청 수만으로 평균 저하를 판정하지 않습니다.
이 표의 CI는 Context-cluster Bootstrap입니다. 독립 Server Start CI가 아닙니다.
기존 Compiled C1 기준과2pp 비열등성 Gate는 변경하지 않습니다.
품질 Gate 통과는 모든 답변이 정답이거나 모든 토큰이 동일하다는 뜻이 아닙니다.

| Family | Start | Arm | Requests | Native와 토큰 전체 일치 | Task 점수 하락 | Task 점수 상승 | Task-macro Δpp | Context95% CI pp | Frozen C1 Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c3_credit_confirmation | 0 | NativeCommon2 | 192 | 144 | 16 | 16 | 0.0243 | [-0.290484747028488, 0.3011310554206477] | True |
| c3_credit_confirmation | 0 | StockToken | 192 | 147 | 15 | 17 | 0.0508 | [-0.25053172186501454, 0.3135425730584803] | True |
| c3_credit_confirmation | 0 | OriginalLayer | 192 | 138 | 24 | 16 | 0.3565 | [-0.370638893988821, 1.057823881880439] | True |
| c3_credit_confirmation | 0 | OptimizedLayer | 192 | 138 | 24 | 16 | 0.2725 | [-0.47915628895410967, 1.0009362397874486] | True |
| c3_credit_confirmation | 0 | OriginalHybrid | 192 | 137 | 22 | 17 | -0.2382 | [-1.0044400639121156, 0.32551308270425666] | True |
| c3_credit_confirmation | 0 | OptimizedHybrid | 192 | 136 | 23 | 16 | -0.1949 | [-0.9630901238281115, 0.36160936842928687] | True |
| c3_credit_confirmation | 1 | NativeCommon2 | 192 | 142 | 26 | 13 | 0.0787 | [-0.23354760137103425, 0.6683870245725307] | True |
| c3_credit_confirmation | 1 | StockToken | 192 | 139 | 27 | 14 | 0.6675 | [-0.22757005542808223, 2.206855929367031] | True |
| c3_credit_confirmation | 1 | OriginalLayer | 192 | 138 | 21 | 17 | 0.4153 | [-0.09633336504207264, 1.1049409626163567] | True |
| c3_credit_confirmation | 1 | OptimizedLayer | 192 | 140 | 18 | 17 | 0.4205 | [-0.09104794148632384, 1.1081245875626993] | True |
| c3_credit_confirmation | 1 | OriginalHybrid | 192 | 136 | 22 | 17 | -0.0833 | [-0.6751562484366743, 0.41803733484297234] | True |
| c3_credit_confirmation | 1 | OptimizedHybrid | 192 | 146 | 18 | 16 | 0.3673 | [-0.15155038524067904, 1.048770523186071] | True |
| c3_credit_confirmation | 2 | NativeCommon2 | 192 | 161 | 9 | 14 | -0.0910 | [-0.4418802658361178, 0.15913478651377905] | True |
| c3_credit_confirmation | 2 | StockToken | 192 | 142 | 14 | 21 | -0.3694 | [-1.0637409709924837, 0.14279570937277372] | True |
| c3_credit_confirmation | 2 | OriginalLayer | 192 | 137 | 18 | 17 | 0.0453 | [-0.4890451786465012, 0.34035566488295016] | True |
| c3_credit_confirmation | 2 | OptimizedLayer | 192 | 139 | 17 | 22 | -0.4676 | [-1.3106636474862297, 0.11199018168144617] | True |
| c3_credit_confirmation | 2 | OriginalHybrid | 192 | 137 | 16 | 20 | -0.4796 | [-1.3823289420870055, 0.14095375565233426] | True |
| c3_credit_confirmation | 2 | OptimizedHybrid | 192 | 140 | 18 | 18 | -0.5342 | [-1.4319297669190194, 0.08665288438872151] | True |
| p_geometry_confirmation | 0 | NativeCommon2 | 192 | 158 | 12 | 11 | 0.0838 | [-0.03045789067659502, 0.22590404083699753] | True |
| p_geometry_confirmation | 0 | StockToken | 192 | 138 | 21 | 17 | -0.1112 | [-0.8144836860456575, 0.4594402272949762] | True |
| p_geometry_confirmation | 0 | OriginalLayer | 192 | 133 | 24 | 17 | 0.1551 | [-0.4351696690852368, 0.6084307918912788] | True |
| p_geometry_confirmation | 0 | OptimizedLayer | 192 | 134 | 25 | 15 | -0.4373 | [-1.3757246155540555, 0.25592230771875235] | True |
| p_geometry_confirmation | 0 | OriginalHybrid | 192 | 141 | 19 | 16 | -0.1500 | [-0.9693651697250727, 0.43870637472192625] | True |
| p_geometry_confirmation | 0 | OptimizedHybrid | 192 | 137 | 17 | 22 | 0.2527 | [-0.31797201497490607, 0.6628325124886197] | True |
| p_geometry_confirmation | 1 | NativeCommon2 | 192 | 153 | 15 | 14 | 0.1346 | [-0.4045165456449694, 0.8152160279192243] | True |
| p_geometry_confirmation | 1 | StockToken | 192 | 141 | 19 | 19 | 0.0257 | [-0.5371505663748073, 0.6920108962288701] | True |
| p_geometry_confirmation | 1 | OriginalLayer | 192 | 135 | 19 | 22 | 0.1639 | [-0.5972730297962235, 0.8329295346124491] | True |
| p_geometry_confirmation | 1 | OptimizedLayer | 192 | 135 | 21 | 20 | 0.1506 | [-0.6083026854079631, 0.8243137201780368] | True |
| p_geometry_confirmation | 1 | OriginalHybrid | 192 | 130 | 21 | 23 | -0.0488 | [-0.9341723292467262, 0.6920872915384793] | True |
| p_geometry_confirmation | 1 | OptimizedHybrid | 192 | 134 | 22 | 19 | 0.1189 | [-0.6359115343307331, 0.7892674905735677] | True |
| p_geometry_confirmation | 2 | NativeCommon2 | 192 | 138 | 20 | 21 | 0.4502 | [0.08877136151818459, 1.0645593123235508] | True |
| p_geometry_confirmation | 2 | StockToken | 192 | 136 | 21 | 24 | 0.0126 | [-0.6612873845500358, 0.7292183847183328] | True |
| p_geometry_confirmation | 2 | OriginalLayer | 192 | 139 | 15 | 19 | 0.3272 | [0.021186206628975614, 0.7227924733537244] | True |
| p_geometry_confirmation | 2 | OptimizedLayer | 192 | 138 | 14 | 20 | 0.2697 | [-0.007568238078474908, 0.6475694007388985] | True |
| p_geometry_confirmation | 2 | OriginalHybrid | 192 | 140 | 12 | 23 | 0.1511 | [-0.44299481930981444, 0.6622471230292745] | True |
| p_geometry_confirmation | 2 | OptimizedHybrid | 192 | 137 | 13 | 23 | 0.2722 | [-0.001317723806422709, 0.6428719910056595] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | NativeCommon2 | 192 | 158 | 11 | 14 | -0.0881 | [-0.4958102536019086, 0.24634972398090943] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | StockToken | 192 | 145 | 16 | 18 | 0.3326 | [-0.8491435768322986, 1.9803635288311203] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | OriginalLayer | 192 | 140 | 15 | 23 | -0.0574 | [-0.6104416117928714, 0.242251165040208] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | OptimizedLayer | 192 | 138 | 17 | 22 | -0.0771 | [-0.6393215970245923, 0.22992566234450823] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | OriginalHybrid | 192 | 140 | 17 | 22 | 0.1306 | [-1.2083290387684982, 1.7212976192551954] | True |
| coordinator_hygiene_warm_regime_confirmation | 0 | OptimizedHybrid | 192 | 135 | 16 | 25 | -0.1459 | [-0.7503708608016324, 0.21150081454652783] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | NativeCommon2 | 192 | 153 | 12 | 16 | 0.1507 | [-0.07415036048877847, 0.49627098621949584] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | StockToken | 192 | 146 | 14 | 21 | 0.1417 | [-0.16453869451447523, 0.5214567117122904] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | OriginalLayer | 192 | 138 | 21 | 19 | 0.4738 | [-0.05767415472575391, 1.159166534334798] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | OptimizedLayer | 192 | 143 | 17 | 18 | 0.4192 | [-0.08528079795315087, 1.0828152799785409] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | OriginalHybrid | 192 | 137 | 19 | 19 | -0.0971 | [-0.6899159373423875, 0.3942566330437108] | True |
| coordinator_hygiene_warm_regime_confirmation | 1 | OptimizedHybrid | 192 | 139 | 18 | 20 | 0.4916 | [-0.041322879153313, 1.1790810021782245] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | NativeCommon2 | 192 | 155 | 15 | 12 | -0.1099 | [-0.475824042590127, 0.1345433204138121] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | StockToken | 192 | 146 | 15 | 15 | -0.3103 | [-1.0129006550005397, 0.25017742513015473] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | OriginalLayer | 192 | 140 | 19 | 15 | 0.0077 | [-0.522735171423465, 0.23059859032403915] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | OptimizedLayer | 192 | 139 | 20 | 15 | 0.0050 | [-0.5278355986328375, 0.2278745807119399] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | OriginalHybrid | 192 | 140 | 21 | 15 | -0.5186 | [-1.420969810578116, 0.09567329747075369] | True |
| coordinator_hygiene_warm_regime_confirmation | 2 | OptimizedHybrid | 192 | 137 | 22 | 18 | 0.0465 | [-0.4942322432060363, 0.32670842342443746] | True |

같은 Numerical Geometry의 Strict KV·Hidden·Greedy 검사가 실패한 새 구성은 Primary에서 제외합니다.
분산 Teacher를 단일 GPU Schema로 읽은 Missing-file 오류는 Reader 수정으로 분리했습니다.
Reader 수정 후 실제 Strict 불일치가 남으면 실패를 그대로 유지합니다.
Artifact의 Reduction 설정 차이는 관측 근거입니다. 이것만으로 원인을 확정하지 않습니다.
공유 Source 경로의 통제 실험은 기존 실패를 삭제하거나 정확도 기준을 완화하지 않습니다.

## 실제 실행 계약 — Primary Equal-output

| Arm | Start | Cut | Hybrid Fraction | Source Window | D Cap | P 최대 Request/Step | D 최대 Request/Step | P Compile | D Compile | P Graph | D Graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NativeStrong | 0 | None | None | 0 | None | 3 | 5 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| NativeCommon2 | 0 | None | None | 0 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| StockToken | 0 | 8 | None | 0 | 0 | 3 | 5 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalLayer | 0 | 2 | None | 16 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedLayer | 0 | 2 | None | 16 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalHybrid | 0 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedHybrid | 0 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| NativeStrong | 1 | None | None | 0 | None | 4 | 5 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| NativeCommon2 | 1 | None | None | 0 | 2 | 4 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| StockToken | 1 | 8 | None | 0 | 0 | 4 | 6 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalLayer | 1 | 2 | None | 16 | 2 | 4 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedLayer | 1 | 2 | None | 16 | 2 | 4 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalHybrid | 1 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedHybrid | 1 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| NativeStrong | 2 | None | None | 0 | None | 4 | 5 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| NativeCommon2 | 2 | None | None | 0 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| StockToken | 2 | 8 | None | 0 | 0 | 4 | 5 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalLayer | 2 | 2 | None | 16 | 2 | 4 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedLayer | 2 | 2 | None | 16 | 2 | 4 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OriginalHybrid | 2 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |
| OptimizedHybrid | 2 | 2 | 0.125 | 8 | 2 | 3 | 2 | 3 | 3 | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE |

D 최대 Request/Step은 Pure Decode Batch만을 뜻하지 않습니다.
Prefill/Decode 분리 Histogram은 Per-start CSV의 Actual Geometry에 있습니다.
Worker Config의 Compile Mode는 Partial Prefix Segment의 내부 컴파일 전체를 설명하지 않습니다.
Partial Prefix/Suffix의 별도 Compiler Flag와 Source Capsule도 함께 확인합니다.
Source Window0은 무제한 Coordinator Window입니다. KV나 Hidden Pool이 무제한이라는 뜻은 아닙니다.

## 새 R0 재현 Bridge

| Arm | Starts | QPS | Arrival Seed | GPU Pair | TTFT p95 ms | ITL p99 ms | TPOT p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NativeStrong | 1 | 2.0 | 2026112281 | [4, 5] | 5139.647 | 18.365 | 8.534 |
| NativeCommon2 | 1 | 2.0 | 2026112281 | [4, 5] | 6081.224 | 8.921 | 8.038 |
| NativeCommon0 | 1 | 2.0 | 2026112281 | [4, 5] | 5937.948 | 15.154 | 12.834 |
| Layer | 1 | 2.0 | 2026112281 | [4, 5] | 10621.337 | 8.433 | 7.940 |
| StockToken | 1 | 2.0 | 2026112281 | [4, 5] | 9039.334 | 9.046 | 8.297 |
| Hybrid | 1 | 2.0 | 2026112281 | [4, 5] | 11595.107 | 8.503 | 7.957 |

R0는 N1 진단 Bridge입니다. Primary N3와 합치지 않습니다.
원래 Historical N3의 Raw 재계산과 새 GPU 실행을 구분합니다.
