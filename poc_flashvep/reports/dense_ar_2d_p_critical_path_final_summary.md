# P-side Critical Path Root Cause & Optimization — 종합

과학적 성능 판정: `NO_GO_ON_TESTED_GRID`. 독립 N3 이상 비교의 Serving GO는 0/28개입니다.

P Critical Path의 원인은 `PARTIAL_ROOT_CAUSE_IDENTIFICATION`입니다.
GPU 계산 증가와 Queue 누적을 관측했습니다. 과거1.76초의 독점적 인과 비중은 완전히 식별하지 못했습니다.
Credit·Geometry·D Admission·Coordinator 개입을 각각 검증했습니다. 개입의 성공 여부는 아래 표에서 구분합니다.
미래 Decode-safe Window와 Online Dynamic Insertion은 `NOT_RUN`입니다.

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

## 질문별 판정 기준

1. 과거1.76초의 가법 Host 차이와 새 GPU Capture를 구분합니다.
   새 계측은 Kernel Resource Demand와 요청 경로의 Kernel 중첩을 따로 제공합니다.
   GPU 계산·Credit·Queue의 독점적인 인과 비중을 완전히 분리했다고 주장하지 않습니다.
2. 제거 가능한 Logical Credit 지연과 Coordinator 전달 지연을 각각 검사합니다.
   Attention Pair의 차이는 Scheduling만으로 제거하지 못합니다.
3. P 지연 감소는 Arrival→P-ready로 판정합니다.
   앞단 Queue의 위치만 바뀐 결과를 개선으로 계산하지 않습니다.
4. D의 이점은 같은 Geometry A/B와 별도 Serial Iso-work에서 검사합니다.
   Last2와 Stock WorkCut8의 D 시간 차이를 동일 GEMM 효율로 표현하지 않습니다.
5. C3 성공에는 Strong Stock 대비 TTFT CI 상한<1과 Decode 보호가 필요합니다.
   SLO-goodput의 분모와 CI도 보존합니다.
6. 긴 Output의 Cut은 동일 Input·Arrival을 사용한 Fixed-output Stress로 검사합니다.
   단일 Start로 보편적인 안전 Cut을 확정하지 않습니다.
7. Hybrid는 Attention·KV 의존성, Source Buffer와 D Residency의 제약을 받습니다.
   작은 D 비용이 작은 P Queue나 높은 SLO-goodput을 보장하지 않습니다.

## 현재 확인 표

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

## 재현성과 제한

원래 Checkout과 Staged Index를 보존합니다. 실패한 실험은 삭제하지 않습니다.
기존 Guard의 조기 Drain은 별도로 기록했습니다. 실행 중 GPU 작업은 재시작하지 않았습니다.
GPU 시작 전 차단된 동일 Config만 별도 Recovery Plan에서 재개했습니다.
Native 검사에서 누락된 State Capture Flag는 새 검사 Version으로 수정했습니다.
기존 Missing-file 실패를 Native 계산 오류나 성공으로 바꾸지 않습니다.

실제 Decode-safe Window와 미래 Deadline Predictor는 아직 별도 Gate가 필요합니다.
Kernel Inactivity만으로 안전한 Offloading Window를 주장하지 않습니다.
검증되지 않은 Dynamic Insertion은 NOT_RUN입니다.
Science 완료, Raw Metric 재계산, Figure 검사, Resource Closeout과 Remote SHA 검증은 별도로 확인합니다.

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
## 현재 과학적 판정

확인된 Serving GO Cell은 0/28개입니다.
Serving GO는 Strong Stock 대비 TTFT·SLO-goodput 우위와 품질·Decode Gate를 함께 요구합니다.
P GPU·Queue의 원인은 PARTIAL_ROOT_CAUSE_IDENTIFICATION입니다. 과거1.76초의 완전한 인과 분해는 아닙니다.
Dynamic Admission과 미래 Safe Window는 미검증입니다. 결과가 없는 방법을 성공으로 포함하지 않습니다.

## 질문별 실측 답변

### Q1. P 측 추가 지연 중 GPU와 Queue·Host의 비용은 얼마인가?

과거 약1.76초의 Host 차이는 완전한 GPU·Host 인과 분해가 아닙니다.
새 Equal-output N3에서 다음 가법 평균 차이를 측정했습니다.

| Historical Raw 재계산 Layer−Stock | Mean 차이 ms [Start-level95% CI] |
| --- | --- |
| Arrival→P-ready | 1755.2174 [1373.1020, 2078.7614], N=3 |
| P-ready→Client First Token | -796.7609 [-1070.9794, -468.3312], N=3 |
| Mean TTFT | 958.4566 [904.7708, 1007.7820], N=3 |

위 Historical N3는 원래9개 Raw 실행의 재계산입니다. 새 GPU N3 실행으로 표시하지 않습니다.


| Layer−Strong Stock | Mean 차이 ms [Start-level95% CI] |
| --- | --- |
| Arrival→P-ready | 2675.2577 [1282.8934, 4317.5930], N=3 |
| P-ready→Client First Token | -1388.3825 [-2173.6343, -661.4240], N=3 |
| Mean TTFT | 1286.8752 [621.4694, 2143.9587], N=3 |

P-ready 이전의 손해가 이후 경로의 절약보다 컸습니다.
이 표는 TTFT p95나 GPU-exclusive Time의 분해가 아닙니다.

| Axis−Stock | CPC-qualified Pair 수 | Kernel Union/N Δms/request | Arrival→P-ready Δms |
| --- | --- | --- | --- |
| Layer | 3 | 98.7010 [98.2897, 99.0887], N=3 | 1856.7112 [1442.9525, 2201.7898], N=3 |
| Hybrid | 3 | 36.8272 [36.5129, 37.1580], N=3 | 2271.4332 [1929.7293, 2603.8654], N=3 |

| P Kernel Category | Layer−Stock Δms/request [Start-level95% CI] |
| --- | --- |
| GEMM | 42.0152 [41.7772, 42.2532], N=3 |
| ATTENTION | 37.8404 [37.7272, 37.9249], N=3 |
| RMSNORM | 11.4794 [11.4419, 11.5196], N=3 |
| FUSED_RMSNORM_RESIDUAL | 1.7410 [1.7353, 1.7461], N=3 |
| ROPE | 6.1715 [6.1564, 6.1880], N=3 |
| UNKNOWN | -3.6805 [-3.6813, -3.6795], N=3 |

Kernel Union/N은 Resource Demand입니다. 대기 중 앞선 Request의 Kernel과 겹친 시간과 다릅니다.
이 비교는 Best Layer2와 Stock WorkCut8입니다. 같은 P/D GEMM Rows 비교가 아닙니다.
따라서 이 표의 GEMM 차이를 같은 Rows의 Shape 효율 차이로 단정하지 않습니다.
현재 Best-vs-best에서 Attention만이 유일한 추가 비용은 아닙니다.
관측된 GPU 계산 증가와 Queue 누적은 Compute-only 또는 Host-only 설명을 구분하는 근거입니다.
그러나 두 원인의 독점적인 인과 비중은 ROOT_CAUSE_NOT_FULLY_IDENTIFIED입니다.

### Q2. 실제로 제거 가능한 비용은 무엇인가?

필요한 ACK와 GPU Fence 이후에도 유지된 Logical Source Credit은 조기에 반환할 수 있었습니다.
별도 진단에서 Evidence 파일 쓰기와 Coordinator Cyclic GC가 Token 전달을 지연했습니다.
이 진단은 원래1.85초 Client Tail 전체의 원인을 확정하지 않습니다.
같은 GEMM Rows의 P Attention Pair 차이는 Host Scheduling만으로 제거할 수 없습니다.

### Q3. Intervention이 P 경로를 얼마나 줄였는가?

Safe-retirement A/B는 Credit 상한과 D 정책을 유지했습니다.
| Metric | Optimized/Original 또는 차이 [95% CI] |
| --- | --- |
| Layer P 경로 Ratio | 0.9839 [0.9764, 0.9876], N=3 |
| Layer P 경로 Δms | -55.3890 [-71.0644, -36.0688], N=3 |
| Layer TTFT p95 Ratio | 0.9846 [0.9658, 1.0037], N=3 |

Last2→Last8은 실제 P 작업을 D로 옮긴 별도 Geometry Intervention입니다.
| Metric | Last8/Last2 [95% CI] |
| --- | --- |
| P 경로 Ratio | 0.6057 [0.4528, 0.7820], N=3 |
| TTFT p95 Ratio | 1.0772 [1.0289, 1.1298], N=3 |

P 경로가 줄었다는 사실만으로 전체 Serving 개선을 판정하지 않습니다.

### Q4. D 측 Layer 이점은 유지되는가?

Serial 실제 Serving API에서 P와 D-first Kernel을 각각 계측했습니다.
D-first에는 Continuation과 첫 출력 처리가 포함됩니다. Suffix-only 비용은 아닙니다.
Best Layer2와 Stock WorkCut8은 D 작업량이 다릅니다.
별도의 Last8/Stock8/Hybrid Iso-work Control은 실제 Rows와 CPC Gate를 따로 제공합니다.
| Serial 비교 | Qualified Starts | P Kernel Δms | D-first Kernel Δms | P+D-first Kernel Δms | D Rows 상대 오차 |
| --- | --- | --- | --- | --- | --- |
| Layer8/StockToken | 1 | 15.4314 [CI 미확정], N=1 | -7.3930 [CI 미확정], N=1 | 8.0384 [CI 미확정], N=1 | [-3.867942492621156e-05] |
| HybridIso8Approx/StockToken | 1 | 8.1057 [CI 미확정], N=1 | 1.5285 [CI 미확정], N=1 | 9.6342 [CI 미확정], N=1 | [-0.0006047379320194231] |

위 표의 N1은 통계적으로 확정한 우위가 아닙니다.32개 Request를32개 독립 Start로 취급하지 않습니다.
Natural C3의 D Operator 비중은 Serial 결과로 대체하지 않습니다.

### Q5. Strong Stock보다 C3 TTFT와 Goodput이 좋아졌는가?

| Family | Workload | Arm/Stock | TTFT p95 Ratio [CI] | SLO-goodput Ratio [CI] | Decode Gate | Cell 판정 | 동일 Output Work |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Credit Retirement | EqualReplay | OriginalLayer | 1.2295 [1.1831, 1.2635], N=3 | 0.8509 [0.7158, 0.9343], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Credit Retirement | EqualReplay | OptimizedLayer | 1.2106 [1.1732, 1.2505], N=3 | 0.8591 [0.7174, 0.9493], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Credit Retirement | EqualReplay | OriginalHybrid | 1.4174 [1.2440, 1.5599], N=3 | 0.6998 [0.3961, 0.9072], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Credit Retirement | EqualReplay | OptimizedHybrid | 1.4414 [1.2399, 1.6228], N=3 | 0.6831 [0.3855, 0.8912], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Credit Retirement | Natural | OriginalLayer | 0.7897 [0.6023, 0.9441], N=3 | 0.7178 [0.4137, 0.9008], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Credit Retirement | Natural | OptimizedLayer | 0.7991 [0.6333, 0.9248], N=3 | 0.8309 [0.6614, 0.9303], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Credit Retirement | Natural | OriginalHybrid | 0.7806 [0.5136, 1.0412], N=3 | 0.6643 [0.3513, 0.9424], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Credit Retirement | Natural | OptimizedHybrid | 0.8986 [0.7216, 1.0477], N=3 | 0.7197 [0.3842, 0.9471], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Partition Geometry | EqualReplay | OriginalLayer | 1.1722 [1.1425, 1.2036], N=3 | 0.8461 [0.7178, 0.9503], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Partition Geometry | EqualReplay | OptimizedLayer | 1.2627 [1.2221, 1.2972], N=3 | 0.7558 [0.6845, 0.8585], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Partition Geometry | EqualReplay | OriginalHybrid | 1.2080 [1.1546, 1.2491], N=3 | 0.7366 [0.6156, 0.8574], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Partition Geometry | EqualReplay | OptimizedHybrid | 1.2770 [1.2165, 1.3408], N=3 | 0.5925 [0.4340, 0.7876], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Partition Geometry | Natural | OriginalLayer | 0.8772 [0.6029, 1.5427], N=3 | 0.8123 [0.6756, 0.9354], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Partition Geometry | Natural | OptimizedLayer | 1.0658 [0.7655, 1.7806], N=3 | 0.7514 [0.5818, 0.9180], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Partition Geometry | Natural | OriginalHybrid | 1.0595 [0.7880, 1.7562], N=3 | 0.7570 [0.6146, 0.8846], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Partition Geometry | Natural | OptimizedHybrid | 1.2106 [0.9651, 1.8816], N=3 | 0.6663 [0.6375, 0.7067], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Warm-regime Coordinator | EqualReplay | OriginalLayer | 1.1441 [1.1137, 1.1737], N=3 | 0.7571 [0.6501, 0.8743], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Warm-regime Coordinator | EqualReplay | OptimizedLayer | 1.2664 [1.2207, 1.3020], N=3 | 0.7243 [0.5954, 0.8532], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Warm-regime Coordinator | EqualReplay | OriginalHybrid | 1.1759 [1.1240, 1.2146], N=3 | 0.6756 [0.6261, 0.7324], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Warm-regime Coordinator | EqualReplay | OptimizedHybrid | 1.2833 [1.2430, 1.3146], N=3 | 0.6482 [0.5223, 0.8040], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Warm-regime Coordinator | Natural | OriginalLayer | 0.9013 [0.6682, 1.4047], N=3 | 0.7209 [0.6177, 0.8116], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Warm-regime Coordinator | Natural | OptimizedLayer | 0.9760 [0.7271, 1.4921], N=3 | 0.6096 [0.5377, 0.6938], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Warm-regime Coordinator | Natural | OriginalHybrid | 0.9037 [0.6810, 1.4181], N=3 | 0.6399 [0.5472, 0.7275], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Warm-regime Coordinator | Natural | OptimizedHybrid | 1.1781 [0.7804, 2.1609], N=3 | 0.4659 [0.2585, 0.6628], N=3 | True | NO_GO_ON_THIS_CELL | False |
| Output192 D cap2→4 | Fixed192 | OriginalLayer | 3.0617 [CI 미확정], N=1 | 0.3937 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Output192 D cap2→4 | Fixed192 | OptimizedLayer | 3.1387 [CI 미확정], N=1 | 0.3876 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Output192 D cap2→4 | Fixed192 | OriginalHybrid | 3.0929 [CI 미확정], N=1 | 0.5461 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Output192 D cap2→4 | Fixed192 | OptimizedHybrid | 2.8212 [CI 미확정], N=1 | 0.5800 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed16 | OriginalLayer | 2.1594 [CI 미확정], N=1 | 0.9841 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed16 | OptimizedLayer | 1.2090 [CI 미확정], N=1 | 0.9995 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed16 | OriginalHybrid | 1.2641 [CI 미확정], N=1 | 0.9997 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed16 | OptimizedHybrid | 1.1190 [CI 미확정], N=1 | 0.9997 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed64 | OriginalLayer | 1.6312 [CI 미확정], N=1 | 0.5047 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed64 | OptimizedLayer | 3.6541 [CI 미확정], N=1 | 0.1270 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed64 | OriginalHybrid | 3.4077 [CI 미확정], N=1 | 0.1650 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed64 | OptimizedHybrid | 4.2994 [CI 미확정], N=1 | 0.1185 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed128 | OriginalLayer | 3.5044 [CI 미확정], N=1 | 0.6314 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed128 | OptimizedLayer | 3.9994 [CI 미확정], N=1 | 0.5910 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed128 | OriginalHybrid | 4.0370 [CI 미확정], N=1 | 0.5883 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed128 | OptimizedHybrid | 4.0021 [CI 미확정], N=1 | 0.5247 [CI 미확정], N=1 | False | HOLD_INSUFFICIENT_STARTS | True |
| Fixed-output Stress | Fixed192 | OriginalLayer | 2.9005 [2.8239, 2.9806], N=3 | 0.4251 [0.3362, 0.5406], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Fixed-output Stress | Fixed192 | OptimizedLayer | 3.1487 [3.0846, 3.1991], N=3 | 0.3328 [0.3077, 0.3720], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Fixed-output Stress | Fixed192 | OriginalHybrid | 2.9720 [2.8737, 3.1114], N=3 | 0.4653 [0.4155, 0.5314], N=3 | True | NO_GO_ON_THIS_CELL | True |
| Fixed-output Stress | Fixed192 | OptimizedHybrid | 3.0319 [3.0085, 3.0542], N=3 | 0.4624 [0.4130, 0.5199], N=3 | True | NO_GO_ON_THIS_CELL | True |

Natural EOS에서 TTFT 우위와 Goodput 열세가 함께 나타날 수 있습니다.
해당 TTFT 우위를 삭제하지 않습니다. 서로 다른 Output Work의 효과를 Compute Speedup으로 표현하지 않습니다.
Equal-output Replay와 Natural EOS를 별도로 판정합니다.

### Q6. 긴 Output에서 어떤 Cut이 Decode를 보호하는가?

Fixed16/64/128은 각각 N1입니다. Fixed192는 사전 지정한 N3입니다.
실측된 두 Native-relative Gate와 Absolute95% Gate를 모두 확인합니다.
N1 Point가 안전해 보여도 반복 검증된 보편적 Cut으로 제시하지 않습니다.
| Fixed192 Arm | Starts | ITL p99 ms [CI] | TPOT p95 ms [CI] | Relative5% | Absolute95% | Start별 Max ITL ms |
| --- | --- | --- | --- | --- | --- | --- |
| NativeStrong | 3 | 30.2918 [29.2914, 31.2367], N=3 | 12.1551 [12.0384, 12.2970], N=3 | False | False | [122.16385640203953, 120.19401229918003, 105.0635278224945] |
| NativeCommon2 | 3 | 9.8152 [9.8058, 9.8264], N=3 | 8.9319 [8.9082, 8.9610], N=3 | True | True | [43.01971569657326, 69.99783031642437, 44.627366587519646] |
| StockToken | 3 | 10.5877 [10.3134, 10.7992], N=3 | 9.9002 [9.6595, 10.0791], N=3 | False | True | [37.63442859053612, 90.46150930225849, 30.24345636367798] |
| OriginalLayer | 3 | 8.5944 [8.5825, 8.6077], N=3 | 8.2502 [8.2243, 8.2695], N=3 | True | True | [47.417210415005684, 43.975261971354485, 43.13487559556961] |
| OptimizedLayer | 3 | 8.5737 [8.5730, 8.5745], N=3 | 8.2629 [8.2616, 8.2644], N=3 | True | True | [49.784692004323006, 52.26055905222893, 52.63741686940193] |
| OriginalHybrid | 3 | 8.6018 [8.5761, 8.6217], N=3 | 8.2522 [8.2486, 8.2549], N=3 | True | True | [95.14244459569454, 86.80446818470955, 87.53248117864132] |
| OptimizedHybrid | 3 | 8.6177 [8.5458, 8.6947], N=3 | 8.2590 [8.2527, 8.2660], N=3 | True | True | [91.054642572999, 426.9907344132662, 103.07363234460354] |

현재 Fixed192 N3 Decode Gate 통과 Arm: NativeCommon2, OriginalLayer, OptimizedLayer, OriginalHybrid, OptimizedHybrid.
이 목록은 QPS2와 해당 실행 정책에 한정됩니다. TTFT·Goodput 최적 Cut 목록이 아닙니다.
ITL p99가 통과해도 큰 Max ITL이 남을 수 있습니다. Max ITL과 모든-ITL 통과율을 함께 보존합니다.

### Q7. 남은 실행 한계는 무엇인가?

P Attention 증가, D Residency, Handoff와 Hybrid DAG 의존성이 남습니다.
P RMSNorm Provider Probe는 Strict KV·Hidden Gate에 실패했습니다. 성능 Primary에는 포함하지 않습니다.
같은 Probe의 Greedy8/8 일치는 State 불일치를 없애지 않습니다.
공유 Source 경로의 Stock Compiler Counterfactual은 기존 Strict 실패와 별도로 기록합니다.
새 Online Geometry 확인은 기존 Best Last2와 Cost 기반 Last8 후보에 한정합니다.
Last4·Last12의 전체 QPS 재탐색은 NOT_RUN입니다. 새 Input×Output×Cut 전수 Map을 만들었다고 주장하지 않습니다.
추가 Low·Knee·High QPS 확인은 유망한 Serving 후보의 후속 조건입니다. 해당 개선 Gate를 통과하지 못하면 NOT_RUN으로 남깁니다.
미래 Decode-safe Window는 검증하지 못했습니다. Active-decode Dynamic Insertion은 NOT_RUN입니다.
Attention-head Offloading PoC는 별도 Operator-parallel 실험입니다. 기존 Serving 성공을 대체하지 않습니다.

## 같은 Geometry의 D Admission 개입

D Cap2→4는 Source Credit과 Geometry를 유지한 별도 One-factor Discovery입니다.
| Candidate/동일 Axis Cap2 | TTFT p95 Ratio | Point Decode Screen | Absolute Decode 통과율 | 선정 |
| --- | --- | --- | --- | --- |
| Layer8Cap4/OriginalLayer | 1.005049 | True | 0.990385 | False |
| HybridIsoCap4/OriginalHybrid | 1.026600 | True | 1.000000 | False |

각 Point는 N1입니다. CI 기반 Decode 보호나 우위로 표시하지 않습니다.
사전 Screen을 통과하지 못한 후보는 N3로 확장하지 않습니다.9개 Discovery Arm의 결과는 모두 보존합니다.

## Compiler Cache와 정합성 반증

Source 경로만 고정한 검사는 공식 Cache Root를 통일하지 못했습니다.
NIXL Port를 고정하자 P와 D의 공식 Cache Root는 각각 일치했습니다.
그러나 최초 컴파일 실행과 이후 캐시 재사용 실행의 KV·Hidden은 일치하지 않았습니다.
| 기존 Raw 비교 | 사전 계획 | Requests | KV Exact | Hidden Exact | Greedy Exact | Strict PASS |
| --- | --- | --- | --- | --- | --- | --- |
| PC_FixedPort_StockToken_NormalGC_Strict8/PC_FixedPort_StockToken_DeferredGC_Strict8 | True | 8 | 0 | 0 | 8 | False |
| PC_FixedPort_StockToken_NormalGC_Strict8/PC_FixedPort_StockToken_NormalGCRepeat_Strict8 | True | 8 | 0 | 0 | 8 | False |
| PC_FixedPort_StockToken_DeferredGC_Strict8/PC_FixedPort_StockToken_NormalGCRepeat_Strict8 | False | 8 | 8 | 8 | 8 | True |

뒤의 두 Warm 실행 비교는 사후 CPU 진단입니다. 기존 실패 Gate를 이 비교로 덮어쓰지 않습니다.
사전에 계획한 새 Warm-only3회 검사의 Strict PASS: True.
이는 CORRECTNESS_ONLY입니다. 책임 Operator나 N3 Serving 이득을 단독으로 증명하지 않습니다.
새 Runtime은 별도의8→32→400 Gate와 실제 Warm 실행 경로 검사를 요구합니다.
기존 V6 실패와 미실행 성능 Cell은 그대로 유지합니다.

## 추가 Attention-head Offloading PoC

실제 Q/K/V와 Paged KV를 사용한 독립 Benchmark Process3개를 측정했습니다.
D에는 Active Decode가 없습니다. 전체 모델과 Serving KV Budget의 공존 비용은 재생하지 않았습니다.
| Tensor 비교 | 관측 수 | Exact 불일치 수 | 최대 절대 오차 |
| --- | --- | --- | --- |
| full_vs_captured | 216 | 0 | 0.0 |
| split_vs_full | 216 | 48 | 0.001953125 |
| remote_vs_local_split | 216 | 0 | 0.0 |
| cached_vs_local_split | 216 | 0 | 0.0 |

| Input | Layer 호출 수 | Remote 경로 | Remote KV Heads | Remote/LocalFull Wall p50 [CI] |
| --- | --- | --- | --- | --- |
| 8181 | 4 | RemoteSplit | 4/8 | 0.9498 [0.9359, 0.9650], N=3 |
| 8181 | 4 | RemoteCachedSplit | 4/8 | 0.9463 [0.9329, 0.9608], N=3 |

위 CI의 N은 Benchmark Process 수입니다. 독립 Serving Start 수가 아닙니다.
표의 Layer 수는 Attention 호출 열입니다. 전체 Transformer Layer 실행시간이 아닙니다.
GEMM·FFN·다음 Hidden 준비·Native Decode 보호는 포함하지 않습니다.
따라서 약5%의 고립 Attention 비용 이득을 P 전체 시간이나 Online TTFT 개선으로 확대하지 않습니다.
전체 모델 Greedy·Task 품질·Online SLO-goodput은 NOT_RUN입니다.

## 새 Common Warm C3 확인

공식 Graph와 AOT Cache를 재사용한 실행을 별도 Family로 확인합니다.
전체 Backbone과 Partial Prefix·Interval의 모든 Cache Region을 검사합니다.
V8의 단일 Cache Region Reader 오류는 CPU에서 수정했습니다.
V9는 완료된 GPU 결과의 Config·Source·Receipt Hash를 검사한 뒤 재사용했습니다.
기존 Reader 실패와 Cold/Warm State 불일치는 삭제하지 않았습니다.
| Arm | 구성 Gate | Natural400 | 품질 Gate | 오류 |
| --- | --- | --- | --- | --- |
| NativeStrong | True | True | True | 없음 |
| NativeCommon2 | True | True | True | 없음 |
| StockToken | True | True | True | 없음 |
| Hybrid | True | True | True | 없음 |
| HybridIso8Approx | True | True | True | 없음 |
| HybridGCControl | True | True | True | 없음 |
| Layer | True | True | True | 없음 |
| Layer8 | True | True | True | 없음 |

Strict Control의 Controlled Pipeline과 실제 C3 Pipeline은 다릅니다.
V10은 실제 C3 Source Window와 Pipeline으로 Cache를 먼저 준비합니다.
Priming8은 정합성 검사입니다. 추가 독립 성능 Start가 아닙니다.
세 독립 Start는 각각 GPU0/1, GPU2/3, GPU6/7을 사용합니다.
각 Paired 비교는 같은 Pair와 Arrival Trace를 사용합니다.
공유 Host와 병렬 Pair의 간섭을 제거했다고 주장하지 않습니다.
이 Family의 Optimized Label은 Partition Geometry를 뜻합니다. Early Credit Retirement가 아닙니다.
앞선 Credit·Geometry Family와 Start를 합치지 않습니다.
Heldout192 Request는 앞선 확인에도 사용했습니다. 새로운 Request 일반화 검사는 아닙니다.
서로 다른 Arrival Seed의 Family 차이만으로 Coordinator 개선율을 추정하지 않습니다.
Equal-output과 Natural EOS의 실행 여부는 Stage별 완료 기록으로 확인합니다.
| Workload | Arm | 독립 Starts | Runtime 경로 Gate | 품질 Gate |
| --- | --- | --- | --- | --- |
| EqualReplay | NativeStrong | 3 | True | True |
| EqualReplay | NativeCommon2 | 3 | True | True |
| EqualReplay | StockToken | 3 | True | True |
| EqualReplay | OriginalLayer | 3 | True | True |
| EqualReplay | OptimizedLayer | 3 | True | True |
| EqualReplay | OriginalHybrid | 3 | True | True |
| EqualReplay | OptimizedHybrid | 3 | True | True |
| EqualReplay | HybridGCControl | 3 | True | True |
| Natural | NativeStrong | 3 | True | True |
| Natural | NativeCommon2 | 3 | True | True |
| Natural | StockToken | 3 | True | True |
| Natural | OriginalLayer | 3 | True | True |
| Natural | OptimizedLayer | 3 | True | True |
| Natural | OriginalHybrid | 3 | True | True |
| Natural | OptimizedHybrid | 3 | True | True |

### Coordinator Cyclic GC의 독립 개입

두 Hybrid는 같은 Worker Source·Geometry·Source Window·D 정책·Buffered I/O를 사용합니다.
측정 구간의 Coordinator Cyclic GC 유예 여부만 다릅니다. Worker GC와 Reference Counting은 바꾸지 않습니다.
| GC 유예/Normal GC | Paired Ratio [Start-level95% CI] |
| --- | --- |
| P Arrival→Ready | 0.9486 [0.9321, 0.9743], N=3 |
| TTFT p95 | 0.9856 [0.9811, 0.9910], N=3 |
| SLO-goodput | 1.0413 [1.0117, 1.0811], N=3 |
| ITL p99 | 0.9973 [0.9941, 0.9997], N=3 |

이 비교는 Equal-output N3입니다. Strong Stock 대비 우위와는 별개입니다.
개입 전후의 실제 Output Work·Runtime 경로·품질·두 Native Decode Gate도 별도로 검사합니다.
이 결과를 원래1.85초 Tail 전체의 원인이나 Native·Stock의 동일 개선율로 확대하지 않습니다.

Natural Stage의 추가 실행은 남은 시간으로 결정합니다. 성능 승패를 선정 기준으로 사용하지 않습니다.
절약한 시간은 기존 최대10시간 안에서 재배분했습니다. 최종 GPU·Git Closeout 시간은 별도로 확보합니다.

## 긴 Decode Residency의 D Admission 개입

Output192에서 D Cap2→4를 별도로 측정했습니다.
이 Family의 OriginalLayer와 OptimizedLayer는 모두 Last8입니다. 각각 D Cap2와4를 사용합니다.
OriginalHybrid와 OptimizedHybrid는 모두 Last2+Suffix3/17입니다. 각각 D Cap2와4를 사용합니다.
같은 Axis의 Source Credit·Geometry·Compiler·Arrival·출력량을 유지합니다.
이 실험은 기존 Runtime Family입니다. 새 Warm C3 결과와 합치지 않습니다.
| Arm | Starts | TTFT p95 ms | ITL p99 ms | TPOT p95 ms |
| --- | --- | --- | --- | --- |
| NativeStrong | 1 | 5731.3952 [CI 미확정], N=1 | 21.9086 [CI 미확정], N=1 | 11.7687 [CI 미확정], N=1 |
| NativeCommon2 | 1 | 64266.4202 [CI 미확정], N=1 | 9.8109 [CI 미확정], N=1 | 8.9212 [CI 미확정], N=1 |
| NativeCommon4 | 1 | 15003.4684 [CI 미확정], N=1 | 12.4022 [CI 미확정], N=1 | 11.5624 [CI 미확정], N=1 |
| StockToken | 1 | 54350.9813 [CI 미확정], N=1 | 9.9482 [CI 미확정], N=1 | 9.5459 [CI 미확정], N=1 |
| OriginalLayer | 1 | 166404.5714 [CI 미확정], N=1 | 8.5919 [CI 미확정], N=1 | 8.2493 [CI 미확정], N=1 |
| OptimizedLayer | 1 | 170593.3308 [CI 미확정], N=1 | 8.5512 [CI 미확정], N=1 | 8.2461 [CI 미확정], N=1 |
| OriginalHybrid | 1 | 168100.0853 [CI 미확정], N=1 | 8.5345 [CI 미확정], N=1 | 8.2669 [CI 미확정], N=1 |
| OptimizedHybrid | 1 | 153333.1145 [CI 미확정], N=1 | 8.8468 [CI 미확정], N=1 | 8.4920 [CI 미확정], N=1 |

N1은 반복 확인된 우위나 Native-relative5% 비열등성을 증명하지 않습니다.
짧은 Replay의 Cap4 Screen 실패를 긴 Output의 결과로 대체하지 않습니다.

### 설정한 D 상한과 실제 Decode Batch

| Arm | 설정 D Cap | 실제 Scheduler 상한 | Pure Decode 최대 Batch | Step 가중 평균 Batch | Registration 유예 요청 수 | 조건부 Registration 대기 Mean ms |
| --- | --- | --- | --- | --- | --- | --- |
| NativeStrong | None | 1024 | 14 | 4.1189 | 정책 미사용 | NOT_APPLICABLE |
| NativeCommon2 | 2 | 2 | 2 | 1.9754 | 184 | 4454.408 |
| NativeCommon4 | 4 | 4 | 4 | 3.6125 | 184 | 2482.949 |
| StockToken | 0 | 1024 | 5 | 2.5946 | 188 | 974.811 |
| OriginalLayer | 2 | 2 | 2 | 1.1294 | 188 | 989.964 |
| OptimizedLayer | 4 | 4 | 2 | 1.1163 | 191 | 976.550 |
| OriginalHybrid | 2 | 2 | 2 | 1.1429 | 191 | 1049.652 |
| OptimizedHybrid | 4 | 4 | 3 | 1.2229 | 190 | 1018.639 |

설정 상한4를 실제 Decode Batch4로 취급하지 않습니다.
Registration 정책은 Output을 시작한 Running Request가 남아 있으면 새 등록을 유예합니다.
이 정책과 Decode-first Scheduling 아래에서 실제 Batch가 어떻게 형성됐는지 확인합니다.
상한을 높여도 Batch가 커지지 않은 결과는 GPU Compute 포화의 인과적 증거가 아닙니다.
Registration 대기는 Host 구간입니다. GPU Kernel 실행시간이나 Physical Idle로 표현하지 않습니다.
등록 Cohort를 바꾸는 추가 GPU 개입은 NOT_RUN입니다. 미래 Decode-safe Window를 검증하지 않았습니다.

## GPU 사용 예산

| 항목 | 관측값 |
| --- | --- |
| GPU Dispatch 기록 수 | 435 |
| 시작부터 마지막 GPU Dispatch 종료까지 hours | 8.6454 |
| 예약 GPU Process-hour 합계 | 57.5222 |
| GPU별 Interval Union의 합계 GPU-hours | 57.5222 |
| 승인한 최대 Wall-clock hours | 10 |

실패한 Startup과 정합성 검사도 포함합니다. GPU별 중첩 구간은 Union으로 한 번만 계산합니다.
이 값은 예약한 GPU 집합과 Dispatcher 경과시간의 곱입니다. GPU-exclusive Kernel 시간이 아닙니다.
Attention Capture는 실제 계산 GPU가1개여도2개를 예약한 비용으로 계산합니다.
Burner 시간과 CPU-only 분석·Archive·Push 시간은 이 GPU Dispatch 예산에 포함하지 않습니다.

## Offered QPS와 순간 도착 집중

| Workload | Start | Offered QPS | 실현된 간격 기준 requests/s | 최소 간격 ms | 직전1초 최대 도착 수 | 동일 Trace |
| --- | --- | --- | --- | --- | --- | --- |
| EqualReplay | 0 | 2.0 | 2.0699 | 1.838 | 7 | True |
| EqualReplay | 1 | 2.0 | 2.6144 | 2.940 | 9 | True |
| EqualReplay | 2 | 2.0 | 1.9466 | 3.019 | 7 | True |
| Natural | 0 | 2.0 | 2.0699 | 1.838 | 7 | True |
| Natural | 1 | 2.0 | 2.6144 | 2.940 | 9 | True |
| Natural | 2 | 2.0 | 1.9466 | 3.019 | 7 | True |

QPS2는 평균 도착률입니다. 동시에2개를 넣는 Batch 설정이 아닙니다.
Poisson Trace에는 짧은 간격의 집중 도착이 있습니다.
느린 P Service는 이런 구간의 Queue를 더 오래 유지할 수 있습니다.
그러나 Arrival Burst만 바꾼 Intervention은 실행하지 않았습니다.
이 표와 Request별 CSV는 도착 집중의 인과적 기여율을 증명하지 않습니다.

## Figure와 대응 수치

| Figure | PNG | PDF | CSV |
| --- | --- | --- | --- |
| p_queue_kernel_overlap | [figures/p_queue_kernel_overlap.png](dense_ar_p_critical_path_20261123/figures/p_queue_kernel_overlap.png) | [figures/p_queue_kernel_overlap.pdf](dense_ar_p_critical_path_20261123/figures/p_queue_kernel_overlap.pdf) | [figures/p_queue_kernel_overlap.csv](dense_ar_p_critical_path_20261123/figures/p_queue_kernel_overlap.csv) |
| p_gpu_operator_resource_demand | [figures/p_gpu_operator_resource_demand.png](dense_ar_p_critical_path_20261123/figures/p_gpu_operator_resource_demand.png) | [figures/p_gpu_operator_resource_demand.pdf](dense_ar_p_critical_path_20261123/figures/p_gpu_operator_resource_demand.pdf) | [figures/p_gpu_operator_resource_demand.csv](dense_ar_p_critical_path_20261123/figures/p_gpu_operator_resource_demand.csv) |
| p_request_critical_path_gantt | [figures/p_request_critical_path_gantt.png](dense_ar_p_critical_path_20261123/figures/p_request_critical_path_gantt.png) | [figures/p_request_critical_path_gantt.pdf](dense_ar_p_critical_path_20261123/figures/p_request_critical_path_gantt.pdf) | [figures/p_request_critical_path_gantt.csv](dense_ar_p_critical_path_20261123/figures/p_request_critical_path_gantt.csv) |
| p_input_output_bottleneck_map | [figures/p_input_output_bottleneck_map.png](dense_ar_p_critical_path_20261123/figures/p_input_output_bottleneck_map.png) | [figures/p_input_output_bottleneck_map.pdf](dense_ar_p_critical_path_20261123/figures/p_input_output_bottleneck_map.pdf) | [figures/p_input_output_bottleneck_map.csv](dense_ar_p_critical_path_20261123/figures/p_input_output_bottleneck_map.csv) |
| credit_retirement_equalreplay_ttft_goodput | [figures/credit_retirement_equalreplay_ttft_goodput.png](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_ttft_goodput.png) | [figures/credit_retirement_equalreplay_ttft_goodput.pdf](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_ttft_goodput.pdf) | [figures/credit_retirement_equalreplay_ttft_goodput.csv](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_ttft_goodput.csv) |
| credit_retirement_equalreplay_pd_host_stages | [figures/credit_retirement_equalreplay_pd_host_stages.png](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_pd_host_stages.png) | [figures/credit_retirement_equalreplay_pd_host_stages.pdf](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_pd_host_stages.pdf) | [figures/credit_retirement_equalreplay_pd_host_stages.csv](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_pd_host_stages.csv) |
| credit_retirement_equalreplay_stock_ratios | [figures/credit_retirement_equalreplay_stock_ratios.png](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_stock_ratios.png) | [figures/credit_retirement_equalreplay_stock_ratios.pdf](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_stock_ratios.pdf) | [figures/credit_retirement_equalreplay_stock_ratios.csv](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_stock_ratios.csv) |
| credit_retirement_equalreplay_decode_protection | [figures/credit_retirement_equalreplay_decode_protection.png](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_decode_protection.png) | [figures/credit_retirement_equalreplay_decode_protection.pdf](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_decode_protection.pdf) | [figures/credit_retirement_equalreplay_decode_protection.csv](dense_ar_p_critical_path_20261123/figures/credit_retirement_equalreplay_decode_protection.csv) |
| credit_retirement_p_intervention | [figures/credit_retirement_p_intervention.png](dense_ar_p_critical_path_20261123/figures/credit_retirement_p_intervention.png) | [figures/credit_retirement_p_intervention.pdf](dense_ar_p_critical_path_20261123/figures/credit_retirement_p_intervention.pdf) | [figures/credit_retirement_p_intervention.csv](dense_ar_p_critical_path_20261123/figures/credit_retirement_p_intervention.csv) |
| fixed_output_residency_host_paths | [figures/fixed_output_residency_host_paths.png](dense_ar_p_critical_path_20261123/figures/fixed_output_residency_host_paths.png) | [figures/fixed_output_residency_host_paths.pdf](dense_ar_p_critical_path_20261123/figures/fixed_output_residency_host_paths.pdf) | [figures/fixed_output_residency_host_paths.csv](dense_ar_p_critical_path_20261123/figures/fixed_output_residency_host_paths.csv) |
| coordinator_hygiene_warm_regime_equalreplay_ttft_goodput | [figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.png](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.png) | [figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.pdf](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.pdf) | [figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.csv](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_ttft_goodput.csv) |
| coordinator_hygiene_warm_regime_equalreplay_decode_protection | [figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.png](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.png) | [figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.pdf](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.pdf) | [figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.csv](dense_ar_p_critical_path_20261123/figures/coordinator_hygiene_warm_regime_equalreplay_decode_protection.csv) |
| output192_d_cap_intervention_fixed192_ttft_goodput | [figures/output192_d_cap_intervention_fixed192_ttft_goodput.png](dense_ar_p_critical_path_20261123/figures/output192_d_cap_intervention_fixed192_ttft_goodput.png) | [figures/output192_d_cap_intervention_fixed192_ttft_goodput.pdf](dense_ar_p_critical_path_20261123/figures/output192_d_cap_intervention_fixed192_ttft_goodput.pdf) | [figures/output192_d_cap_intervention_fixed192_ttft_goodput.csv](dense_ar_p_critical_path_20261123/figures/output192_d_cap_intervention_fixed192_ttft_goodput.csv) |
