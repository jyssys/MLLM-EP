# Protected Token Split — Discovery50 Online 결과

5개 Static 비율과 Idle-fallback을 실제 E/P/D Online 경로에서 측정했다.
증거 수준은 `MEASURED_ONLINE` Runtime Screening이다. 각 Family의 독립 초기화는 1회다.
전체 품질과 독립 N3 CI는 미확정이다. 이 결과로 GO를 선언하지 않는다.

## 고정 계약

Main100의 기존 Discovery50만 사용했다. Heldout50 성능은 사용하지 않았다.
Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
GPU4/5/6, Original BF16, BI0, Full28, 원본 E와 Processor를 유지했다.
P/D는 공통 Compiled 설정, B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
Native에는 Token Adapter나 새 D Cap을 적용하지 않았다. Stock NativeStrong도 변경하지 않았다.
모든 Arm의 요청 순서, QPS2.0과 상대 Arrival은 같았다.
각 Static 비율은 같은 Chunk 경계의 N1 Teacher와 KV/Output Strict 검사를 통과한 뒤 측정했다.
측정 중 CPU Tensor Capture와 GPU Profiler는 껐다. Warm-up은 표에서 제외했다.
보호 정책은 기존 D HTTP Stream이 없는 경우에만 추가 Suffix를 시작한다.
Native 요청의 대기는 첫 생성 Token에서 해제한다. 전체 Decode 동안 모든 Native를 막지는 않는다.
이 정책은 미래 Safe Window 또는 Dynamic Split-size Predictor가 아니다.

## 모든 측정 지점

| Output | Method | D Fraction | Output Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | Goodput tok/s | Joint % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| natural | Native | 0.00 | 1714 | 1179.996 | 3197.962 | 14.137 | 50.657 | 100.000 |
| replay | Native | 0.00 | 1701 | 1168.775 | 3204.322 | 13.748 | 50.192 | 100.000 |
| natural | Static Token | 0.25 | 1612 | 1232.477 | 3317.399 | 8.635 | 47.379 | 100.000 |
| replay | Static Token | 0.25 | 1701 | 1236.612 | 3412.474 | 7.842 | 49.980 | 100.000 |
| natural | Static Token | 0.50 | 1577 | 1331.085 | 3908.603 | 8.346 | 45.802 | 100.000 |
| replay | Static Token | 0.50 | 1701 | 1386.803 | 3853.050 | 7.738 | 49.625 | 100.000 |
| natural | Static Token | 0.10 | 1856 | 1359.460 | 3377.018 | 8.456 | 54.288 | 100.000 |
| replay | Static Token | 0.10 | 1701 | 1238.488 | 3320.145 | 7.988 | 48.176 | 98.000 |
| natural | Static Token | 0.30 | 1588 | 1363.500 | 3261.194 | 8.774 | 46.604 | 100.000 |
| replay | Static Token | 0.30 | 1701 | 1249.002 | 3452.668 | 8.150 | 49.884 | 100.000 |
| natural | Static Token | 0.20 | 1861 | 1377.430 | 3205.200 | 8.569 | 54.564 | 100.000 |
| replay | Static Token | 0.20 | 1701 | 1261.232 | 3330.970 | 7.902 | 49.684 | 100.000 |
| natural | Idle-fallback Token | 0.10 | 1892 | 1219.487 | 3085.072 | 12.832 | 55.547 | 100.000 |
| replay | Idle-fallback Token | 0.10 | 1701 | 1168.718 | 2917.444 | 10.200 | 49.933 | 100.000 |

## 사전 정의한 선택과 해석

Static 비율 10/20/25/30/50%를 모두 보존했다. 모든 비율이 Runtime Point Gate를 모두 통과하지는 못했다.
Point Gate에는 TTFT p50, ITL p99와 TPOT p95의 Native 대비 1.05 제한이 있었다.
모든 Static 비율의 Replay TTFT p50이 이 제한을 넘었다.
실패 시 사전 규칙에 따라 Replay p95가 가장 작은 10%를 진단용으로 동결했다.
25%는 별도의 최고 Replay Goodput 비율이었다. 두 비율 모두 확정된 Serving Winner가 아니다.
Idle-fallback은 이 10%와 현재 P/D HTTP 상태만 사용했다.
P에 Outstanding 요청이 있고 D HTTP Stream이 없으면 Split을 계획했다. 그 외에는 Native를 유지했다.
이는 정확한 P GPU Queue 또는 미래 EOS를 입력으로 사용한 것이 아니다.
계획 이후에도 실제 D 시작 전에는 기존 D Stream의 완료를 기다렸다.
Idle-fallback Replay p95는 2917.444ms였다. Native는 3204.322ms였다.
차이는 약 8.95%의 단일 실행 Point Gain이다. 독립 반복과 Heldout에서 재현해야 한다.
Replay Goodput은 Native50.192, Idle49.933 tok/s였다.
Natural의 Goodput 차이에는 생성 길이 변화도 포함된다. 이를 계산 Speedup으로 단정하지 않는다.

## Host 경로와 한계

요청별 TTFT를 중복 없는 Host 경계로 나눴다. 합계가 원시 TTFT와 일치했다.
E/P HTTP Enclosure에는 Queue, Host 처리와 GPU 작업이 함께 들어 있다.
D Submit 이후 구간에도 KV 전송, Backend Admission, Forward와 Client Delivery가 함께 들어 있다.
이 구간을 GPU-exclusive Kernel Time으로 표시하지 않았다.
Coordinator D Gate 대기를 제거한 값은 별도 `OFFLINE_ORACLE` CSV다.
이 Oracle은 Queue Feedback과 Decode 경쟁을 다시 계산하지 않는다. 실제 개선율이 아니다.
실패 8개 조합의 Request Count와 Output Token Weight도 저장했다.
ITL p99가 낮아도 모든-ITL 조건에서 탈락한 요청은 삭제하지 않았다.

## Split 구성과 품질 한계

기존 Discovery/Heldout은 Source 비율이 다르다. 분할을 사후 변경하지 않았다.

| Split | Source | Requests |
|---|---|---:|
| discovery | TIGER-Lab/Mantis-Eval | 3 |
| discovery | lmms-lab/DocVQA | 13 |
| discovery | lmms-lab/MP-DocVQA | 25 |
| discovery | yubo2333/MMLongBench-Doc | 9 |
| heldout | TIGER-Lab/Mantis-Eval | 22 |
| heldout | lmms-lab/DocVQA | 0 |
| heldout | lmms-lab/MP-DocVQA | 0 |
| heldout | yubo2333/MMLongBench-Doc | 28 |

이 분포 차이는 Generalization 한계다. Main100 전체가 Production Traffic을 대표한다고 주장하지 않는다.
Raw Task Score를 같은 원래 Scorer로 계산했다. MMLongBench-Doc 공식 답변 추출은 미실행이다.
DocVQA Raw ANLS Floor도 남아 있다. 전체 Task Quality는 계속 `PENDING`이다.
Heldout 전에 비율과 현재 상태 Rule을 동결했다. 새 SLO를 결과에 맞춰 설정하지 않았다.
CSV는 `analysis/token_discovery_C34_*_v0.csv`다. 실제 Figure는 `figures/token_discovery_C34_*_v0`에 있다.
