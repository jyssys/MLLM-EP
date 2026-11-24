# Generic Exact Token Split: 실제 Online Serving N3

Runtime 판정은 NO_GO_ON_TESTED_GRID다. 검사한 10% Static·현재 상태 Idle-fallback 정책에 한정한다.
전체 Task Quality는 별도로 HOLD다. Runtime 실패를 품질 미확정만으로 설명하지 않는다.
실제 Online Vision Encoding·Continuous P/D·KV Handoff를 포함했다. C0나 Offline Oracle의 성능표가 아니다.

## 동결 계약

Original BF16 Qwen2.5-VL-7B-Instruct, BI0, Full28, 원본 Vision·Processor·MRoPE를 유지했다.
Model·Processor Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.
같은 H100 GPU4/5/6을 E/P/D로 사용했다. P/D는 B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.
Main400을 보존했다. 사용자 변경 지시에 따라 후속 C3는 동결한 Main100을 사용했다.
Main100 SHA256은 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1이다.
Main100과 그 안의 Heldout50은 별도로 평가했다. 이미지·문서 Context 분리는 원래 Manifest를 유지했다.
동일 Arrival Seed2026112402와 Order Seed2026112404를 사용했다. Offered QPS는 2.0이다.
각 Method는 독립 초기화 3회다. Natural EOS와 요청별 동일 Output 수의 Replay는 분리했다.
Natural Output Cap은 2048이다. Replay는 길이만 같으며 Answer String 일치를 보장하지 않는다.

## 비교군과 Admission

| Method | 실행 경로 | D 정책 |
|---|---|---|
| NativeStrong | Stock Compiled Full P, 원래 NIXL D | Token Adapter나 추가 D Cap 없음 |
| NativeCommon | 후보와 공통 Compiled Reduction 설정, Full P | 원래 Native Admission 유지 |
| Static10 | 실제 전체 입력의 약10% Contiguous Suffix를 D에 이전 | 기존 D HTTP Stream이 없을 때만 추가 Suffix 실행 |
| Idle10 | 계획 시 P HTTP 작업이 있고 D HTTP 작업이 없을 때만 10% Split, 아니면 Native | Prefix 완료 후 D 상태를 다시 검사하고 안전 조건까지 대기 |

10%는 실제 LLM Token Sequence 기준이다. Integer Cut은 요청별 Raw Partition에 있다.
필요한 P/D 이미지 Consumer Descriptor는 미리 발행했다. 후보를 On-demand EC RPC 방식으로 잘못 설명하지 않는다.
현재 HTTP 상태는 GPU Queue·미래 Safe Window와 다르다. 이 정책을 검증된 Dynamic Split-size Predictor로 부르지 않는다.
모든 입력 Token은 첫 Output 전에 전체 LLM Layer를 완료했다.
Exact은 계산 누락과 KV 근사가 없다는 뜻이다. 모든 BI0 Batch Geometry의 Byte-wise 출력 동일성을 뜻하지 않는다.
10% 후보는 동결 후 N1/N8 조건부 KV·출력 검사를 통과했다. 25%의 N8 KV 실패는 제외한 채 보존했다.

## Latency: 독립 Start Metric의 평균

| Subset | Output | Method | TTFT mean ms | TTFT p50 ms | TTFT p95 ms | Mean TBT ms | ITL p99 ms | TPOT p95 ms/token |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Main100 | natural | NativeStrong | 2572.281 | 2242.314 | 5167.593 | 6.349 | 15.705 | 6.924 |
| Main100 | natural | NativeCommon | 2450.351 | 2242.050 | 4720.466 | 6.361 | 16.013 | 6.913 |
| Main100 | natural | Static10 | 3496.340 | 2676.541 | 8004.792 | 6.158 | 8.496 | 6.394 |
| Main100 | natural | Idle10 | 2736.518 | 2201.887 | 6363.958 | 6.269 | 13.724 | 6.644 |
| Main100 | replay | NativeStrong | 2611.181 | 2301.934 | 5186.367 | 6.337 | 15.069 | 6.842 |
| Main100 | replay | NativeCommon | 2458.971 | 2204.997 | 4809.002 | 6.365 | 15.346 | 6.819 |
| Main100 | replay | Static10 | 3392.034 | 2554.745 | 7890.089 | 6.147 | 8.087 | 6.380 |
| Main100 | replay | Idle10 | 2464.364 | 2224.058 | 5050.368 | 6.279 | 14.190 | 6.703 |
| Heldout50 | natural | NativeStrong | 1687.054 | 1646.841 | 2687.459 | 6.496 | 15.624 | 6.881 |
| Heldout50 | natural | NativeCommon | 1662.302 | 1620.918 | 2679.739 | 6.518 | 15.852 | 6.954 |
| Heldout50 | natural | Static10 | 4175.625 | 3476.674 | 7509.907 | 6.207 | 8.719 | 6.410 |
| Heldout50 | natural | Idle10 | 1878.998 | 1669.454 | 3180.897 | 6.448 | 14.748 | 6.995 |
| Heldout50 | replay | NativeStrong | 1696.305 | 1651.931 | 2709.802 | 6.464 | 15.597 | 6.801 |
| Heldout50 | replay | NativeCommon | 1678.435 | 1641.313 | 2665.276 | 6.477 | 15.421 | 6.820 |
| Heldout50 | replay | Static10 | 4377.259 | 3758.441 | 7732.145 | 6.203 | 7.970 | 6.413 |
| Heldout50 | replay | Idle10 | 1797.199 | 1717.267 | 3099.517 | 6.413 | 15.582 | 6.833 |

Mean TBT는 모든 Client Token Interval의 평균이다. 같은 Token을 여러 Stage에서 중복 집계하지 않았다.
표의 Percentile은 Start별 Percentile의 평균이다. 요청 300개를 합친 Percentile이 아니다.

## Throughput와 SLO

| Subset | Output | Method | Total tok/s | Output tok/s | Completed req/s | Goodput tok/s | ITL pass % | TPOT pass % | Joint % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Main100 | natural | NativeStrong | 20704.277 | 97.994 | 1.868 | 78.519 | 96.333 | 100.000 | 87.333 |
| Main100 | natural | NativeCommon | 20880.453 | 98.258 | 1.884 | 87.786 | 95.667 | 100.000 | 91.667 |
| Main100 | natural | Static10 | 19542.154 | 95.193 | 1.763 | 66.057 | 97.333 | 100.000 | 69.000 |
| Main100 | natural | Idle10 | 20332.742 | 93.267 | 1.835 | 75.061 | 97.333 | 100.000 | 85.000 |
| Main100 | replay | NativeStrong | 20802.636 | 98.905 | 1.877 | 83.639 | 96.667 | 100.000 | 88.333 |
| Main100 | replay | NativeCommon | 20939.927 | 99.558 | 1.889 | 89.476 | 95.667 | 100.000 | 91.667 |
| Main100 | replay | Static10 | 19501.758 | 92.720 | 1.760 | 62.523 | 98.333 | 100.000 | 68.000 |
| Main100 | replay | Idle10 | 20937.913 | 99.548 | 1.889 | 92.795 | 97.667 | 100.000 | 93.667 |
| Heldout50 | natural | NativeStrong | 20221.672 | 136.917 | 1.869 | 116.372 | 96.667 | 100.000 | 96.667 |
| Heldout50 | natural | NativeCommon | 20124.717 | 134.333 | 1.860 | 119.934 | 95.333 | 100.000 | 95.333 |
| Heldout50 | natural | Static10 | 18131.120 | 119.075 | 1.676 | 81.891 | 99.333 | 100.000 | 58.000 |
| Heldout50 | natural | Idle10 | 17853.010 | 136.004 | 1.648 | 130.735 | 97.333 | 100.000 | 96.667 |
| Heldout50 | replay | NativeStrong | 20085.721 | 132.465 | 1.856 | 132.465 | 100.000 | 100.000 | 100.000 |
| Heldout50 | replay | NativeCommon | 20085.453 | 132.464 | 1.856 | 130.226 | 98.667 | 100.000 | 98.667 |
| Heldout50 | replay | Static10 | 17919.217 | 118.177 | 1.656 | 83.930 | 100.000 | 100.000 | 58.000 |
| Heldout50 | replay | Idle10 | 19770.274 | 130.385 | 1.827 | 119.200 | 98.667 | 100.000 | 98.000 |

Total tok/s는 요청의 실제 Input+Output을 첫 예정 Arrival부터 Drain까지의 시간으로 나눈 값이다.
Raw Output tok/s는 완료한 요청의 실제 Output 수를 같은 시간으로 나눈 값이다.
SLO-goodput은 Joint PASS 요청의 Output Token 수만 같은 시간으로 나눈 값이다.
Joint는 TTFT≤5000ms AND 모든 Client ITL≤34ms AND Request TPOT≤33ms/token이다.
Single-output의 TPOT는 0이고 ITL은 빈 집합이다. 원래 Metric 계약을 유지했다.
Absolute Decode는 요청별 All-ITL AND TPOT다. 각 독립 Start에서 95% 이상이어야 한다.
완료된 모든 C37 Raw를 Timestamp에서 다시 계산했고 두 개별 Pass Rate와 실제 AND를 구분했다.
각 Start의 8개 실패 조합·실제 요청 수·Output Token 수는 원시 CSV에 보존했다.
Finite-cohort 처리량을 Sustainable Serving Capacity로 부르지 않는다.

## Paired Gate와 불확실성

| Subset | Output | Candidate | Native | Metric | Ratio of Start-means | Paired 95% CI |
|---|---|---|---|---|---:|---|
| Main100 | natural | Static10 | NativeStrong | TTFT_p50_ms | 1.193652 | [1.1266311122632076, 1.2727764007805011] |
| Main100 | natural | Static10 | NativeStrong | TTFT_p95_ms | 1.549037 | [1.4391708652805466, 1.701766420062693] |
| Main100 | natural | Static10 | NativeStrong | ITL_p99_ms | 0.541000 | [0.5299138486816304, 0.5535174219395145] |
| Main100 | natural | Static10 | NativeStrong | TPOT_p95_ms | 0.923422 | [0.9075210463336043, 0.9396594744948643] |
| Main100 | natural | Static10 | NativeStrong | SLO_goodput | 0.841293 | [0.7616459451047739, 0.9422373536666971] |
| Main100 | natural | Static10 | NativeCommon | TTFT_p50_ms | 1.193792 | [1.140094941067597, 1.2585347341853543] |
| Main100 | natural | Static10 | NativeCommon | TTFT_p95_ms | 1.695763 | [1.5087248859384372, 1.8769944147839337] |
| Main100 | natural | Static10 | NativeCommon | ITL_p99_ms | 0.530601 | [0.5280076313094603, 0.5332516759693057] |
| Main100 | natural | Static10 | NativeCommon | TPOT_p95_ms | 0.924861 | [0.9178866488570325, 0.9328876090295536] |
| Main100 | natural | Static10 | NativeCommon | SLO_goodput | 0.752482 | [0.7321321872894908, 0.7797635393341303] |
| Main100 | natural | Idle10 | NativeStrong | TTFT_p50_ms | 0.981971 | [0.9199329398444342, 1.0745190690644886] |
| Main100 | natural | Idle10 | NativeStrong | TTFT_p95_ms | 1.231513 | [1.1417900356704258, 1.371028945136858] |
| Main100 | natural | Idle10 | NativeStrong | ITL_p99_ms | 0.873892 | [0.7399699897128713, 0.9877719680003092] |
| Main100 | natural | Idle10 | NativeStrong | TPOT_p95_ms | 0.959524 | [0.9347031659589747, 0.9812927619132508] |
| Main100 | natural | Idle10 | NativeStrong | SLO_goodput | 0.955960 | [0.9237963460872999, 0.9963783746004141] |
| Main100 | natural | Idle10 | NativeCommon | TTFT_p50_ms | 0.982086 | [0.9545297566132254, 1.0090738289307335] |
| Main100 | natural | Idle10 | NativeCommon | TTFT_p95_ms | 1.348163 | [1.1941933864972525, 1.4964645002229369] |
| Main100 | natural | Idle10 | NativeCommon | ITL_p99_ms | 0.857094 | [0.7149828841699603, 0.9774310892248472] |
| Main100 | natural | Idle10 | NativeCommon | TPOT_p95_ms | 0.961020 | [0.9453120055470758, 0.9776198873731952] |
| Main100 | natural | Idle10 | NativeCommon | SLO_goodput | 0.855044 | [0.7814727756227804, 0.9242979386750116] |
| Main100 | replay | Static10 | NativeStrong | TTFT_p50_ms | 1.109825 | [1.046211196638743, 1.1858250518623292] |
| Main100 | replay | Static10 | NativeStrong | TTFT_p95_ms | 1.521313 | [1.3958670540542613, 1.6804830146355918] |
| Main100 | replay | Static10 | NativeStrong | ITL_p99_ms | 0.536699 | [0.5257465974408811, 0.5494137523254702] |
| Main100 | replay | Static10 | NativeStrong | TPOT_p95_ms | 0.932390 | [0.9279016797147875, 0.9380074408486105] |
| Main100 | replay | Static10 | NativeStrong | SLO_goodput | 0.747526 | [0.7019914608155234, 0.7938481845817305] |
| Main100 | replay | Static10 | NativeCommon | TTFT_p50_ms | 1.158616 | [1.121975216544097, 1.198245328442343] |
| Main100 | replay | Static10 | NativeCommon | TTFT_p95_ms | 1.640692 | [1.6090262563179138, 1.6868133591696093] |
| Main100 | replay | Static10 | NativeCommon | ITL_p99_ms | 0.526994 | [0.5063768454185538, 0.5476656176050224] |
| Main100 | replay | Static10 | NativeCommon | TPOT_p95_ms | 0.935522 | [0.9251011698450023, 0.9448583747397739] |
| Main100 | replay | Static10 | NativeCommon | SLO_goodput | 0.698764 | [0.6808034563617132, 0.7200521876790732] |
| Main100 | replay | Idle10 | NativeStrong | TTFT_p50_ms | 0.966169 | [0.9520793200867532, 0.9774382398836945] |
| Main100 | replay | Idle10 | NativeStrong | TTFT_p95_ms | 0.973778 | [0.9440708625050621, 0.9998590448213838] |
| Main100 | replay | Idle10 | NativeStrong | ITL_p99_ms | 0.941682 | [0.8589271919643456, 1.0252520129936131] |
| Main100 | replay | Idle10 | NativeStrong | TPOT_p95_ms | 0.979567 | [0.9699795228089909, 0.988984557995663] |
| Main100 | replay | Idle10 | NativeStrong | SLO_goodput | 1.109462 | [1.0709205747786188, 1.162643784016883] |
| Main100 | replay | Idle10 | NativeCommon | TTFT_p50_ms | 1.008644 | [0.9550371480118172, 1.0510964935639544] |
| Main100 | replay | Idle10 | NativeCommon | TTFT_p95_ms | 1.050191 | [0.9512946573606382, 1.1455719927591828] |
| Main100 | replay | Idle10 | NativeCommon | ITL_p99_ms | 0.924653 | [0.875098704261392, 0.9794012536822719] |
| Main100 | replay | Idle10 | NativeCommon | TPOT_p95_ms | 0.982857 | [0.9631034161293494, 1.0044863540950562] |
| Main100 | replay | Idle10 | NativeCommon | SLO_goodput | 1.037091 | [1.0080074297936599, 1.067240772157356] |
| Heldout50 | natural | Static10 | NativeStrong | TTFT_p50_ms | 2.111117 | [1.946161617498141, 2.2867432176243283] |
| Heldout50 | natural | Static10 | NativeStrong | TTFT_p95_ms | 2.794426 | [2.697282163664295, 2.8800841440597624] |
| Heldout50 | natural | Static10 | NativeStrong | ITL_p99_ms | 0.558060 | [0.5414885280373061, 0.5754256334297067] |
| Heldout50 | natural | Static10 | NativeStrong | TPOT_p95_ms | 0.931541 | [0.9144614101020584, 0.9456167596268014] |
| Heldout50 | natural | Static10 | NativeStrong | SLO_goodput | 0.703693 | [0.644779861689298, 0.7569317223435185] |
| Heldout50 | natural | Static10 | NativeCommon | TTFT_p50_ms | 2.144879 | [2.0052559430699026, 2.2923988804448716] |
| Heldout50 | natural | Static10 | NativeCommon | TTFT_p95_ms | 2.802477 | [2.7136395439758267, 2.8927862748471203] |
| Heldout50 | natural | Static10 | NativeCommon | ITL_p99_ms | 0.550044 | [0.5056003583087547, 0.6069101581080674] |
| Heldout50 | natural | Static10 | NativeCommon | TPOT_p95_ms | 0.921664 | [0.9175238258753916, 0.9260471906976789] |
| Heldout50 | natural | Static10 | NativeCommon | SLO_goodput | 0.682795 | [0.6671546211260413, 0.69564852080692] |
| Heldout50 | natural | Idle10 | NativeStrong | TTFT_p50_ms | 1.013731 | [0.9526174686590414, 1.0688292881256547] |
| Heldout50 | natural | Idle10 | NativeStrong | TTFT_p95_ms | 1.183607 | [1.0332346576713676, 1.3623613832036174] |
| Heldout50 | natural | Idle10 | NativeStrong | ITL_p99_ms | 0.943919 | [0.8759821755809579, 1.0183743484320118] |
| Heldout50 | natural | Idle10 | NativeStrong | TPOT_p95_ms | 1.016578 | [0.9675494760652458, 1.0747268632146825] |
| Heldout50 | natural | Idle10 | NativeStrong | SLO_goodput | 1.123418 | [0.9798230025777979, 1.261338758157618] |
| Heldout50 | natural | Idle10 | NativeCommon | TTFT_p50_ms | 1.029943 | [0.9788982230017936, 1.079098935216451] |
| Heldout50 | natural | Idle10 | NativeCommon | TTFT_p95_ms | 1.187017 | [1.039486122952207, 1.3685123335744607] |
| Heldout50 | natural | Idle10 | NativeCommon | ITL_p99_ms | 0.930362 | [0.8493450656554562, 1.036650791092252] |
| Heldout50 | natural | Idle10 | NativeCommon | TPOT_p95_ms | 1.005800 | [0.9699483379892956, 1.0528463303901316] |
| Heldout50 | natural | Idle10 | NativeCommon | SLO_goodput | 1.090054 | [1.0552236987492498, 1.1188917149329043] |
| Heldout50 | replay | Static10 | NativeStrong | TTFT_p50_ms | 2.275180 | [2.226493505300745, 2.3203989260642515] |
| Heldout50 | replay | Static10 | NativeStrong | TTFT_p95_ms | 2.853399 | [2.818124558220652, 2.8807841888422216] |
| Heldout50 | replay | Static10 | NativeStrong | ITL_p99_ms | 0.510980 | [0.5074734459292388, 0.5144785958023996] |
| Heldout50 | replay | Static10 | NativeStrong | TPOT_p95_ms | 0.942997 | [0.9404495084526259, 0.9455313929756408] |
| Heldout50 | replay | Static10 | NativeStrong | SLO_goodput | 0.633597 | [0.6308488698303509, 0.6359783757925008] |
| Heldout50 | replay | Static10 | NativeCommon | TTFT_p50_ms | 2.289899 | [2.216742654514309, 2.3634363636820797] |
| Heldout50 | replay | Static10 | NativeCommon | TTFT_p95_ms | 2.901068 | [2.8206528679714067, 2.9739391874407826] |
| Heldout50 | replay | Static10 | NativeCommon | ITL_p99_ms | 0.516835 | [0.5139008580588825, 0.5202309130736563] |
| Heldout50 | replay | Static10 | NativeCommon | TPOT_p95_ms | 0.940371 | [0.9326189148012414, 0.9474460033456459] |
| Heldout50 | replay | Static10 | NativeCommon | SLO_goodput | 0.644493 | [0.6317603605960567, 0.6612574678804609] |
| Heldout50 | replay | Idle10 | NativeStrong | TTFT_p50_ms | 1.039551 | [0.9923963046597931, 1.0877101433215792] |
| Heldout50 | replay | Idle10 | NativeStrong | TTFT_p95_ms | 1.143817 | [1.0260079860346456, 1.301057482631515] |
| Heldout50 | replay | Idle10 | NativeStrong | ITL_p99_ms | 0.999019 | [0.9829641852725853, 1.013090044861472] |
| Heldout50 | replay | Idle10 | NativeStrong | TPOT_p95_ms | 1.004781 | [0.9923204470953966, 1.0213913194169515] |
| Heldout50 | replay | Idle10 | NativeStrong | SLO_goodput | 0.899857 | [0.8088812696562928, 0.9786806988987704] |
| Heldout50 | replay | Idle10 | NativeCommon | TTFT_p50_ms | 1.046277 | [0.9875237253260396, 1.0926828642898145] |
| Heldout50 | replay | Idle10 | NativeCommon | TTFT_p95_ms | 1.162925 | [1.0319052874791963, 1.3267231581234793] |
| Heldout50 | replay | Idle10 | NativeCommon | ITL_p99_ms | 1.010467 | [0.9921093689417632, 1.0287333482017318] |
| Heldout50 | replay | Idle10 | NativeCommon | TPOT_p95_ms | 1.001983 | [0.9833533694846571, 1.0253922493393275] |
| Heldout50 | replay | Idle10 | NativeCommon | SLO_goodput | 0.915331 | [0.8194755941316155, 0.9915231080845824] |

Paired CI는 같은 Block의 독립 Start3쌍을 사용했다. 27개 Empirical Resample을 모두 계산했다.
N3의 CI 해상도는 제한적이다. 요청이나 Token을 독립 Server Start로 취급하지 않았다.
Relative ITL p99·TPOT p95와 TTFT p50의 CI 상한은 두 Native 각각에 대해 1.05 이하여야 한다.
개선 조건은 TTFT p95 최소5% 감소 또는 SLO-goodput 최소5% 증가와 우위 방향의 CI다.
Static10은 Decode Gate를 통과했지만 TTFT p50과 Serving 개선 조건을 통과하지 못했다.
Idle10도 전체 Runtime Gate를 통과하지 못했다. Heldout Natural에서는 상대 Decode Gate도 실패했다.
TTFT 개선으로 Decode 악화를 상쇄하거나, 낮은 ITL만으로 성공을 선언하지 않았다.

## 원인과 미측정

Main Replay의 Static10은 NativeCommon보다 P HTTP 평균이 305.329ms 짧았다.
반면 추가 D Coordinator 대기는 평균 971.956ms 증가했다.
평균 TTFT 차이는 +933.063ms였다. Paired Start-level 95% CI는 [817.498, 1020.626]ms다.
이는 같은 요청의 중복 없는 Host 경계다. GPU-exclusive 계산이나 TTFT p95의 가법 분해가 아니다.
관측된 D 대기를 0으로 만드는 Oracle은 Queue Feedback과 Decode 경쟁을 다시 계산하지 않는다.
그 Oracle을 실제 Online 개선으로 표시하지 않았다.
Clean 실행의 실제 GPU Batch·GPU-exclusive Operator별 비용·실제 미래 Safe Budget은 NOT_MEASURED다.
적격 저부하 Physical Idle을 고부하의 안전한 Suffix Admission 근거로 대체하지 않았다.
BI0 Encoder Batch 변화와 Task Score 불확실성은 별도의 품질 한계다.
MMLongBench-Doc 공식 외부 Answer Extraction은 승인 없이 호출하지 않았다. Raw Score는 전체 품질 PASS가 아니다.

## 재현 자료

Evidence Namespace는 mllm_epd_prefill_heavy_token_split_20261124다.
Config는 configs/token_CONFIRM_C37.json이다. 각 Server Lease에 GPU UUID·Source Archive SHA256·명령이 있다.
원시 Request와 Proxy Timeline은 raw/token_CONFIRM_C37_* 및 raw/proxy_token_CONFIRM_C37_*에 있다.
최초 N3 집계는 analysis/token_confirmation_C37_*_v0다. 새 표는 analysis/token_online_C37_*_v0다.
Figure는 figures/token_confirmation_C37_*_v0와 figures/main100_C37_joint_failure_breakdown_v0다.
각 Figure의 PNG/PDF와 대응 CSV를 보존했다. Native Motivation의 별도 N3 결과와 합쳐 평균내지 않았다.

관련 보고서: [정합성](mllm_epd_generic_token_split_correctness.md), [원시 N3](mllm_epd_protected_token_confirmation_v1.md), [Host 경로](mllm_epd_protected_token_critical_path_v1.md), [품질 한계](mllm_epd_n3_task_quality_limits_v1.md).
