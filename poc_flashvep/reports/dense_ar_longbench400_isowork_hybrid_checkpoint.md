# LongBench400 Exact-work Hybrid Checkpoint

작성일: 2026-11-21 KST.
전체 Cell 검사는 진행 중이다.
이 문서는 완료한 대표 Cell의 범위와 한계를 기록한다.

## 실행 계약

실제 LongBench Prompt를 사용했다.
C0의 실제 P Prefill, D Continuation, D Decode Batch는 1/1/1이다.
Source Credit은 4다.
BI0/BF16/B8192/S1024/KV42GiB/Block128을 사용했다.
Natural EOS Main400과 달리 이 검사는 Output Length를 통제한다.

Layer8, Token8, Hybrid4/Suffix≈L/8을 비교한다.
Hybrid의 D Rows는 `4L + 32s`다.
`s=L/8`이면 D Rows는 `8L`이다.
세 방법의 Aggregate P Rows와 D Rows를 정확히 맞췄다.
실제 P/D Partition Log와 Coverage에서도 같은 Rows를 확인했다.
Attention Work와 Transfer Payload는 같지 않다.
Native는 Full36 P Prefill과 명시적인 Last-prompt-query 재실행을 유지한다.
Native를 Offloaded-work가 같은 Arm이라고 표현하지 않는다.

각 Cell은 Held-out 실제 Prompt 16개를 사용했다.
독립 Start는 3회다.
Arrival Seed는 2026112351/2026112352/2026112353이다.
각 Seed의 네 방법은 같은 Prompt와 Arrival Trace를 사용했다.
CI는 Paired Start-level Empirical Bootstrap의 27개 Resample로 계산했다.
Pointwise CI이며 여러 Cell에 대한 Familywise 보정은 적용하지 않았다.

## 완료한 Long-input / Output64 Cell

Input Bucket은 20K–36K다.
QPS는 1.253873이다.
각 Start의 P Rows는 12,676,608이다.
각 Start의 D Rows는 3,621,888이다.

| 방법 | TTFT p95 (ms) | ITL p99 (ms) | TPOT p95 (ms/token) | SLO-goodput (Tokens/s) |
|---|---:|---:|---:|---:|
| Native | 5829.594 | 9.044 | 8.565 | 41.918 |
| Layer8 | 4241.532 | 9.053 | 8.564 | 65.102 |
| Token8 | 4257.191 | 9.023 | 8.571 | 64.784 |
| Hybrid4/Suffix≈L/8 | 4174.436 | 9.077 | 8.583 | 65.077 |

위 값은 세 Start의 각 Metric 평균이다.
Hybrid/Token TTFT p95 비율은 0.98056이다.
95% CI는 [0.97124, 0.99028]이다.
Hybrid/Layer 비율은 0.98418이다.
95% CI는 [0.97220, 0.99659]이다.
이 Cell에서는 Hybrid가 두 Iso-work Pure Axis보다 조금 빠르다.

Hybrid/Native TTFT p95 비율은 0.71608이다.
95% CI는 [0.69811, 0.73049]이다.
Hybrid/Native SLO-goodput 비율은 1.55248이다.
95% CI는 [1.16522, 2.18586]이다.
Finite-cohort Goodput이며 Sustainable Capacity를 입증하지 않는다.

세 Offload 방법 모두 Native-relative Decode 5% Gate를 통과했다.
Hybrid/Native ITL p99 CI는 [0.99048, 1.02149]다.
Hybrid/Native TPOT p95 CI는 [1.00118, 1.00302]다.
Absolute Decode Attainment도 각 Start에서 95% 이상이었다.
실제 ITL Timestamp를 수정하지 않았다.

## 반례

같은 Long-input Bucket의 Output16에서는 Hybrid가 Token보다 느렸다.
Hybrid/Token TTFT p95 비율은 1.04644다.
95% CI는 [1.02061, 1.06655]다.
Hybrid/Layer 비율은 0.92546이다.
95% CI는 [0.89492, 0.95903]이다.
이 Cell은 Hybrid가 Token을 이긴 결과가 아니다.

6K–10K / Output16에서도 Hybrid가 두 Pure Axis보다 느렸다.
Hybrid/Token 비율은 1.05933이다.
95% CI는 [1.04448, 1.07738]이다.
Hybrid/Layer 비율은 1.05263이다.
95% CI는 [1.01099, 1.07970]이다.
불리한 Cell도 전체 Table과 Figure에 유지한다.

## 제한과 다음 Gate

위 결과는 `MEASURED_ONLINE` C0 Control이다.
일반 C3 Continuous Batching의 성능으로 재사용하지 않는다.
한 Iso-work Frontier의 승리를 Best-tuned Token 전체에 대한 승리로 확대하지 않는다.
큰 차이가 아닌 1.6–1.9%의 Pure-axis 대비 TTFT 이득이다.
Hybrid/Token SLO-goodput 비율의 CI는 1을 포함한다.

기존 QPS Reference는 P+D Serial Service를 바탕으로 만들었다.
실제 Pipeline의 병목 Service Rate와 같지 않다.
따라서 이 Cell만으로 실제 Capacity Knee 전체를 설명하지 않는다.
별도의 Pipeline-load Supplement를 사전에 동결했다.
기존 Cell과 SLO는 변경하지 않는다.

Strong Static Discovery, 일반 Continuous Serving 및 Decode 보호 CI는 별도 검사다.
Dynamic Window 또는 Axis Selection 성공은 아직 입증하지 않았다.

## 재현 근거

- `configs/iso8_hybrid_confirmation_frozen_v1.json`
- `analysis/iso8_hybrid_heldout_confirmation_v1.json`
- Code: `iso8_confirmation_jobs.py`, `iso8_analysis.py`.
- Source: `iso8_hybrid_heldout_confirmation_v1_20261121.zip`.
- Source SHA256: `b6d758fa5d1bf4533e7c21c645643227f1a8ef74df5824a7746364fc39ffc38e`.

경로는 `dense_ar_longbench400_2d_20261121` Namespace 기준이다.
