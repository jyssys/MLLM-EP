# H200 실제 Prefill Cut에 정렬한 D 여유 분석

기존 Generic Shadow는 Main100 B8 QPS5 Natural에서 P callback 29건 중 21건을 빈 D admission 상태에서 관측했다. 20건은 예측된 예산 부족, 1건은 Cost Model 지원 범위 초과로 거절됐다. 나머지 8건에서는 D에 작업이 있었다. 해당 실행에서는 Suffix를 실행하지 않았다.

## 기존 Cut 시점 재분석

| 정책과 출력 | 전체 callback | 중앙 ROI callback | 중앙 ROI의 D HTTP 무작업 | 다음 다른 P-ready까지 ≥160ms | 중앙값 |
|---|---:|---:|---:|---:|---:|
| Generic Natural | 29 | 9 | 7 | 5 | 195.67ms |
| Generic Replay | 32 | 15 | 7 | 2 | 28.27ms |
| Hard Natural | 27 | 10 | 4 | 3 | 271.80ms |
| Hard Replay | 19 | 7 | 2 | 2 | 447.55ms |
| Late10 Natural | 13 | 7 | 4 | 3 | 202.73ms |

OFFLINE_ORACLE에서 D가 비어 있던 Generic Natural의 7개 Cut 중 3개는 기존 비용 추정치와 10ms 여유를 합쳐도 다음 다른 요청의 P-ready보다 먼저 끝날 수 있었다. 해당 Suffix는 1,759, 4,188, 4,363 Tokens였고 관측된 구간은 각각 224, 423, 896ms였다. Replay에서는 2,342-token Suffix 한 건이 같은 비교에서 Fit했다. 다른 요청의 P 완료 시점을 고정한 분석이므로 실제 Split과 SLO 안전성은 실험으로 확인해야 한다.

시간축은 한 서버의 time.monotonic이다. 관측 D HTTP 점유는 P-ready부터 D HTTP EOF까지다. 같은 요청의 Native D 구간을 제외한 경우는 미래 여유의 Oracle로 표시했다. KV 준비 시각, D GPU 물리적 Idle, 실제 Suffix 실행 지연은 각각 별도 측정이 필요하다. 중앙 ROI는 예정 Arrival p20–p80이며 p80을 넘는 구간은 하한만 기록한다. 전체 실행에서 마지막 Arrival 이후 Drain을 별도 표시했다.

## Stock 관측 및 Active 검증

Stock P 관측은 실제 Scheduled Batch의 Query Rows와 원래 Scheduler가 이미 발행한 후속 Work를 기록한다. 후속 Work가 이미 발행된 Cut은 Hold 없이 Split할 수 있는 경계에서 제외한다. D Scheduler의 Running, Pending, Remote KV Wait, Free Blocks는 같은 호스트 시계를 사용하는 Transition Trace에서 얻는다. D Engine 관측이 빠진 경우는 공란으로 남긴다. 같은 실행의 CUPTI 커널 구간과 교차할 수 있고 신뢰성 판정에는 별도의 Clean/Profile/Clean이 필요하다.

첫 Stock B8 BI1 시도는 P/D Scheduler 인터페이스 점검 중 종료했다. 후속 B8 BI1은 Main100 완료와 Timing을 확보했지만 Worker 종료에 의존한 Trace Flush가 동작하지 않았다. 두 실행은 Cut 학습 자료에서 제외했다. 후속 관측은 E/P/D 요청이 모두 끝난 뒤 P/D Engine의 명시적 Flush 응답과 함께 Raw Trace를 기록한다.

## Free BI0 QPS5 Stock Cut 실측

예정 Arrival p20–p80의 내부 P Chunk 완료 70건 중 19건이 목표 Suffix 범위에 들어갔다. 해당 19건은 18개 요청에 걸쳐 있었다. D HTTP와 D Scheduler가 모두 비어 있던 Cut은 8건이다. 8건에서 D는 모두 6,143 KV Blocks를 비워 두고 있었다. 아래의 미래 구간은 해당 Native 실행에서 관측한 호스트 및 Scheduler 구간이다.

| Suffix Tokens | 실제 비율 | 이미 발행된 후속 P Rows | D Scheduler 무작업 | 실제 HTTP 무작업 | 다음 다른 P-ready Oracle |
|---:|---:|---:|---:|---:|---:|
| 3,905 | 15.74% | 3,905 | 288.8ms | 272.8ms | 274.4ms |
| 4,820 | 16.71% | 4,820 | 314.5ms | 298.8ms | 479.9ms |
| 984 | 14.36% | 984 | 41.0ms | 25.4ms | 25.4ms |
| 2,501 | 10.71% | 2,501 | 263.2ms | 242.2ms | 242.9ms |
| 4,496 | 32.36% | 4,496 | 239.2ms | 220.6ms | 1,240.1ms |
| 8,874 | 30.65% | 8,192 | 570.3ms | 552.5ms | 553.9ms |
| 682 | 2.36% | 682 | 254.7ms | 236.9ms | 238.3ms |
| 3,401 | 33.52% | 3,401 | 21.8ms | 10.6ms | 10.6ms |

8건 중 6건의 다음 다른 P-ready까지는 225ms 이상이었다. 전체 100개 요청에서는 Suffix 범위에 들어가는 Cut 43건을 관측했으며, 이미 발행된 후속 P Rows가 0인 Cut은 0건이었다. 같은 Server Start에서 별도 Batch Counter를 켠 반복 관측에서도 중앙 19건 및 전체 50건의 범위 내 Cut에 모두 후속 P 작업이 이미 발행돼 있었다. 이 조건에서는 자연스러운 Cut만 사후 발견하는 No-hold 경로로 실행 가능한 제안이 생기지 않는다. 64/128/256/512/1024/2048/4096 Tokens와 정확히 일치하는 Suffix는 두 실행의 모든 내부 Chunk에서 관측되지 않았다. 첫 Cohort에서 정책 범위 내 가장 짧은 Suffix는 179 Tokens였고, 가장 짧은 내부 Chunk의 남은 2 Tokens는 정책 범위 밖이었다.

Natural-hold는 목표 범위에 들어가는 기존 Stock Chunk를 발행한 뒤 다음 Query Rows 발행을 잠시 막는다. 기존 Generic Natural Shadow의 29개 callback에서 Worker RPC 시간 중앙값은 1.92ms, Hold 시도 중앙값은 1회였다. Hold 시작부터 Chunk 완료까지의 중앙값 230.57ms는 이미 발행된 Chunk의 처리와 겹치는 구간이다. 순수 추가 지연은 No-hold 및 Hold 대조 실험으로 확인해야 한다.

Free BI0 Trace에서 활성 Suffix와 Physical Kernel Idle은 계측하지 않았다. 중앙 ROI의 환경에서는 Scheduler와 HTTP 기준 여유가 있었으며, 실효 Suffix 비용과 Hold로 달라지는 Queue에서 이를 실제로 활용할 수 있는지는 Active 실험이 필요하다.

## Active A, B8 BI1 QPS5

Slice-aware Inline의 Late10을 실제 Main100 C3에서 실행했다. 첫 Cohort의 12개 callback과 같은 Start의 Batch Counter 반복 18개 callback 모두 D_ADMISSION_BUSY로 거절됐다. 관측된 활성 D HTTP 요청 수는 1–6개이며 Pending은 0이었다. 최초 Cohort의 정책 범위는 실제 Suffix 67–3,990 Tokens였다. 모든 callback에서 실제 Hold는 1회였고 P Worker의 Preemption Counter는 0이었다. Worker 결정 RPC 중앙값은 각각 1.962ms 및 1.617ms였다.

모든 callback에서 당시 활성 D HTTP 요청 각각이 Cut 후에도 Client에 Token을 전달했다. 첫 Cohort에서 가장 가까운 다음 Client Token까지의 중앙값은 12.06ms이고 범위는 4.36–49.97ms였다. Counter 반복의 중앙값은 9.41ms였다. HTTP 소유권만 오래 유지됐다는 설명으로는 이 결과를 설명할 수 없다. Client 전달 시각은 별도의 D Kernel 실행 시간과 구분한다.

이 Active A에는 비용과 예산 필터를 사용하지 않았다. 모든 후보에서 D는 기존 HTTP 요청을 처리 중이었고 D Engine의 KV를 조회하기 전에 Admission이 거절됐다. Free BI0에서 확인한 Cut별 여유와 BI1 B8의 D 동작은 각각의 실행 조건에서 판정한다.

## Native B8 Eager QPS5

같은 Stock 계측에서 중앙 ROI의 내부 Cut 67건 중 정책 범위 22건, D Scheduler와 HTTP가 모두 비어 있던 Cut 2건을 관측했다. 두 건 모두 D Free Blocks가 6,143이었고 모든 후속 P Rows가 이미 발행돼 있었다.

| Suffix Tokens | 실제 비율 | D Scheduler 무작업 | 실제 HTTP 무작업 | 다음 다른 P-ready Oracle |
|---:|---:|---:|---:|---:|
| 2,842 | 9.85% | 260.3ms | 247.2ms | 1,011.1ms |
| 3,067 | 12.36% | 285.8ms | 266.7ms | 268.5ms |

전체 Cohort의 정책 범위는 51 Cut, D가 빈 경우는 6 Cut이었다. Counter 반복에서는 중앙 14 Cut 중 빈 D가 0건, 전체 48 중 빈 D가 2건이었다. 두 Cohort 모두 정책 범위 내에서 바로 실행할 수 있는 No-hold Cut은 0건이었다. Eager Native의 최초 Cohort는 Absolute Decode SLO 78%였다.

## Active A, B8 Eager

QPS5의 최초 Cohort에서 14 callback 중 1건, Counter 반복의 15 callback 중 3건이 실제 Suffix를 실행했다. QPS4에서는 최초 17 callback 중 2건, Counter 반복의 9 callback 중 0건이었다. 모든 callback에서 실제 Hold는 1회였고 P Worker의 Preemption Counter는 0이었다. 네 Cohort는 한 Server Start를 공유한다.

| Cohort | Suffix Tokens | Cut → D 첫 Token | E export RPC | 다음 다른 P-ready | 예약 중 도착한 Native | 최대 Native D 대기 |
|---|---:|---:|---:|---:|---:|---:|
| QPS5 최초 | 2,742 | 2,017.77ms | 1,731.80ms | 676.97ms | 5 | 1,341.38ms |
| QPS5 Counter | 1,122 | 352.30ms | 106.24ms | 420.34ms | 0 | — |
| QPS5 Counter | 1,811 | 1,022.74ms | 784.06ms | 1,069.66ms | 0 | — |
| QPS5 Counter | 3,197 | 2,735.47ms | 2,470.68ms | 737.89ms | 5 | 1,997.80ms |
| QPS4 최초 | 2,137 | 2,381.28ms | 2,130.55ms | 452.55ms | 8 | 1,928.85ms |
| QPS4 최초 | 2,605 | 201.88ms | 4.55ms | 744.44ms | 0 | — |

실제 Suffix 6건 모두 한 Image Span 내부를 잘랐고, D에서 나머지 이미지의 기존 P Inline Descriptor를 사용했으며 이미 P가 사용한 이미지 하나의 Descriptor를 E에 요청했다. QPS5 최초 요청의 D Submit부터 첫 Token까지는 약 1,993ms였다. 이 가운데 E export RPC가 1,731.80ms, Import·Clone·Sync의 Host 경계는 55.60ms였다. 이 요청을 포함한 세 실행은 D가 비어 있을 때 시작했지만 다음 P 완성 전에 끝나지 않았다. Native의 대기에는 SUFFIX_PREFILL_IN_PROGRESS가 기록됐다. 순수 GPU Suffix Kernel 비용은 별도로 계측해야 한다.

중앙 ROI에서 실행된 Suffix는 QPS5 Counter의 1,122-token 요청 한 건이며 해당 요청의 Decode SLO를 충족했다. 2,605-token 요청도 Decode SLO를 통과했다. 나머지 4건은 해당 요청의 Decode SLO를 충족하지 못했다. 실제로 실행되는 순간의 D 가용성, 요청별 안전성, 전체 서비스 이득을 각각 확인한다.

## Active A, B8 BI1 Spare Descriptor

실제 7.5–15% 남은 입력을 허용한 A로 QPS5/4/3을 실행했다. QPS5 최초 및 Counter에서 각각 11 및 9 callback 모두 D가 바빴다. QPS4 최초는 8개 중 0개, Counter는 8개 중 1개, QPS3 최초와 Counter는 각각 8개 중 1개 및 10개 중 1개 Suffix를 실행했다.

| Cohort | Suffix Tokens | Cut → D 첫 Token | E export RPC | 다음 다른 P-ready | 예약 중 도착한 Native | 최대 Native D 대기 |
|---|---:|---:|---:|---:|---:|---:|
| QPS4 Counter | 3,068 | 269.78ms | 0ms | 820.14ms | 0 | — |
| QPS3 최초 | 4,032 | 294.71ms | 0ms | 215.69ms | 2 | 79.26ms |
| QPS3 Counter | 2,842 | 244.52ms | 0ms | 220.23ms | 2 | 24.48ms |

세 실행에서 실제 D는 Spare Descriptor를 한 개씩 사용했고, Clone·Sync Host 경계는 10.59–12.27ms였다. D Engine의 Running/Pending/Unfinished는 예약 당시 0이었고 Free KV는 모두 6,143 Blocks였다. 세 Split 요청은 Decode SLO를 통과했다. 모든 실제 Split은 중앙 ROI 밖에서 발생했으며 QPS4 Counter의 한 건은 Drain이었다. QPS3에서 다음 Native-ready 요청에 남은 24–79ms 대기는 기존 긴 E export를 제거한 이후에도 D 완료까지의 기본 비용이 중요함을 보여준다.

E Producer는 별도 Crossing Descriptor 2,965개를 발행했다. 세 개를 D가 Claim하고 ACK했으며 나머지 2,962개는 unused-revoke 했다. 실험 종료 시 남은 Image 및 Export는 0개였다. 현재의 넓은 사전 준비 범위에서 사용한 Descriptor 비율은 약 0.10%다.

## 같은 Transport의 Native-Matched BI1 Stock

BI1 B8에 동일한 Spare 발행을 적용한 Native-Matched로 QPS4/5/3의 Cut을 기록했다. 검증된 Nominal 10/20%를 합한 실제 Suffix 범위는 7.5–25%다.

| QPS | 최초 중앙 내부 Cut | 검증 범위 최초 / Counter | 두 상태 모두 빈 D 최초 / Counter | 최초 관측 Suffix와 다음 다른 P-ready |
|---:|---:|---:|---:|---|
| 4 | 63 | 12 / 12 | 0 / 0 | — |
| 5 | 54 | 8 / 7 | 1 / 0 | 2,842 Tokens, 315.74ms |
| 3 | 45 | 6 / 6 | 1 / 0 | 7,077 Tokens, 616.00ms |

두 빈 D Cut의 실제 비율은 각각 9.85%, 23.18%였고 P에서 해당 Suffix의 Rows가 이미 모두 발행돼 있었다. 두 대상은 Discovery 요청이다. QPS4에서는 초기 조건과 Counter 반복 모두 정책 범위에 해당하면서 D가 빈 Cut이 없었다. 본 관측은 Scheduler와 HTTP 증거다.

## B-new의 실제 Suffix 실행

Native-Matched BI1 Spare의 QPS4로 P 모델을 보정했고, 별도의 BI1 Spare C0 비용을 사용했다. 500ms Cap 및 Point-gain을 실제 Main100 Natural에서 적용했다.

| QPS | Callback 최초 / Counter | 실제 Split 최초 / Counter | 실제 Tokens 최초 / Counter | 중앙 ROI 실제 Split |
|---:|---:|---:|---:|---:|
| 5 | 24 / 26 | 1 / 1 | 1,482 / 1,752 | 0 |
| 4 | 28 / 32 | 2 / 3 | 3,234 / 5,989 | 1, Counter |
| 3 | 24 / 28 | 2 / 3 | 3,234 / 6,076 | 0 |

162 callback 중 140건에서 D가 이미 바빴다. 같은 P Step의 다른 완료 Output으로 4건, 비용 지원 범위·예산·예상 이득으로 각각 2건씩 P에서 계속 처리했다. 실제 Suffix 12건은 모두 Spare를 Claim했으며 E export RPC는 0ms였다. 한 건만 중앙 ROI에 있었고 나머지는 중앙 구간 이전에 발생했다. 마지막 Arrival 이후 Drain의 실제 실행은 없었다.

12건 중 11건이 해당 요청의 Decode SLO를 통과했다. 여섯 Suffix 동안 Native-ready 요청 일곱 개가 들어왔고 예약 해제까지 관측된 최대 대기는 70.10ms였다. 중앙의 2,755-token Cut은 예상 예산 379.05ms, 예상 비용 상한 304.22ms, 실제 점유 294.09ms였다. 같은 실행에서 다른 P 요청이 236.79ms에 준비되며 57.44ms를 기다렸다. QPS3 Counter의 1,752-token Suffix는 실제 311.16ms로 예산 244.56ms 및 비용 상한 228.51ms를 함께 넘었다. 이 경우 Clone·Sync Host 경계가 188.62ms였고, 이전 P 완료가 충분히 늦어 추가 Native 대기는 없었다.

E Producer는 2,965개의 Spare를 발행했고 실제 사용된 12개를 ACK했으며 나머지 2,953개를 회수했다. 종료 시 남은 Image 또는 Export는 0이었다. 정책이 실제 Suffix를 실행했다는 근거와 지속 부하의 활성 여유 및 TTFT 이득은 각각의 관측을 사용한다.

## C-Hard의 실제 Cut 제한

같은 BI1 Slice·Spare 및 P-ready 모델을 사용한 C-Hard는 QPS5/4 최초 및 Counter에서 총 네 Cohort를 완료했고 실제 Suffix 실행은 0건이었다. QPS5 최초의 23 callback은 모두 D가 바빴다. Counter는 D Busy 27건과 같은 P Step의 다른 완료 1건이었다.

QPS4 최초 및 Counter에서 각각 3건과 2건이 D HTTP가 비어 있었지만 Image 내부 Cut으로 거절됐다. 두 Cohort 모두 B-new가 실제 Suffix를 실행했던 MP-DocVQA 1953의 1,752-token Suffix와 MP-DocVQA 41691의 1,482-token Suffix를 같은 Prefix 위치에서 거절했다. 이 결과는 같은 Slice Transport에서 Hard Boundary 제약이 줄인 선택 기회를 실측으로 보여준다.

## B-new Equal-output Replay QPS5

최초 및 Counter Cohort 모두 100개 요청에서 5,269 Output Tokens를 완료했다. 최초 28 callback 중 26개는 D Busy, 2개는 실제 Split이었다. 1,482 및 2,755 Tokens를 이전했고, 뒤의 건은 중앙 ROI에 있었다. Counter 31 callback에서는 D Busy 29건, 예산 부족 1건, 실제 1,752-token Split 1건이었다. 세 경우 모두 Spare를 Claim했고 E export RPC는 0ms였다.

최초 실행의 1,482-token Suffix가 Native-ready 한 건을 54.57ms 기다리게 했다. 중앙 2,755-token Suffix는 Native-ready 두 건에 37.76–38.32ms 대기를 발생시켰다. 실제 Suffix 세 건 중 두 건이 해당 요청의 Decode SLO를 통과했다. Natural EOS와 이 Replay 결과는 각각의 Workload 결과로 보존한다.

[Equal-output Replay의 실제 Cut과 실행 기록](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bnew_bi1_preissue_replay) · [C-Hard의 실제 Cut과 거절 사유](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/chard_bi1_preissue_natural) · [B-new Natural 실제 Cut과 실행 기록](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/bnew_bi1_preissue_natural) · [Native-Matched BI1 Spare의 Stock Cut](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_b8_bi1_preissue) · [Active A BI1 Spare의 Cut별 Funnel](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/a_bi1_preissue_natural) · [Active A Eager의 Cut별 Funnel](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/a_eager_slice_natural) · [Active A BI1의 Cut별 Funnel](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/a_bi1_slice_natural) · [B8 Eager Stock Cut](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_b8_eager) · [Free BI0 Stock Cut](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/stock_free_bi0) · [이전 Shadow의 Cut별 CSV와 Funnel](../mllm_epd_h200_dynamic_correctness_slice_aware/results/slack/history) · [분석 코드](../mllm_epd_h200_dynamic_correctness_slice_aware/cut_slack.py)
