# CUDA Event 실행 구간 계측 — C19

증거 수준은 `DIAGNOSTIC`이다. 계측은 고정한 5% Gate를 통과하지 못했다.
아래 수치를 Primary GPU Kernel Busy 또는 Decode-safe Window로 사용하지 않는다.

Native EPD를 H100 GPU4/5/6에서 실행했다. 모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.
원래 Compiled 경로, Main100, QPS2, 동일 Arrival과 5269 Tokens의 Equal-output Replay를 사용했다.
독립 초기화는 1회다. 절차는 Clean/Profile/Clean이다. 추가 Prefill 삽입은 없었다.

Worker의 원래 `execute_model` 호출을 CUDA Event 두 개로 둘렀다.
매 Step의 GPU Synchronization은 추가하지 않았다. Begin/End의 Clock Anchor에서만 기다렸다.
Event 구간에는 Host Submission Gap과 Stream Dependency가 포함될 수 있다.
별도 Stream의 Copy 또는 Execute 외부 Export는 포함하지 않을 수 있다.
따라서 이 구간은 GPU-exclusive Kernel Service가 아니다. 구간 밖 시간도 곧바로 Physical Idle이 아니다.

## 계측 검증

Profile 변화율의 분모는 두 Clean 값의 평균이다. Clean Spread도 같은 분모를 사용한다.
Gate는 변화율의 절댓값과 Clean Spread가 모두 5% 이하여야 한다.
Profile이 더 빨랐다는 이유로 이 기준을 생략하지 않았다.

| Metric | Profile 변화율 | Clean Spread | Gate |
|---|---:|---:|---|
| Duration | −1.3283% | 0.9784% | PASS |
| TTFT p50 | +2.7853% | 7.2993% | FAIL |
| TTFT p95 | −5.3064% | 11.1018% | FAIL |
| ITL p99 | −6.3955% | 4.1594% | FAIL |
| TPOT p95 | +0.3588% | 1.3896% | PASS |

이 결과로 계측 Overhead의 단독 크기를 확정할 수 없다. 시간에 따른 실행 변동도 존재한다.
기존 CUPTI 계측의 실패도 그대로 보존했다. 새 계측이 기존 실패를 Primary PASS로 바꾸지는 않는다.

## 실행 구간의 진단값

Central Arrival Window는 예정 Arrival의 p10–p90 구간이다. 길이는 39.293247초다.
이 구간을 Sustainable Steady State로 단정하지 않는다.

| Stage | Event 실행 구간 비율 | 실행 구간 밖 비율 | 관측 Waiting 평균 | 관측 Waiting 최대 |
|---|---:|---:|---:|---:|
| E | 52.8750% | 47.1250% | 0.0000 | 0 |
| P | 66.9446% | 33.0554% | 0.8963 | 9 |
| D | 51.5101% | 48.4899% | 0.0000 | 0 |

Waiting은 Scheduler Sample의 단순 평균과 최대다. 정확한 시간 가중 Queue는 아니다.
관측 Sample에서 Queue가 0이었다고 모든 순간의 Queue가 0이라고 주장하지 않는다.
전체 Capture의 최대 Scheduled Requests는 E6/P5/D4다. 이는 Central Window만의 최대값이 아니다.
B8192/S1024와 Actual Scheduled Batch를 구분한다. E의 별도 Token Budget은 114688이다.

E와 D가 동시에 실행 구간 밖인 비율은 Central Window의 21.1177%다.
그 상태와 P 실행 구간이 겹친 비율은 Central Window의 12.0869%다.
같은 겹침은 P 실행 구간 길이의 18.0551%다. 분모가 서로 다르다.
이 비율은 Physical Idle Overlap 또는 Actionable Slack Overlap이 아니다.

E의 실행 구간 밖 Window는 p90 709.161ms, p95 1009.473ms였다.
D는 p90 0.023437ms, p95 1.519528ms였다. Window Count 기준의 Percentile이다.
각 Stage에는 Execute 호출 사이의 매우 짧은 Gap도 포함된다.
원시 CSV에는 Count 기준과 Duration 가중 Threshold 분석에 필요한 모든 Window를 보존했다.

## 판정과 후속 검사

`DECODE_SAFE_HEADROOM_NOT_CONFIRMED`와 `ENCODE_SAFE_HEADROOM_NOT_MEASURED`를 유지한다.
낮은 QPS0.5에서 같은 계측의 안정성을 별도 확인한다. 이 결과와 QPS2를 합쳐 평균하지 않는다.
Low-load에서 계측이 통과하더라도 High-load Headroom을 자동으로 인정하지 않는다.

원시 Event와 Clock Anchor는 `raw/gpu_enclosures/ec_GPU_ENCLOSURES_C19_CPC_q2_v0/`에 있다.
검증은 `analysis/ec_GPU_ENCLOSURES_C19_CPC_q2_v0_qualification.json`에 있다.
진단 요약과 CSV는 `analysis/ec_GPU_ENCLOSURES_C19_CPC_q2_v0_enclosure_analysis_v0.json`과 대응 CSV에 있다.
CDF는 `figures/ec_GPU_ENCLOSURES_C19_CPC_q2_v0_outside_bracket_CDF_v0.{png,pdf,csv}`다.
경로는 모두 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.
