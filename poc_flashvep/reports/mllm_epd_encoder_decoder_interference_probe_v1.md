# E/D 실제 Prefill Probe — 독립 N3 간섭 검사

증거는 `INTERFERENCE_PROBE`다. Probe는 실제 요청의 완료에 기여하지 않는다.
Main100은 매번 실제 Online E/P/D를 실행했다. Probe만 별도 Calibration Embedding을 미리 준비했다.
추가 계산으로 Client Throughput이 개선됐다고 주장하지 않는다.

## 계약

GPU4/5/6, Original BF16 Qwen2.5-VL-7B, BI0, Full28, 원본 Vision/Processor를 유지했다.
모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.
B8192/S1024, P/D KV42GiB, Block128, Prefix Cache Off다. E Helper는 추가 Full Model과 KV8GiB를 사용했다.
복제된 Vision Weight 메모리도 포함했다. 최대 Encoder Batch Capacity는 측정하지 않았다.
각 Block은 새 Stock Native와 별도의 Common/Probe 초기화를 사용했다.
N1/N8 출력 검사와 E/D128/512의 Same-GPU Teacher KV 검사를 통과한 뒤 측정했다.
Clean/Profile 계측을 Main에 삽입하지 않았다. E HTTP Enclosure는 GPU-exclusive Encode Time이 아니다.
E는 Pending E HTTP가 없고 P HTTP 작업이 있을 때만 Probe를 허용했다.
D는 기존 Stream과 Pending Admission이 모두 없을 때만 Probe를 허용했다.
새 Encode Arrival은 이미 실행 중인 Helper Kernel을 기다릴 수 있다. Stream Priority를 Preemption으로 가정하지 않았다.
128/512는 원래 Full28 Prefix다. Imported Prefix KV를 사용하는 Suffix 비용이 아니다.

## 사전 고정한 보호 Gate

| Mode | Probe | Relative HTTP Protection | Absolute Decode≥95% | Actual Additional Work | Decision |
|---|---|---|---|---|---|
| natural | E128 | False | False | True | PROBE_PROTECTION_NOT_CONFIRMED |
| replay | E128 | False | True | True | PROBE_PROTECTION_NOT_CONFIRMED |
| natural | E512 | False | True | True | PROBE_PROTECTION_NOT_CONFIRMED |
| replay | E512 | False | True | True | PROBE_PROTECTION_NOT_CONFIRMED |
| natural | D128 | False | False | True | PROBE_PROTECTION_NOT_CONFIRMED |
| replay | D128 | False | False | True | PROBE_PROTECTION_NOT_CONFIRMED |
| natural | D512 | False | True | True | PROBE_PROTECTION_NOT_CONFIRMED |
| replay | D512 | False | True | True | PROBE_PROTECTION_NOT_CONFIRMED |

NativeStrong·NativeCommon·동일 초기화의 No-probe 평균을 각각 대조군으로 사용했다.
E HTTP p95·ITL p99·TPOT p95·TTFT p50/p95의 Paired CI 상한은 각각 1.05 이하여야 한다.
Absolute Decode는 Request별 All-ITL과 TPOT의 AND다. 각각의 Pass Rate만 검사하지 않았다.
N3의 모든 27개 Paired Resample을 사용했다. CI 해상도는 제한적이다.
HTTP 보호 Gate를 통과해도 미래 GPU Safe Window나 Prefill-heavy를 확인한 것은 아니다.
실제 Encoder-exclusive Service p95의 상대 Gate는 미측정이다. 이를 HTTP Enclosure로 대체하지 않았다.
이 결과만으로 Full ENCODE_SAFE_HEADROOM_CONFIRMED 또는 Online Token Split GO를 부여하지 않는다.
모든 실패 조합의 Request Count와 Token Weight를 보존했다. 전체 Task Quality도 계속 PENDING이다.
Probe 순서는 Clean-before, E128, E512, D128, D512, Clean-after로 고정했다. 순서 효과를 완전히 제거하지 못했다.

## 실제 메모리 표본

모든 Block의 Helper 추가 전·추가 후·마지막 표본을 CSV로 보존했다.
E와 Helper는 GPU4를 공유한다. Device Used는 E 표본 한 번만 사용했다.
Process Allocated와 Reserved는 별도로 표시했다. Reserved를 Weight 크기로 해석하지 않았다.
Model Parameter의 논리적 Bytes와 실제 Unique KV Storage Bytes를 별도로 저장했다.
Peak Allocated는 Process 시작 이후의 최대값이다. Cohort별 Activation Peak가 아니다.
Helper 생성 전의 Helper 메모리는 NOT_APPLICABLE이다. 이를 실제 0-byte 측정값으로 채우지 않았다.
Helper KV8GiB와 D KV42GiB는 다르다. 동일한 KV Capacity 또는 최대 Encoder Batch를 입증한 것은 아니다.
남은 Device Memory가 있더라도 안전한 Prefill Window나 Encoder 성능 보호를 의미하지 않는다.
Original C40R1의 Port Collision 실패와 그 수정은 별도 보고서에 있다. 실패한 실행은 삭제하지 않았다.

## 재현 자료

Config는 `configs/interference_PROBE_C40R4.json`이다. Main100 Manifest와 SLO는 변경하지 않았다.
CSV/JSON은 `analysis/interference_PROBE_C40R4_*_v0`에 있다. Figure는 `figures/interference_PROBE_C40R4_*_v0`에 있다.
Probe의 실제 HTTP 비용·수락/거부·추가 Token-layer Rows는 별도 CSV에 있다.
추가 Probe의 1-token Sample은 버렸다. Main Request의 Output Token 수에 더하지 않았다.
Natural EOS와 Equal-output Replay의 결과를 합쳐 평균을 계산하지 않았다.
