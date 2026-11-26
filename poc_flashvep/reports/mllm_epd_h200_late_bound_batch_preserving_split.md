# Method A Late-bound Fixed Split

P가 이미 계산한 실제 chunk 경계에서만 결정을 요청한다. D가 full-prefill 요청을 갖고 있거나 기다리는 경우, 기존 P 요청이 같은 cache와 위치에서 다음 chunk를 계산하는 경로를 검사한다. 승인 시에는 이미 생성한 prefix KV를 정확히 지정한 상태로 이전하고 D에서 원본 suffix를 처리한 뒤 native decode queue에 합류한다.

| 검증 경로 | 확인할 증거 |
| --- | --- |
| D 예약 승인 | 예약 snapshot, P actual computed boundary, KV handoff, Suffix first-token, 실제 mixed Native/Split decode batch |
| D active 또는 pending | 늦은 D 대기 없이 예약을 거절하고 동일 P request에서 토큰을 순서대로 이어서 계산. 전후 실행 counter와 request ID |
| KV 부족, 경쟁 예약, timeout | 시작 전 거절 원인, 예약 및 buffer 반환. 불확실한 이전 이후 재실행의 semantics와 비용 별도 기록 |
| 내부 및 경계 Cut | 동일 Vision embedding, 위치, Prefix/Final Full28 KV와 Greedy output. 20/30%를 인증할 때 같은 comparator 사용 |

약 10%는 목표 suffix이며 실제 남은 길이는 P scheduler cut에 의해 결정된다. 새 boundary 삽입 시에 생기는 pause 및 제어 비용도 TTFT에 포함된다. 이미 처리한 P 토큰 수와 실제 재실행 토큰 수를 별개로 기록한다. 정합성 검사는 실측 enqueue와 cleanup의 대체가 아니다. [이전 정확성 범위](mllm_epd_generic_token_split_correctness.md)

## H200 상태 Probe 결과

직렬 7개 사례가 통과했다. 내부 cut의 5/10/20/30%, image boundary, 27,840-token 문서의 30%, 그리고 예약을 거절한 10% P continuation을 포함한다. 동일 요청의 P query range, Full28 prefix 및 final KV, 원래 token ID와 MRoPE, Greedy sequence를 비교했다. Forced continuation 사례는 기존 P 요청을 유지했고 Full28 cache와 출력이 일치했다.

| Mixed Natural N8 | D suffix / P continuation | Final KV tolerance 및 query coverage 통과 | Greedy 일치 | D prepared decode 최대 | D single-request forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 3 / 5 | 3/8 | 8/8 | 6 | 62.44% |
| 10% | 2 / 6 | 3/8 | 6/8 | 6 | 67.81% |
| 20% | 2 / 6 | 6/8 | 7/8 | 6 | 63.46% |
| 30% | 2 / 6 | 6/8 | 8/8 | 6 | 63.77% |

Mixed probes는 도착이 겹치는 correctness 시험이다. D count는 prepared forward에서 실제 decode phase 요청만 센 것이다. 여러 요청이 같은 forward에서 실행됐지만 이 증거는 clean C3와 동등한 batch 효율을 측정하지 않는다. 10%의 233개 decode-containing forward에서 step-weighted mean은 1.494였다.

예약을 실제 실행한 9건의 P-ready→D-submit host 경계는 0.059–0.175 ms였고, D reserve→first token은 113.84–377.36 ms였다. 이는 별도 tensor 및 batch probe가 켜진 구간이다. 이를 기존 Idle10 QPS5 wait p95와 직접 비교해 TTFT 개선으로 귀속하지 않는다. 10% mixed의 cache tolerance를 넘긴 5건은 P continuation이었다. 두 split의 imported prefix와 Greedy는 일치했다. 직렬 teacher 대비 혼합 실행 차이가 관측됐고 그 차이의 원인은 이 probe 한 개로 특정할 수 없다.

Mixed correctness는 네 비율에서 충족되지 않았다. [정합성 요약과 진단 host 시간](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_correctness.json)

## 동일 E Embedding으로 분리한 Native 대 Late

Serial E lock과 post-E barrier를 적용하고 P/D는 요청을 함께 실행하게 했다. 요청 8개를 비율 4개에서 반복한 32개 사례 모두 각 arm의 E embedding이 hash 기준 byte-identical이었다. 모든 Native 및 late 경로에서 관측한 NIXL import 1,792개 layer segment가 해당 export와 일치했다. 두 arm의 독립 P/D scheduler로 인해 actual batch geometry는 달랐다. 아래 통과 수는 조건별 비교다.

| Forced cut | D suffix / P continuation | Late 대 Native Full28 tolerance | Late 대 Native exact Greedy | 두 비교와 원본 상태 통과 |
| --- | ---: | ---: | ---: | ---: |
| 5% | 2 / 6 | 4/8 | 8/8 | 4/8 |
| 10% | 1 / 7 | 5/8 | 7/8 | 4/8 |
| 20% | 2 / 6 | 7/8 | 8/8 | 7/8 |
| 30% | 1 / 7 | 8/8 | 7/8 | 7/8 |

10%의 tolerance 실패 3건은 모두 P continuation이었다. 세 건 모두 P prefix 28층은 byte-identical이었고, P suffix의 layer relative L2 최대치는 요청별 2.90%, 3.87%, 6.00%였다. D import 이전 P에서 같은 차이가 이미 보였으므로 이 세 건은 E embedding 변화나 NIXL 전송 없이 재현됐다. 별도로, 실제 D로 이전한 10% 요청 하나는 prefix 28층이 byte-identical이고 D에서 완성한 prompt KV는 tolerance 이내였으나 Greedy sequence가 달랐다. 동시 실행하는 uncut Native 자체도 serial teacher와 비교하면 KV tolerance를 통과하지 않는 요청이 있었다. 독립 D 준비 geometry의 차이는 이 probe로 통제하지 않았다.

네 forced cut 모두 동시성 하에서 full gate를 통과하지 못했다. 30%가 cache tolerance를 통과했지만 exact Greedy 7/8이었다. [Serial-E 4×N8 진단 receipt](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_serial_E_geometry.json)

## Stock P Chunk의 자연 경계와 추가 제어 비용

Forced arm은 목표 cut에서 다음 P query row 수를 실제로 줄인다. Natural arm은 P의 기존 scheduler가 발행한 chunk를 그대로 실행한 뒤 퇴역한 prefix를 검사한다. 다음 chunk가 이미 실행 중이면 그 구간을 건너뛰고 같은 요청을 P에서 이어간다. Natural-hold는 stock scheduler가 후보 chunk를 처음 발행한 뒤부터 다음 row의 발행만 잠시 보류한다. 퇴역 지점, 보류 실행 시도 횟수 및 보류 시간을 기록한다. 이 보류의 batch geometry 및 host 비용은 별도 gate의 대상이다. [Forced / natural boundary와 hold 구현](../mllm_epd_h200_modality_aware_dynamic_split/late_worker.py#L95)

Inline-redirect 후보에서는 E가 기존대로 P용 descriptor를 제공하고, late commit 뒤 P가 사용하지 않은 사본을 D가 한 번 가져갈 수 있다. P가 같은 원본 이미지 descriptor를 이미 사용한 경우 D가 별도 export를 요청한다. 그래서 P와 D가 같은 image span을 읽는 cut의 full-image clone 메커니즘은 남아 있다. 실제 clone과 raw byte 관측은 표의 격리 실험으로 제한된다. [Inline claim / fallback과 clone 기록](../mllm_epd_prefill_heavy_token_split/cuda_ipc_ec.py#L181)

보류를 사용하지 않는 serial no-hold probe에서 6/6 요청이 P 전체 계산을 완료했고 실제 D suffix는 0/5였다. 다섯 시험은 목표 위치보다 먼저 또는 나중에 실행이 끝나는 기존 chunk가 발행됐다. 12,053-token 문서에서는 목표 8,192 위치가 실제로 퇴역했지만 다음 3,861 rows가 이미 실행 중이었다. 동일 요청의 원래 chunk 기록은 아래와 같다. 이 정확한 목표 위치 시험은 일반 ratio band 전체에 대한 수용률을 측정하지 않는다.

| 요청 유형 | 목표 Prefix | 실제 P prefill chunk 종료 위치 |
| --- | ---: | --- |
| 신규 split 및 continuation에서 사용한 27,840-token 문서 | 24,576 | 8,192 / 16,083 / 24,275 / 27,840 |
| 18,237-token 문서 | 16,384 | 8,192 / 15,947 / 18,237 |
| 17,334-token 문서 | 16,384 | 8,192 / 16,083 / 17,334 |
| 21,193-token multi-image | 16,384 | 8,192 / 15,929 / 21,193 |
| 12,053-token 문서 | 8,192 | 8,192 / 12,053 |

Natural-hold는 마지막 문서 요청에서 stock 8,192-row chunk 한 번을 유지하고 다음 발행을 한 차례 보류했다. 3,861-token suffix가 D에서 실행됐고, Full28 imported prefix와 D final prompt KV가 직렬 teacher와 각각 28/28 byte-identical이었다. 원본 token ID와 MRoPE 및 Greedy sequence도 일치했다. Observed hold-to-retirement는 0.463 ms, decision host enclosure는 7.678 ms였다. 이는 검사 instrumentation이 켜진 한 건의 serial 증거다. 나머지 다섯 지정 위치는 해당 실행에서도 P가 발행하는 자연 경계에 없었다.

후속 inline-redirect hold에서는 위에서 실제 관측한 stock 위치를 지정했다. 실제 P chunk의 위치와 EC descriptor 방식이 함께 바뀌었으므로 이 실행은 transport만 변경한 대조군과 구분한다. 다섯 D suffix가 모두 실제 실행되어 직렬 teacher와 Full28 imported prefix 및 final prompt KV, 원본 입력·MRoPE, Greedy 출력이 일치했다.

| 실제 P prefix / 전체 input | 실제 D suffix | 실제 ratio |
| --- | ---: | ---: |
| 24,275 / 27,840 | 3,565 | 12.81% |
| 15,947 / 18,237 | 2,290 | 12.56% |
| 16,083 / 17,334 | 1,251 | 7.22% |
| 15,929 / 21,193 | 5,264 | 24.84% |
| 8,192 / 12,053 | 3,861 | 32.03% |

다섯 승인 사례의 보류→퇴역 host 구간은 0.326–0.386 ms, decision은 5.044–7.868 ms였다. 의도적으로 예약을 거절한 별도 27,840-token continuation의 P query는 기존 요청에서 중복 없이 완료됐다. P final prompt와 실제 D import는 28/28 byte-identical이었고 D 마지막 prompt query도 28/28 허용오차 이내였다. Greedy는 serial P decode teacher에서 56 tokens, late continuation 뒤 D decode에서는 60 tokens였다. 이 사례의 llm_probe tensor capture를 켠 퇴역 대기값 259.330 ms는 진단 receipt에 남겼다.

같은 요청을 D에서 decode한 no-hold full-prefill 대조와 hold 거절 경로를 다시 비교하자 실제 P query row 수열 [8192, 7891, 8192, 3565], P와 D tensor bundle SHA256, Greedy 60개 token ID 전체가 일치했다. 이 serial 대조에서 P continuation의 상태를 확인했다. [동일 D 경로의 Natural continuation 대조](../mllm_epd_h200_modality_aware_dynamic_split/results/natural_continuation_pair.json)

관측한 stock 위치와 inline-redirect를 적용한 뒤 휴지 없이 같은 요청을 다시 실행했다. 여섯 지정 위치는 모두 실제 퇴역했지만 다음 P chunk가 이미 실행 중이었고, 추가 실행 중인 rows는 사례별 1,251–5,264개였다. 다섯 D split 시험 모두 0%로 완료했다. No-hold와 hold의 제약을 실제 발행 상태에서 보여준 결과이며, 이 serial 대조의 결과를 C3의 후보 출현율로 일반화하지 않는다.

[Natural boundary 정합성 receipts](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_natural_correctness.json)

## Natural-hold 혼합 N8

Inline-redirect와 serial E 제어를 사용해 Native 및 natural-hold를 각각 독립 스케줄했다. 명목상 10% 및 20% 조건 모두 E embedding 8/8, 실제 query coverage·paired Full28 KV tolerance 및 Greedy 8/8을 통과했다. 각 Native arm에서 계산된 적격 stock 후보는 각각 1건과 2건이었으나 late arm에서는 지정된 내부 위치에서 offer가 발생하지 않았다. 16/16건이 전체 P 계산을 이어갔고 실제 D suffix는 0이었다.

| Nominal target | Native / late D 실제 decode 최대 | Native / late D step mean | Native / late D single-request forwards |
| --- | ---: | ---: | ---: |
| 10% | 4 / 3 | 1.461 / 1.278 | 59.09% / 76.70% |
| 20% | 2 / 3 | 1.082 / 1.331 | 91.83% / 68.05% |

표는 prepared forward에서 decode 단계의 요청만 센다. 두 arm의 독립 scheduler geometry가 달라도 여러 요청이 실제로 함께 decode하고 no-split 상태가 일치했다. Mixed test에서 실제 cross-GPU suffix와 continuation을 함께 검증하는 gate는 두 조건에서 모두 미충족이었다. [Natural Mixed per-request geometry](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_natural_mixed_geometry.json)

## 실제 D suffix와 Native follower가 겹치는 검증

이전 serial 실행에서 관측한 stock cut을 재사용했다. 두 arm 모두 첫 요청을 먼저 P로 보내고, 첫 요청의 P 응답 뒤 대기 중인 일곱 요청을 풀었다. 이 동일한 도착 제어로 실제 D suffix 뒤에 첫 요청이 Native follower와 같은 prepared decode forward에 들어가는지 검사했다. 이 제어는 상태 검증용이다.

| 실제 cut | D suffix / Native | Lead와 follower의 D co-batch steps, Native / Late | Co-batch 최대, Native / Late | 원본 상태와 paired KV tolerance | 전체 exact Greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image 내부, 24,275 / 27,840 | 1 / 7 | 31 / 41 | 5 / 8 | 8/8 | 7/8 |
| Image 경계, 15,929 / 21,193 | 1 / 7 | 12 / 43 | 2 / 8 | 8/8 | 7/8 |

두 승인 요청의 imported prefix는 28/28층에서 export와 byte-identical이었다. Native 대비 D에서 완성한 suffix의 최대 layer relative L2는 각각 8.84×10⁻⁵와 7.77×10⁻⁵였다. Interior lead는 Native에서 60 tokens, Late에서 56 tokens였고 0부터 센 token index 13에서 처음 달랐다. Image-edge lead의 58 tokens는 일치했다. 대신 suffix를 이전하지 않은 Mantis-Eval follower 한 건이 74→60 tokens로 바뀌었다. 해당 follower의 P query는 Native [2318, 2714]에서 Late [5032]로, 첫 차이의 D prepared batch는 2에서 3으로 바뀌었다. 관측된 geometry와 출력 차이는 같은 실행에서 발생했다. D batch와 P chunk의 영향을 이 검증 하나로 분리할 수 없다.

실제 D suffix를 끝낸 요청이 Native follower와 함께 최대 batch8로 decode하므로, 이 실행 경로는 Static10의 batch1 직렬화를 재현하지 않았다. Contiguous exactness 및 압축 없는 KV 이전을 확인했지만 두 조건 모두 전체 Greedy gate는 미달했다. [실제 mixed route·layer tolerance·batch 증거](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_natural_lead_geometry.json)

NativeCommon도 같은 시작의 clean-before/after에서 output drift가 있었다. 동일 Main100과 arrival schedule에서 전체 token ID가 일치한 요청은 B8 86/100, Free 88/100이었다. 총 Natural output은 B8 5,322→5,463, Free 5,458→5,508이었다. 이 A/A 결과는 기존 runtime의 greedy 재현 차이를 보여준다. 실제 suffix를 포함한 두 검증에는 기존 exact gate를 그대로 적용했다. [Native A/A 원자료 요약](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_native_aa_summary.csv)

Method A의 QPS5 Natural shadow는 실제 stock promotion callback이 13/100건에서 발생했다. 그중 6건의 D reservation snapshot이 통과했고 후보 suffix는 합계 12,193 tokens였다. 실제로는 100건 모두 P가 전체 입력을 계산했다. Callback이 있었던 요청에서 callback host 합의 p95는 4.492 ms였다. 이 값에는 active suffix, D reserve와 후속 Decode에 대한 지연이 포함되지 않는다. [QPS5 shadow와 비교 기준선](mllm_epd_h200_generic_vs_modality_aware_online_serving.md#qps5-shadow-natural-eos)
