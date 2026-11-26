# H200 Imported-KV Suffix Cost

비용의 기본 측정량은 실제 P-prefix 전달에서 D 최초 출력까지의 경과 시간이다. P export, D KV import, EC 준비, scheduler admission, full28 suffix forward, sampling, delivery 및 마지막 buffer release가 개별 event가 있는 범위에서 별도 기록된다. Host HTTP에는 구성 단계가 중첩되므로 포함된 handoff 비용을 predictor에 다시 더하지 않는다.

| 계약 | 정의 |
| --- | --- |
| 요청 선택 | 동결 Calibration32에서 입력 길이 구간과 image 구조를 고르고, 128/256/512/1024/2048/4096 tokens와 0/5/10/20/30%의 대표 geometry를 제한적으로 커버 |
| Attention work | p×n + n×(n+1)/2 causal attention pairs. p는 실제 prefix, n은 실제 suffix |
| Generic feature | 입력·prefix·suffix 길이, KV context와 occupancy, 실제 batch, P queue, D active/pending, 관측 handoff 및 실행 |
| 검증 | 같은 요청의 반복을 양쪽 split에 넣지 않음. Context-disjoint heldout MAE, 상대오차, p95 underprediction, 구축 가능한 경우 실제 budget 기준 false-safe 및 과도한 거절 |
| 정확성 | 원본 embedding 순서·MRoPE·Full28 prefix/final KV·Greedy, N1/N8과 실제 mixed online coverage를 별개 기록 |

이전 H100의 NIXL suffix cost는 15개 geometry를 검증했고 Output1 Diagnostic을 수행했다. D HTTP clean p95는 입출력 길이와 상태에 따라 비단조적이었다. GPU 시간은 stream enclosure였고 당시 전체 Observer Gate는 실패했다. 여기서의 H200 비용 적합에는 그 값을 사용하지 않는다. [이전 imported KV cost](mllm_epd_imported_kv_suffix_quantum_cost_v1.md)

## H200 on-demand 결과

두 반복에서 총 310개 trial을 완료했다. 비용 238개와 transfer 72개다. 비용 arm은 실제 입력 2,044–30,195 tokens와 128/256/512/1024/2048/4096 고정 suffix 및 nominal ratio 후보를 포함했다.

| Nominal suffix | D one-token HTTP p50 (ms) | p95 (ms) | 원본 unsplit 대비 첫 Greedy token 일치 |
| ---: | ---: | ---: | ---: |
| 0% | 51.49 | 98.23 | 30/30 |
| 5% | 105.89 | 181.84 | 30/30 |
| 10% | 130.36 | 285.12 | 30/30 |
| 20% | 165.79 | 347.35 | 30/30 |
| 30% | 152.79 | 481.44 | 30/30 |

이 지연에는 실제 KV import와 embedding 준비, D scheduler/forward, sampling, HTTP 응답이 포함된다. 위 값은 여러 입력 geometry를 모은 비율별 분포다. 입력 길이와 실제 잔여 토큰의 영향은 Figure 3의 원시 geometry별 비용을 참조한다. 첫 토큰 일치는 최종 Full28 KV 및 Natural sequence 전체의 정확성 검사를 대신하지 않는다.

별도 Scheduler/stream-enclosure 진단은 8,352와 9,059개의 suffix가 실제로 8,192+160, 8,192+867 두 forward로 나뉨을 확인했다. 원 데이터는 worker stream enclosure이며, GPU-exclusive kernel time은 NOT_MEASURED다.

[전체 비용 CSV](../mllm_epd_h200_modality_aware_dynamic_split/results/suffix_calibration.csv) · [실행 영수증](../mllm_epd_h200_modality_aware_dynamic_split/results/suffix_calibration_receipt.json) · [D forward 진단](../mllm_epd_h200_modality_aware_dynamic_split/results/suffix_forward_diagnostic.json) · [Figure 3](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig03_h200_suffix_cost.pdf)

## 같은 geometry의 inline-redirect 결과

별도 runtime 시작에서 동일한 310개 trial을 다시 완료했다. 실제 D에서 inline descriptor를 재지정해 사용한 이미지는 406장이다. 238개 비용 trial의 inline−on-demand host 시간 중앙값은 E −6.20 ms, P −2.46 ms, D +8.45 ms였다. 동일 geometry라도 두 실행은 서로 다른 단일 시작이다.

| Nominal suffix | D one-token HTTP p50 (ms) | p95 (ms) | 원본 unsplit 대비 첫 Greedy token 일치 |
| ---: | ---: | ---: | ---: |
| 0% | 53.42 | 101.75 | 30/30 |
| 5% | 109.77 | 210.83 | 30/30 |
| 10% | 140.94 | 290.73 | 30/30 |
| 20% | 179.14 | 379.78 | 30/30 |
| 30% | 185.14 | 507.03 | 30/30 |

모든 310개 trial에서 첫 토큰이 일치했고 8,352 및 9,059-token suffix는 기존과 같은 두 chunk로 처리됐다. Shadow의 B/C는 이 inline transport의 비용 모델을 사용했다. On-demand 보정과 inline 보정을 섞어 fit하지 않았다. [Inline 전체 비용과 실제 경로](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/suffix_calibration.csv) · [Inline 실행 영수증](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/suffix_calibration_receipt.json) · [Inline Figure 3](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig03_h200_suffix_cost.pdf)
