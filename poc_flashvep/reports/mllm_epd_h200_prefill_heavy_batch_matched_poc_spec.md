# H200 MLLM EPD Prefill-heavy 및 Batch1 Token Split 실험 사양

## 목표와 근거

H200 3대의 실제 E/P/D 온라인 서빙에서 P 계산 포화와 D의 no-active-decode 구간이 공존하는 조건을 찾는다. 같은 실제 D Decode Batch1에서 단순 contiguous suffix split의 TTFT 개선과 ITL·TPOT 보호를 검사한다. E의 연산 역할은 원래 Vision Encode로 유지한다. E가 바쁜 조건도 포함하며, E의 공급 부족으로 P가 쉬는 조건은 P-heavy로 판정하지 않는다. 2026-11-25 Notion 상단 결정과 이번 캠페인 지시가 이전 Dual-side 계획을 갱신했다. [연구 계획](https://app.notion.com/p/4220ee8277e48153a020e374317777f8) · [실험 아카이브](https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489)

H100 Main100 QPS2/4에서 Native EPD는 Agg DP3보다 Decode Tail이 짧고 TTFT·Throughput이 불리했다. Main100 QPS2 NativeCommon 대비 Static10의 Equal-output Replay에서 P HTTP가 평균 305.329ms 줄었고 D Coordinator Wait는 971.956ms, TTFT는 933.063ms 늘었다. 고부하 Kernel 계측은 5% Gate를 통과하지 못했다. 별도 Counter 실험은 Static10 D Prepared Batch의 최대값 1과 NativeCommon의 최대값 4를 확인했으나 관측 간섭 Gate는 실패했다. 이 차이는 H200 통제 실험의 가설이며 이전 Clean N3의 Batch 관측값으로 사용하지 않는다. [H100 종합](mllm_epd_prefill_heavy_token_split_final_summary.md) · [Baseline](mllm_epd_main100_baseline_motivation_v1.md) · [Serving](mllm_epd_generic_token_split_online_serving.md) · [Critical path](mllm_epd_protected_token_critical_path_v1.md) · [Batch counter](mllm_epd_actual_batch_counter_validation_v1.md)

Dense AR에서 D Admission Cap을 바꾸자 Decode Tail이 개선되면서 TTFT와 Goodput이 악화됐다. 같은 Token Arm에서 출력 길이를 맞춘 Replay에도 대기 증가가 남았다. H200에서는 같은 D Admission을 공유하는 Native와 Split을 짝지으며, KV Credit 수명을 기록하고 원래 Unrestricted Native를 유지한다. [Dense AR 최종](https://github.com/jyssys/MLLM-EP/blob/449c3e1873c65206435a6bb9411a889a62e55be7/poc_flashvep/reports/dense_ar_ttft_slo_regression_final_summary.md) · [Native 대기 원인](https://github.com/jyssys/MLLM-EP/blob/449c3e1873c65206435a6bb9411a889a62e55be7/poc_flashvep/reports/dense_ar_native_relative_ttft_root_cause.md) · [통제 개입](https://github.com/jyssys/MLLM-EP/blob/449c3e1873c65206435a6bb9411a889a62e55be7/poc_flashvep/reports/dense_ar_pd_critical_path_controlled_interventions.md)

## 고정 계약

- Branch `flashvep/mllm-epd-h200-prefill-heavy-batch-matched`, 인수인계 HEAD `449c3e1873c65206435a6bb9411a889a62e55be7`, 과학 결과 H100 Commit `38e67f91f0099dc546d470c83ff03af638f7af16`. H200 소스와 실행 증거는 별도 Namespace에 기록한다.
- `Qwen/Qwen2.5-VL-7B-Instruct`, 모델·Processor Revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. BF16, BI0, 원래 Vision·Processor·MRoPE, Causal Attention, Full28, Greedy, Prefix Cache Off. P/D B8192·S1024·KV42GiB·Block128, E B114688·Renderer2·MM Cache0·Sync Scheduling·Custom Ops None.
- 동결 Main100 원본 SHA256 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`. DocVQA 13, Mantis 25, MP-DocVQA 25, MMLongBench-Doc 37. Calibration32는 사전 정합성에 사용한다. Natural EOS·Cap2048을 주 표로, 고정 5,269 Output Tokens Replay를 별도 표로 유지한다. 실제 H200 출력 길이와 종료 사유는 별도 기록한다. [환경 및 정합성](mllm_epd_h200_environment_and_correctness.md)
- GPU 0–7 사용 승인 범위에서 새 서버 UUID로 소유권을 검증한다. 핵심 Paired 비교는 GPU 4/5/6의 E/P/D에서 수행하며 실험 중 나머지 다섯 GPU에서 Burner를 유지한다. Burner 상주 여부와 배치를 실행별 기록한다. 실험용 GPU는 선점 전에 소유 Burner만 비우고, 완료 후 다른 프로세스가 없는 승인 GPU에 이 서버의 `train_lm.py`를 복원한다. 일정은 최대 약 8–10시간이며 결과 확인을 위해 기본 계약을 바꾸지 않는다.

## 실행 순서와 측정 기준

| 단계 | 실행 및 판정 |
|---|---|
| A. H200 Baseline | 기존 Native와 Token10 정합성 자료를 재사용한다. NativeStrong·NativeCommon의 Main100 온라인 QPS2에서 E→P의 원래 Embedding, NIXL P→D, 실제 D Decode, 도착·완료·Client TTFT/ITL을 확인한다. |
| B. D Batch1 | 같은 D FIFO Admission을 적용해 동시에 한 요청만 D에 보낸다. Token Suffix 상태와 Native Decode 대기를 별도 기록한다. E/P와 전체 Client는 open-loop 및 continuous scheduling을 유지한다. Unrestricted NativeStrong/Common도 보존한다. B1 변경 정합성을 별도 확인하고, 실제 scheduled/prepared Decode Request Count·Query Rows·CUDA Graph Padding·D Pending Queue를 측정한다. |
| C. Capacity 및 QPS | Main100 Natural EOS Native에서 QPS 0.5/1/2/3/4/5/6/8 중 용량 무릎을 N1으로 탐색하고 유망 구간을 독립 N3로 확인한다. E-only는 원래 이미지, P-only는 원래 E의 Embedding과 비동기 KV Drain, D-only는 실제 KV와 고정 출력 분포를 사용한다. D-only의 B1과 Unrestricted를 각각 측정한다. 실제 요청/s 및 Token/s가 비교 가능한지 Geometry·Workload Mix로 확인한다. |
| D. 계측 | Clean Before→Profile→Clean After에서 TTFT p50/p95·ITL p99·TPOT p95·Throughput의 간섭 및 Clean Spread를 각각 5% 이내로 확인한다. 실패 Trace는 DIAGNOSTIC이다. GPU Kernel Busy는 겹치는 Kernel Interval의 합집합으로 계산한다. NVML과 Host Epoch는 별도 단위로 기록한다. |
| E. P-heavy/D-idle | 예정 Arrival p20–p80을 기본 중앙 구간으로 고정하고 실제 Capture 교집합과 분모를 명시한다. P Busy ≥80%, 비교 가능한 P Service Proximity ≥85%, Embedding-ready 대기 또는 지속 Capacity Pressure, E 공급 Starvation 부재, D No-active/pending Idle 및 KV/Buffer 여유, Absolute Decode SLO ≥95%를 모두 확인한다. 실제 적격 Profile과 독립 Capacity·Queue 근거를 요구한다. |
| F–G. 비교 | E에서 조건을 확보하면 Static10-B1부터 시작한다. 각 후보의 N1/N8 정합성 이후 Static5/15/20, 마지막으로 기존 Idle10을 검토한다. 기존 Decode가 남아 있으면 D Suffix를 실행하지 않는다. 결과가 유망한 경우 동일 H200·Arrival·Decode Admission에서 독립 Paired N3를 실행하고 별도로 Unrestricted Native와 비교한다. |

P Queue는 E 실행 완료 이후 실제 P 가용성이 부족해 기다린 요청을 기준으로 본다. P HTTP, P 스케줄러, KV Credit, Embedding Transfer 대기를 따로 기록한다. 현재 또는 미래 Suffix 가용성은 단순 Kernel Gap에서 추론하지 않는다. P Busy와 D Idle의 겹침은 전체 중앙 구간과 P Busy 구간 두 분모로 산출하고, Queue Nonempty 교집합도 보고한다. 고부하 Capture가 좁으면 전체 중앙 구간의 값 대신 실제 ROI를 명시한다. A/B의 유한 Cohort Latency와 SLO는 모든 요청이 분모이며 Throughput은 예정 첫 Arrival부터 Drain 종료까지의 벽시계 시간으로 계산한다. [이전 병목 판정](mllm_epd_prefill_bottleneck_and_decode_idle.md) · [이전 Idle 종합](mllm_epd_encoder_decoder_idle_final_summary.md) · [Timeline 범위 감사](mllm_epd_idle_scope_and_required_figures_v2.md)

Candidate 성능 자료에 앞서 160ms를 Typical 검토 기준으로 동결한다. 이는 과거 C41R3의 1024-row 세 Context에서 Clean Before/After 중 큰 HTTP p95의 중앙값 160.078ms를 반올림한 진단 참조다. Long-context 2048-row의 최대 Clean HTTP p95 224.265ms를 반올림한 225ms를 민감도 기준으로 보고한다. P Busy·P Backlog와 동시 발생하는 D No-active/pending Window에서 160ms 이상 Window가 P Busy 시간의 10% 이상을 덮는지 우선 확인한다. D idle 기간 분포와 128–2048-row별 Coverage를 함께 공개하며, H200의 실제 Suffix 비용이 확보되면 그 값과도 비교한다. Offline Window Fit에는 다음 요청 도착을 알고 있었다는 가정을 명시하고 `OFFLINE_ORACLE`로 표시한다. 실제 Suffix 실행 시에는 Prefix-ready D Active/Pending과 KV 여유, D Wait·Suffix·첫 Output을 Request별로 기록한다. [이전 Suffix 비용](mllm_epd_imported_kv_suffix_quantum_cost_v1.md)

## 성공과 종료

동일 B1 Native 대비 TTFT p95가 5% 이상 감소하고 Paired CI가 개선 방향을 지지해야 한다. TTFT p50 비열등, ITL p99와 TPOT p95 비율의 Paired 95% CI 상한 ≤1.05, Absolute Decode Attainment ≥95%, Task Quality 비열등, Joint SLO와 Goodput 유지도 확인한다. 요청별 SLO는 TTFT ≤5000ms, 모든 Client ITL ≤34ms, TPOT ≤33ms/token이다. 표와 CI의 독립 표본은 서버 Start다. Natural과 Replay, Batch1과 실용 Unrestricted 비교, H100과 H200은 각각 따로 보고한다. E→P Descriptor 정책을 고정하며 Host HTTP 경계와 GPU-exclusive 연산 시간은 각 정의로 집계한다. [EC 정책 인과 검사](mllm_epd_native_ec_descriptor_causal_control_v1.md)

P-heavy/D-idle을 입증하지 못하면 광범위한 Split Ratio 검색을 중단한다. 적격 측정에서 Gate를 통과하지 못하면 `P_HEAVY_D_IDLE_NOT_CONFIRMED`, 핵심 계측·Capacity가 빠져 판정할 수 없으면 `P_HEAVY_D_IDLE_NOT_MEASURED`다. 이전 H100의 정합성 관측은 Same-geometry Prefix KV와 실제 Vision Batch Geometry 차이를 함께 고려한다. 이번 Correctness Gate에는 원래 Embedding·Token Order·Image Grid/MRoPE·Full28 KV·Continuation·Greedy Output·Buffer Lifetime이 포함된다. 전체 품질이 미확정이면 Research GO를 보류한다. [이전 정합성](mllm_epd_generic_token_split_correctness.md) · [계약 및 편차](mllm_epd_campaign_contract_and_deviations_v1.md)

적격 P-heavy에서 재현된 Online Gain과 Decode·품질 보존은 `GO_ONLINE_TOKEN_SPLIT`, B1에서만 유리하거나 강한 기준선·Heldout·품질 근거가 부족한 관측은 `CONDITIONAL_BATCH_MATCHED_BENEFIT`, 적격 P-heavy 및 실제 D Headroom에서 시험한 후보에 순이익이 없으면 `NO_GO_ON_TESTED_GRID`다. Gate·계측·정합성·품질·예산이 남으면 `HOLD`다. Figure는 원시 CSV/JSON에서 생성하며 미실행 조건은 표에 표시한다. 대표 Timeline은 중앙 ROI의 중점을 기준으로 정하고 Capture가 짧으면 실제 경계를 함께 기록한다.
