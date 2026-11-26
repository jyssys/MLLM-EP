# H200 Dynamic Token Split과 Modality-aware 최종 요약

**Generic은 HOLD다. Modality-aware는 MODALITY_MECHANISM_ONLY다.** 실제 D suffix를 실행한 두 N8 진단에서 Native와 함께 Decode Batch 8을 형성했고 Full28 상태 기준은 8/8이었다. 두 실행 모두 전체 Greedy gate가 7/8이어서 active C3를 시작하지 않았다. QPS5 online shadow의 D suffix는 모두 0이었다. B8과 Free는 모두 5% CPC 기준에 실패해 P_HEAVY_D_IDLE_NOT_MEASURED를 유지한다.

| 질문 | 확인 범위 |
| --- | --- |
| Q1. Static10의 정확한 Batch1 원인 | **소스 및 기존 Counter 확인.** suffix guard가 active decode 전체의 종료를 요구한다. 첫 token에서 owner를 반납해도 EOS까지 active인 이전 요청 때문에 다음 suffix를 기다린다. 기존 Static10 Free/B4 actual max=1, single=100% |
| Q2. Late-bound의 D Coordinator Wait | 초기 mixed N8 상태 계측의 승인 9건은 P-ready→D-submit 0.059–0.175 ms였다. 기존 Idle10-B8 clean의 suffix 33건은 p95 3.438 s였다. 서로 다른 cohort에서 측정했다. Stock boundary serial의 5개 D suffix와 same-D P continuation은 strict check를 통과했다. Active late-bound C3 wait는 NOT_RUN |
| Q3. 실제 Continuous Batching | Natural-hold 상태 검증에서 실제 D suffix 뒤 lead가 Native follower와 41/43개 step에서 함께 decode했다. 두 조건 모두 이 shared forward의 actual batch 최대는 8이었다. 두 fixture의 exact Greedy는 각 7/8에 그쳤다. Native 수준의 C3 효율은 NOT_MEASURED |
| Q4. Idle Budget과 실제 동적 ratio | 0/5/10/20/30% isolated 비용과 직렬 stock cut의 실제 7.22–32.03% suffix를 확인했다. A Natural shadow는 6/100건에서 D reservation 가능 snapshot을 봤다. B와 C Hard는 Natural 및 Replay 각각 0/100건이었다. 실제 온라인 Offload는 0 |
| Q5. Generic 대 Native B8/Free | Active C3 Natural/Replay는 NOT_RUN. B shadow TTFT p95는 Natural 13.383 s, Equal-output Replay 13.108 s였다. 두 실행 모두 P 전체 prefill을 수행했다 |
| Q6. Decode 보호 및 CI | Actual suffix가 없는 clean shadow의 Absolute Decode SLO는 96–98%였다. Active split의 ITL/TPOT 보호와 independent paired 95% CI는 NOT_RUN |
| Q7. 추가 modality 메커니즘 | 현재 whole-image CUDA IPC에서 intra-image cut은 원본 crossing 텐서를 E에서 두 worker로 보내 clone한다. On-demand와 inline의 각 18/18 matched pair에서 추가 복제 및 E 방향송신을 측정했다 |
| Q8. Intra-image 대 Boundary | Inline 18개 one-token pair에서 NIXL bytes 동일. Intra가 D 구간 E NVLink TX +3.874–53.253 MiB(중앙값 +32.812)와 D clone 하나를 추가했다. D one-token HTTP 중앙값 +24.970 ms, 17/18 증가. On-demand에서도 같은 메커니즘을 확인했다. 온라인 peak resident 및 causal TTFT은 NOT_MEASURED |
| Q9. Encoding Feature의 heldout 값 | Inline Generic MAE/p95 underprediction 26.0690/90.5902 ms, Full 26.8655/87.4991 ms, 160 ms false-safe 1/16 대 1/17. 두 transport에서 Full이 p95 과소예측을 줄였고 MAE는 높아졌다. Soft feature 채택 규칙 미달 |
| Q10. Feature/후보/work 교란 분리 | H-M1은 동일 요청·동일 prefix-KV block·1-token 차이 및 NIXL bytes를 통제했다. Shadow B/Hard에는 동일 stock P 후보 조건을 적용했다. Active feature-off, block-alignment, fake-boundary의 온라인 효과는 NOT_RUN |
| Q11. Hard 대 Soft | Stock Hard는 Natural과 Replay에서 각각 11개 image-interior callback을 거절했다. Soft는 inline heldout 규칙 미충족으로 shadow와 active 모두 NOT_RUN |
| Q12. Method B 대 C 온라인 | Shadow 같은 요청 200쌍에서 B/Hard는 모두 실제 예약과 D suffix가 0이었다. Active TTFT, ITL, TPOT, goodput 및 품질의 incremental effect는 NOT_RUN |
| Q13. Workload 별 이득 | Main100의 5–30% true-edge 존재는 single 0/13, few 13/45, many 5/5, long-document 37/37. 10% logical crossing tensor 중앙값은 single 32.94 MiB, long-document 4.21 MiB. Workload별 Serving 이득은 NOT_MEASURED |
| Q14. 최종 판정 | Generic HOLD · Modality-aware MODALITY_MECHANISM_ONLY |
| Q15. 실제 논문 범위 | Static10의 drain guard, 동일 D 경로로 검증한 P-state continuation, 실제 D suffix 후 Native follower와의 batch8 co-execution, 현재 whole-image transfer의 image-span crossing 비용을 관측했다. 두 mixed 조건은 전체 Greedy gate에 미달했다. Generic dynamic과 readiness-aware scheduling의 선행 범위를 정리했으며 C3 순이익은 미검증이다 |

## 실행 환경

H200 GPU4/5/6을 E/P/D로 사용했다. 서버의 GPU 8개는 모두 143,771 MiB이며, 각 GPU 쌍의 topology는 NV18이다. 실험 그룹은 NUMA1에 있다. NVIDIA Driver 580.126.16의 보고 CUDA 상한은 13.0이다. Runtime은 Python 3.14.6, vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129, Transformers 5.16.1, NIXL 1.4.1이다. 실험한 run의 바깥 GPU 5개에는 기존 Burner가 있었으며, 이번 계측으로 Burner 간섭 자체를 분리하지 않았다. Main100과 Calibration32의 원본 및 relocated manifest hash를 각각 다시 확인했다. [GPU UUID·실제 명령·환경·hash 증거](../mllm_epd_h200_modality_aware_dynamic_split/results/campaign_execution.json)

마지막 GPU 실험은 17:30:15 KST에 종료됐다. 17:31:09에 소유권을 확인한 기존 train_lm.py Burner가 GPU0–7 전체에서 실행 중이었다. 그 표본의 GPU utilization은 77–90%였다. 최종 대상 CPU regression은 49/49 통과했다. [리소스 복원 및 실행 증거](../mllm_epd_h200_modality_aware_dynamic_split/results/campaign_execution.json)

## 실행하지 않은 비교

Active A/B/C의 timed C3, 그에 대한 독립 N3 이상과 QPS4/6, 새 요청으로 구성한 Heldout, 전체 Task Quality의 비열등 검증은 NOT_RUN이다. 기존 pinned scorer의 Native/Idle10 N1 진단과 MMLongBench-Doc RAW 진단은 별도 보존했다. Soft feature는 inline Calibration에서 채택 기준을 통과하지 못했다. [온라인 비교·정합성·품질 범위](mllm_epd_h200_generic_vs_modality_aware_online_serving.md)

## 증거 상세

- [동결 실험계약과 Figure 인덱스](mllm_epd_h200_modality_aware_dynamic_split_poc_spec.md)
- [실제 admission guard와 Counter](mllm_epd_h200_dynamic_split_admission_root_cause.md)
- [H200 suffix cost](mllm_epd_h200_suffix_cost_model.md)
- [Method A와 continuation](mllm_epd_h200_late_bound_batch_preserving_split.md)
- [Method B 및 idle budget](mllm_epd_h200_generic_dynamic_token_split.md)
- [Modality 가설과 선행 연구](mllm_epd_h200_why_modality_aware_motivation.md)
- [Image Span 및 데이터 이동](mllm_epd_h200_image_span_transfer_cost_ablation.md)
- [Encoding Feature와 비용 예측](mllm_epd_h200_encoding_aware_cost_prediction.md)
- [Method C Hard/Soft](mllm_epd_h200_modality_aware_dynamic_token_split.md)
- [온라인 Generic/Modality 비교](mllm_epd_h200_generic_vs_modality_aware_online_serving.md)

## Figure와 원자료

| Figure | 실제 출력 및 범위 |
| --- | --- |
| 1 | [이번 B8/Free CPC 진단](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig01_current_motivation.pdf), [이전 Free reference](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig01_historical_prefill_heavy.pdf). 모두 DIAGNOSTIC |
| 2 | [이전 Static admission](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig02_historical_static_admission.pdf), [이번 forced N8 준비 Batch](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig02_12_mixed_correctness_batch_and_outcomes.pdf), [D suffix 이후 실제 Native co-batching](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig02_staged_natural_lead_decode.pdf) |
| 3 | [On-demand H200 isolated suffix 비용](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig03_h200_suffix_cost.pdf) · [Inline suffix 비용](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig03_h200_suffix_cost.pdf) |
| 4 | [On-demand intra-image 대 boundary matched work](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig04_image_cut_work_matched.pdf) · [Inline matched work](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig04_image_cut_work_matched.pdf) |
| 5 | [On-demand Generic 대 modality의 heldout 비용](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig05_generic_modality_cost_prediction.pdf) · [Inline heldout 비용](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig05_generic_modality_cost_prediction.pdf) |
| 6 | [원본 image 구조별 길이](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig06_frozen_workload_structure.pdf), [입력 길이와 구조별 inline transfer 차이](../mllm_epd_h200_modality_aware_dynamic_split/results/inline_calibration/figures/fig06_image_structure_transfer.pdf) |
| 7–8 | Active C3 ratio 및 Hard/Soft 선택 효과 NOT_RUN |
| 9 | [현재 Native, 기존 Idle10과 zero-offload shadow의 clean TTFT CDF](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig09_clean_ttft_cdf.pdf). Active B/C 곡선 NOT_RUN |
| 10–11 | Active C3 feature ablation 및 두 arm에서 실제 suffix가 실행된 matched timeline NOT_RUN |
| 12 | [Mixed 상태 route](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig02_12_mixed_correctness_batch_and_outcomes.pdf), [Natural serial probe](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig12_natural_boundary_correctness.pdf), [Shadow decision breakdown](../mllm_epd_h200_modality_aware_dynamic_split/results/figures/fig12_shadow_decision_eligibility.pdf) |

[Raw-derived online cohorts](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_cohorts.csv) · [Per-request online intervals](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_requests.csv) · [정합성 요약](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_correctness.json) · [동일 E 제어 원자료](../mllm_epd_h200_modality_aware_dynamic_split/results/dynamic_serial_E_geometry.json) · [Figure 재현 계약](mllm_epd_h200_modality_aware_dynamic_split_poc_spec.md#결과-재현)
