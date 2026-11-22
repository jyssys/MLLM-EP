# LongBench400: Figure와 재계산 절차

각 Figure의 PNG와 PDF는 같은 Caption과 Source Manifest를 사용한다.
실측, Isolated Kernel, Structural Accounting과 Offline Model을 구분한다.
초기 Discovery Figure도 보존한다. 이를 독립 반복이 끝난 Main Figure로 표시하지 않는다.

## 권장 읽기 순서

1. `dataset400_*`: 원문과 Natural Output을 사용한 Dataset 감사.
2. `motivation400_confirmed_*`: 개선 전 Installed Agg/Native의 실제 Continuous Sweep.
3. `axis400_heldout_*`: C0 Actual Batch1/1/1의 64-cell Control.
4. `controlled_output_and_load_counterfactuals`: Output 또는 부하만 바꾼 반증.
5. `hybrid400_qualified_long_operator_cost`: Observer Gate를 통과한 고립 Operator 비용.
6. `final_c3_*`: Discovery에서 동결한 후보의 Natural400 Continuous 비교.
7. `controlled_C3_*`: 작은 Controlled-output Bridge. Natural400과 별도다.
8. `hybrid400_bounded_offline_oracle`: 제한된 Model 탐색. Online Speedup이 아니다.

`adaptive_repair_confirmed_*`는 Main 증상 이후의 READ Repair를 표시한다.
`registration_repair_confirmed_*`는 별도 Source의 Host Request 등록 Intervention을 표시한다.
`registration_repair_q28_confirmed_*`는 같은 수정의 별도 QPS2.8 Extension을 표시한다.
`registration_controlled_C3_*`는 원래 세 Controlled Bridge 조건에 같은 수정을 적용한 후속 검사다.
`eager_prefill_native_decode_*`는 모든 Prefill과 Decode의 Kernel 경로를 분리한 마지막 제한적 검사다.
이 검사는 I3/Output16/Cap1이다. Natural400의 성능 또는 Dynamic 결과가 아니다.
두 후속 검사를 원래 동결한 Main의 결과로 교체하지 않는다.

개별 Figure의 Caption이 위 접두사 설명보다 구체적인 실행 범위를 지정한다.
최종 Figure가 아직 없으면 해당 검사를 완료했다고 주장하지 않는다.

## 모든 생성 Figure

| Stem | Evidence | PNG | PDF |
|---|---|---|---|
| adaptive_repair_confirmed_goodput | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_goodput.pdf) |
| adaptive_repair_confirmed_itl | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_itl.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_itl.pdf) |
| adaptive_repair_confirmed_tpot | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_tpot.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_tpot.pdf) |
| adaptive_repair_confirmed_ttft | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/adaptive_repair_confirmed_ttft.pdf) |
| axis400_both_axes_decode_safe_winner_map | TRACE_ANALYSIS_OF_MEASURED_ONLINE_C0 | [PNG](dense_ar_longbench400_2d_20261121/figures/axis400_both_axes_decode_safe_winner_map.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/axis400_both_axes_decode_safe_winner_map.pdf) |
| axis400_heldout_decode_itl_upper | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_decode_itl_upper.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_decode_itl_upper.pdf) |
| axis400_heldout_decode_tpot_upper | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_decode_tpot_upper.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_decode_tpot_upper.pdf) |
| axis400_heldout_goodput_ratio | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_goodput_ratio.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_goodput_ratio.pdf) |
| axis400_heldout_ttft_ratio | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_ttft_ratio.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/axis400_heldout_ttft_ratio.pdf) |
| continuation_chunk_residency_host_audit | TRACE_ANALYSIS | [PNG](dense_ar_longbench400_2d_20261121/figures/continuation_chunk_residency_host_audit.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/continuation_chunk_residency_host_audit.pdf) |
| controlled_C3_I3_bridge_goodput | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_goodput.pdf) |
| controlled_C3_I3_bridge_itl | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_itl.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_itl.pdf) |
| controlled_C3_I3_bridge_tpot | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_tpot.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_tpot.pdf) |
| controlled_C3_I3_bridge_ttft | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_bridge_ttft.pdf) |
| controlled_C3_I3_hybrid_token_runtime_bridge | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_hybrid_token_runtime_bridge.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_C3_I3_hybrid_token_runtime_bridge.pdf) |
| controlled_output_and_load_counterfactuals | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/controlled_output_and_load_counterfactuals.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/controlled_output_and_load_counterfactuals.pdf) |
| credit_fixed_pool_sensitivity | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/credit_fixed_pool_sensitivity.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/credit_fixed_pool_sensitivity.pdf) |
| dataset400_balance_audit | MEASURED_NATURAL_OUTPUT_REFERENCE | [PNG](dense_ar_longbench400_2d_20261121/figures/dataset400_balance_audit.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/dataset400_balance_audit.pdf) |
| dataset400_input_output_scatter | MEASURED_NATURAL_OUTPUT_REFERENCE | [PNG](dense_ar_longbench400_2d_20261121/figures/dataset400_input_output_scatter.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/dataset400_input_output_scatter.pdf) |
| dataset400_length_distribution | MEASURED_NATURAL_OUTPUT_REFERENCE | [PNG](dense_ar_longbench400_2d_20261121/figures/dataset400_length_distribution.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/dataset400_length_distribution.pdf) |
| eager_prefill_native_decode_goodput | MEASURED_ONLINE_ADAPTIVE_COMPILER_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_goodput.pdf) |
| eager_prefill_native_decode_noninferiority | MEASURED_ONLINE_PAIRED_DECODE_GATE | [PNG](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_noninferiority.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_noninferiority.pdf) |
| eager_prefill_native_decode_ttft | MEASURED_ONLINE_ADAPTIVE_COMPILER_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/eager_prefill_native_decode_ttft.pdf) |
| exploratory_minimal_demand_rule_comparison | OFFLINE_MODEL_EXPLORATORY_COMPARISON | [PNG](dense_ar_longbench400_2d_20261121/figures/exploratory_minimal_demand_rule_comparison.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/exploratory_minimal_demand_rule_comparison.pdf) |
| final_c3_decode_noninferiority_common_policy | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_decode_noninferiority_common_policy.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_decode_noninferiority_common_policy.pdf) |
| final_c3_decode_noninferiority | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_decode_noninferiority.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_decode_noninferiority.pdf) |
| final_c3_main400_itl_p99 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_itl_p99.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_itl_p99.pdf) |
| final_c3_main400_joint_slo | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_joint_slo.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_joint_slo.pdf) |
| final_c3_main400_output_tokens | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_output_tokens.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_output_tokens.pdf) |
| final_c3_main400_slo_goodput | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_slo_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_slo_goodput.pdf) |
| final_c3_main400_tpot_p95 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_tpot_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_tpot_p95.pdf) |
| final_c3_main400_ttft_p95 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_ttft_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/final_c3_main400_ttft_p95.pdf) |
| hybrid400_bounded_offline_oracle | OFFLINE_ORACLE | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid400_bounded_offline_oracle.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid400_bounded_offline_oracle.pdf) |
| hybrid400_offline_replay_validation | OFFLINE_REPLAY | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid400_offline_replay_validation.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid400_offline_replay_validation.pdf) |
| hybrid400_qualified_long_operator_cost | ISOLATED_KERNEL_MEASUREMENT | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid400_qualified_long_operator_cost.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid400_qualified_long_operator_cost.pdf) |
| hybrid400_structural_DAG | STRUCTURAL_ACCOUNTING | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid400_structural_DAG.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid400_structural_DAG.pdf) |
| hybrid400_structural_attention_distribution | STRUCTURAL_ACCOUNTING | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid400_structural_attention_distribution.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid400_structural_attention_distribution.pdf) |
| hybrid_iso8_confirmed_decode_gate | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_decode_gate.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_decode_gate.pdf) |
| hybrid_iso8_confirmed_goodput | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_goodput.pdf) |
| hybrid_iso8_confirmed_ttft | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/hybrid_iso8_confirmed_ttft.pdf) |
| motivation400_confirmed_itl_p99 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_itl_p99.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_itl_p99.pdf) |
| motivation400_confirmed_output_tokens | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_output_tokens.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_output_tokens.pdf) |
| motivation400_confirmed_slo_goodput | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_slo_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_slo_goodput.pdf) |
| motivation400_confirmed_tpot_p95 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_tpot_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_tpot_p95.pdf) |
| motivation400_confirmed_ttft_p95 | MEASURED_ONLINE | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_ttft_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_confirmed_ttft_p95.pdf) |
| motivation400_itl_p99 | MEASURED_ONLINE_DISCOVERY | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_itl_p99.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_itl_p99.pdf) |
| motivation400_output_tokens | MEASURED_ONLINE_DISCOVERY | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_output_tokens.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_output_tokens.pdf) |
| motivation400_slo_goodput | MEASURED_ONLINE_DISCOVERY | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_slo_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_slo_goodput.pdf) |
| motivation400_tpot_p95 | MEASURED_ONLINE_DISCOVERY | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_tpot_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_tpot_p95.pdf) |
| motivation400_ttft_p95 | MEASURED_ONLINE_DISCOVERY | [PNG](dense_ar_longbench400_2d_20261121/figures/motivation400_ttft_p95.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/motivation400_ttft_p95.pdf) |
| qualified_projection_shape_efficiency | ISOLATED_KERNEL_MEASUREMENT | [PNG](dense_ar_longbench400_2d_20261121/figures/qualified_projection_shape_efficiency.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/qualified_projection_shape_efficiency.pdf) |
| recovered_native_observed_kernel_copy_timeline | MEASURED_KERNEL_ACTIVITY | [PNG](dense_ar_longbench400_2d_20261121/figures/recovered_native_observed_kernel_copy_timeline.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/recovered_native_observed_kernel_copy_timeline.pdf) |
| recovered_native_observed_kernel_gap_distribution | MEASURED_KERNEL_ACTIVITY | [PNG](dense_ar_longbench400_2d_20261121/figures/recovered_native_observed_kernel_gap_distribution.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/recovered_native_observed_kernel_gap_distribution.pdf) |
| recovered_native_shared_ROI_activity_and_queues | MEASURED_KERNEL_ACTIVITY_WHERE_OBSERVER_PASSED | [PNG](dense_ar_longbench400_2d_20261121/figures/recovered_native_shared_ROI_activity_and_queues.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/recovered_native_shared_ROI_activity_and_queues.pdf) |
| registration_controlled_C3_I3_bridge_goodput | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_goodput.pdf) |
| registration_controlled_C3_I3_bridge_itl | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_itl.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_itl.pdf) |
| registration_controlled_C3_I3_bridge_tpot | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_tpot.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_tpot.pdf) |
| registration_controlled_C3_I3_bridge_ttft | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_bridge_ttft.pdf) |
| registration_controlled_C3_I3_hybrid_token_runtime_bridge | MEASURED_ONLINE_CONTROLLED_OUTPUT | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_hybrid_token_runtime_bridge.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_controlled_C3_I3_hybrid_token_runtime_bridge.pdf) |
| registration_repair_confirmed_decode_gates | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_decode_gates.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_decode_gates.pdf) |
| registration_repair_confirmed_goodput | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_goodput.pdf) |
| registration_repair_confirmed_itl | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_itl.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_itl.pdf) |
| registration_repair_confirmed_tpot | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_tpot.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_tpot.pdf) |
| registration_repair_confirmed_ttft | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_confirmed_ttft.pdf) |
| registration_repair_q28_confirmed_decode_gates | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_decode_gates.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_decode_gates.pdf) |
| registration_repair_q28_confirmed_goodput | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_goodput.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_goodput.pdf) |
| registration_repair_q28_confirmed_itl | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_itl.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_itl.pdf) |
| registration_repair_q28_confirmed_tpot | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_tpot.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_tpot.pdf) |
| registration_repair_q28_confirmed_ttft | MEASURED_ONLINE_ADAPTIVE_REPAIR | [PNG](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_ttft.png) | [PDF](dense_ar_longbench400_2d_20261121/figures/registration_repair_q28_confirmed_ttft.pdf) |

## 데이터가 없어 생성하지 않은 Figure

빈 Plot이나 예측값을 실측 Figure로 채우지 않았다. 아래 항목은 과학적 중단 범위를 기록한다.

| 항목 | 상태 | 이유 |
|---|---|---|
| prepared_state_layer_token_safe_fit_rate | NOT_MEASURED | 실제 준비된 KV/Hidden과 미래 Decode Deadline을 결합한 Safe-fit Gate를 검증하지 못했다. |
| online_dynamic_2d_decision_distribution | NOT_RUN | 안전 Window가 미검증이다. 위험한 Online Dynamic 삽입과 의사결정 Histogram은 만들지 않았다. |
| qualified_C3_exclusive_operator_share | NOT_MEASURED | 적격 Operator Capture는 C0다. Mode-specific Hook으로 C3 Compiled 경로를 적격 계측하지 못했다. |
| qualified_SM_HBM_tensor_core_counters | NOT_MEASURED | Clean Observer Gate를 통과한 해당 Hardware Counter가 없다. Kernel 이름이나 FLOPs로 대신 채우지 않았다. |

Caption, File SHA256과 Source Manifest SHA256은 다음 Ledger에 있다.
[Figure Ledger](dense_ar_longbench400_2d_20261121/analysis/figure_catalog_v1.json).

## CPU 재계산

아래 명령은 이 서버의 기존 Environment와 Raw Evidence를 사용한다.
별도 머신에서 실행하려면 Runtime Path와 원본 Dataset Cache를 먼저 재현해야 한다.
Canonical Upstream Wheel 또는 Container의 완전한 재현을 주장하지 않는다.
설치된 vLLM의 Python Source 2234개는 `references/installed_vllm_python_sources_v1.zip`에 보존했다.
별도 `installed_vllm_source_licenses_metadata_v1.zip`은 원래 License, METADATA, WHEEL과 RECORD를 보존한다.
이 Metadata Snapshot이 모든 Compiled GPU Binary를 재현하는 것은 아니다.
Task Metric의 별도 Dependencies는 `references/quality_dependency_audit_v1.json`에 Version과 File Hash로 기록했다.
Fuzzywuzzy0.18.0, Jieba0.42.1, Rouge1.0.1과 Six1.17.0을 사용했다.
Fuzzywuzzy의 SequenceMatcher는 Python `difflib`다. Levenshtein 경로를 조용히 추가하지 않는다.
Quality Dependencies의 로컬 Tree를 완전한 독립 Runtime으로 재배포한다고 주장하지 않는다.
기존 사용자 Environment를 Snapshot으로 덮어쓰지 않는다.
작업 디렉터리는 `/home/esjung/MLLM-EP-github`다.
Python은 `/home/esjung/.venvs/looped-pd-abc/bin/python`이다.
모든 Module 앞에는 `poc_flashvep.dense_ar_longbench400_pd.`를 붙인다.

| Module | 목적 | GPU 실행 |
|---|---|---|
| `cpu_final_tests` | Syntax, Row Ownership, Policy Restoration와 Work Conservation | 없음 |
| `independent_metric_audit` | Client Event에서 Metric을 독립 재계산 | 없음 |
| `budget_audit` | 모든 완료 Worker의 B/S와 Actual Query 검사 | 없음 |
| `quality_dependency_audit` | 공식 Task Metric의 Dependency Version과 Hash | 없음 |
| `confirmation_analysis` | C0 Paired Start CI와 Gate | 없음 |
| `axis_joint_safety_audit` | 양축 모두 Native Decode Gate를 통과한 C0 부분집합 | 없음 |
| `iso8_analysis` | Exact Iso-work Hybrid와 Decode Gate | 없음 |
| `iso8_figures` | 8개 Iso-work Hybrid Cell의 전체 Figure | 없음 |
| `pipeline_load_analysis` | 새 Load Regime의 Queue와 Drain | 없음 |
| `controlled_bridge_analysis` | C3 Controlled Output과 Cap Intervention | 없음 |
| `controlled_bridge_report` | 전체 Bridge Metric와 같은 ID의 Host 경계 | 없음 |
| `final_confirmation_analysis` | Final C3 Metric과 두 Native 기준 CI | 없음 |
| `hybrid_read_repair_analysis` | 원래 Main과 분리한 Adaptive Repair 검사 | 없음 |
| `repair_report_figures` | READ Repair Table과 Figure | 없음 |
| `registration_deferral_analysis` | Host 등록 Intervention, Paired Gate와 Figure | 없음 |
| `registration_deferral_analysis --plan request_registration_q28_extension_frozen_v1.json --suffix _q28` | 같은 수정의 별도 QPS2.8 분석 | 없음 |
| `registration_controlled_bridge analyze` | 같은 수정의 별도 Controlled C3 Bridge | 없음 |
| `eager_prefill_decode_campaign analyze` | 최종 Kernel 정책의 같은 조건 Paired Metric와 Gate | 없음 |
| `final_compiler_artifacts --figures` | Teacher 실제 Scope와 최종 Kernel 정책 Figure | 없음 |
| `final_report_tables` | 모든 Final Start의 Table | 없음 |
| `final_service_decomposition` | 같은 ID와 Arrival의 Mean TTFT 경계 차이 | 없음 |
| `final_protocol_integrity_audit` | 동결 시점, 실제 GPU UUID와 Worker 계약 | 없음 |
| `figures dataset` | Dataset Figure | 없음 |
| `confirmed_figures` | Motivation과 C0 Matrix | 없음 |
| `mechanism_figures` | DAG, Attention, Kernel Cost와 Offline Model | 없음 |
| `projection_shape_audit` | 적격 Capture의 실제 GEMM Shape, Launch와 GPU Union | 없음 |
| `supplementary_figures` | Counterfactual, 공통 ROI와 Knob | 없음 |
| `controlled_bridge_figures` | 작은 C3 Bridge Figure | 없음 |
| `final_serving_figures` | Main C3와 Decode CI Figure | 없음 |
| `figure_catalog` | Figure Hash와 이 목록 | 없음 |

예시는 다음과 같다. 이 명령은 분석과 Figure 파일을 다시 생성한다.
Raw Evidence, 동결한 Dataset과 실행 Config는 변경하지 않는다.

```bash
cd /home/esjung/MLLM-EP-github
taskset -c 160-191 /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_longbench400_pd.cpu_final_tests
taskset -c 160-191 /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_longbench400_pd.final_confirmation_analysis
taskset -c 160-191 /home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dense_ar_longbench400_pd.final_serving_figures
```

## 새 GPU 실행

`run <name>`은 완료한 Raw 결과가 있으면 그 결과를 재사용할 수 있다.
따라서 같은 명령을 다시 실행한 것을 새 Independent Start로 세지 않는다.
진짜 새 Start는 새 이름, 독립 Server 초기화와 승인된 GPU Lease가 필요하다.
Dataset/Source SHA, Arrival Offset, Seed, GPU UUID와 실제 Batch를 함께 보존해야 한다.
기존 Raw Folder나 실패 결과를 덮어쓰지 않는다.
Closeout한 Namespace의 `STOP_DISPATCH.json`을 삭제해 재실행하지 않는다.
새 GPU 캠페인은 별도 Namespace와 새 Resource 계약을 먼저 만들어야 한다.
Final 동결 선택은 `configs/final_continuous_static_choices_frozen_v1.json`에 있다.
실행 순서와 Pair 배정은 `configs/final_continuous_execution_plan_v1.json`에 있다.
현재 계획을 사후 수정해 재실행한 결과를 같은 Held-out Confirmation으로 표시하지 않는다.

## Archive와 Publication

GPU 작업이 모두 종료된 뒤에만 Evidence Archive를 생성한다.
Public Archive는 Text Evidence와 실패 Log를 보존한다.
큰 Tensor는 로컬 무손실 Archive에 보존한다. 각 Tensor의 SHA256도 기록한다.
Byte-identical Raw Source ZIP은 Public Capsule 한 개와 경로별 동일 SHA256으로 복원한다.
원래 Raw ZIP은 로컬 무손실 Archive에도 그대로 보존한다.
전체 Member Ledger가 단일 Git File 제한을 넘으면 검증된 `.json.gz`와 작은 `.index.json`을 공개한다.
먼저 Index의 Compressed SHA256을 확인한다. Gzip을 해제한 뒤 Expanded SHA256도 확인한다.
이 절차는 원래 JSON Ledger를 Byte 단위로 복원한다. Member Hash나 실패 기록을 생략하지 않는다.
Archive Manifest의 순서대로 Public Part를 연결한다. 전체 SHA256을 검사한 뒤 압축을 해제한다.
Archive 경로를 검사한 뒤 Repository Root에 복원해야 한다.
Remote SHA 검증과 Burner 복원 Receipt는 별도 Operational Closeout 근거다.
이 Receipt 자체를 GPU Safety 또는 Serving Speedup의 근거로 사용하지 않는다.
