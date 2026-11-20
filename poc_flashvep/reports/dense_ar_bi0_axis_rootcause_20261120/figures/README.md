# BI0 Axis 연구 Figure

PNG와 PDF는 같은 데이터를 사용한다.
누락된 QPS는 실제 측정점으로 보간하지 않는다.
CI는 명시한 독립 Server Start에서만 계산한다.
Request 단위의 Task-quality CI와 성능 CI는 서로 다르다.

## Primary Controlled 결과

Full200, QPS2.35, BI0와 실제 Batch1/1/1을 사용한다.
각 Cut과 Axis에는 독립 Server Start3회가 있다.
Source Credit4, Hidden Slots4, KV128과 B8192/S1024를 유지한다.
Natural EOS Cap은2048이다. Finite-cohort 결과다.

| Figure | 파일 | 증거 |
|---|---|---|
| Cut별 TTFT·SLO-goodput 비율 | [PNG](full200_highload_paired_confirmation.png), [PDF](full200_highload_paired_confirmation.pdf) | `MEASURED_ONLINE` |
| Native-relative ITL·TPOT5% 보호 | [PNG](native_relative_decode_noninferiority.png), [PDF](native_relative_decode_noninferiority.pdf) | `MEASURED_ONLINE` |

## Queue와 실행 경로

| Figure | 파일 | 증거와 제한 |
|---|---|---|
| P/D Queue의 정확한 체인 분해 | [PNG](exact_source_destination_queue_chains.png), [PDF](exact_source_destination_queue_chains.pdf) | `TRACE_ANALYSIS`. 서로 다른 두 Family다. GPU-exclusive 비용이 아니다. |
| Confirmation의 P/D Host Demand | [PNG](confirmed_P_D_host_service_demand.png), [PDF](confirmed_P_D_host_service_demand.pdf) | `TRACE_ANALYSIS`. Cell별3회다. Kernel Time이 아니다. |
| Source Credit Intervention | [PNG](source_credit_intervention.png), [PDF](source_credit_intervention.pdf) | `MEASURED_ONLINE`. Cell별1회다. Hidden Pool32는 Primary와 다르다. |
| KV16/128 Intervention | [PNG](KV_block_one_factor_sensitivity.png), [PDF](KV_block_one_factor_sensitivity.pdf) | `MEASURED_ONLINE`. Cell별1회다. |
| Serial/Online Host Service | [PNG](matched_serial_online_service.png), [PDF](matched_serial_online_service.pdf) | `DIAGNOSTIC`. Fixed-output Host 합계다. TTFT가 아니다. |
| 적격 P Kernel Breakdown | [PNG](qualified_source_kernel_breakdown.png), [PDF](qualified_source_kernel_breakdown.pdf) | Observer·Correlation Gate를 통과한 Profile만 표시한다. |
| D Suffix Kernel 측정 상태 | [PNG](qualified_destination_suffix_kernel_breakdown.png), [PDF](qualified_destination_suffix_kernel_breakdown.pdf) | `NOT_MEASURED` Primary. 적격 Profile이 없다. |
| 실제 Input Bucket별 TTFT 비율 | [PNG](input_bucket_axis_ratio.png), [PDF](input_bucket_axis_ratio.pdf) | `DIAGNOSTIC`. N≤8, Cell별1회다. |
| D Attention Pair 배치 | [PNG](structural_D_attention_placement.png), [PDF](structural_D_attention_placement.pdf) | `STRUCTURAL_ACCOUNTING`. GPU 시간이 아니다. |

## Historical Bridge

| Figure | 파일 | 증거와 제한 |
|---|---|---|
| Historical/Current TTFT 비율 | [PNG](historical_current_axis_ttft_ratio.png), [PDF](historical_current_axis_ttft_ratio.pdf) | 각 Family를 별도로 표시한다. Runtime 단일 변경 실험이 아니다. |
| Historical Ready→D Guard 겹침 | [PNG](historical_ready_D_guard_overlap.png), [PDF](historical_ready_D_guard_overlap.pdf) | `TRACE_ANALYSIS`. Guard 제거 효과가 아니다. |
| Historical/Current P/D Host Demand | [PNG](historical_current_P_D_host_balance.png), [PDF](historical_current_P_D_host_balance.pdf) | `TRACE_ANALYSIS`. GPU 사용률 Map이 아니다. |
| Coalescing Host Flag Factorial | [PNG](BI1_exact_byte_coalescing_axis_factorial.png), [PDF](BI1_exact_byte_coalescing_axis_factorial.pdf) | `DIAGNOSTIC`. 실제 Descriptor 감소는 없었다. 최초1회와 추가 Seed2회를 포함한다. |

## QPS Discovery와 Continuous Bridge

N50 QPS Discovery의27개 측정 Cell은 모두 사전 Task-quality CI Gate를 통과하지 못했다.
따라서 다음 QPS Curve는 적격 Primary 성능이 아니다.
빈 Cell은 `NOT_RUN`이다. 각 측정점은 독립 Start1회다.

| Figure | PNG | PDF |
|---|---|---|
| Cut별 TTFT p95 | [PNG](pilot50_cut_qps_ttft.png) | [PDF](pilot50_cut_qps_ttft.pdf) |
| Cut별 ITL p99 | [PNG](pilot50_cut_qps_itl.png) | [PDF](pilot50_cut_qps_itl.pdf) |
| Cut별 TPOT p95 | [PNG](pilot50_cut_qps_tpot.png) | [PDF](pilot50_cut_qps_tpot.pdf) |
| Cut별 Raw Output tokens/s | [PNG](pilot50_cut_qps_throughput.png) | [PDF](pilot50_cut_qps_throughput.pdf) |
| Cut별 SLO-goodput | [PNG](pilot50_cut_qps_goodput.png) | [PDF](pilot50_cut_qps_goodput.pdf) |
| Cut×QPS 비율 Map | [PNG](pilot50_cut_qps_ratio_heatmap.png) | [PDF](pilot50_cut_qps_ratio_heatmap.pdf) |
| 정상 Continuous Native/Agg | [PNG](continuous_baseline_bridge.png) | [PDF](continuous_baseline_bridge.pdf) |

Continuous Baseline은 N200, QPS2.35와 설정별1회다.
이를 실제 Batch1/1/1 비교라고 표시하지 않는다.
Layer의 적격 C3 Bridge는 현재 Adapter에서 차단됐다.
Token C3의 진단 Timing도 적격 Layer/Token 우위로 사용하지 않는다.
