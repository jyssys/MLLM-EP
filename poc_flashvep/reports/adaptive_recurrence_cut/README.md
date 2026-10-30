# Dynamic A Evidence Index

Overall verdict: CONDITIONAL. The main nine-arm serving evaluation is NOT_RUN.
Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
Branch: `flashvep/adaptive-recurrence-cut-poc`.

Read the [final summary](../adaptive_recurrence_cut_final_summary.md) first.

| Artifact | Contents |
|---|---|
| `static_cut_metrics_by_start.csv` | Actual per-start TTFT, TPOT, observed ITL, E2E, and finite throughput |
| `fixed_cut_mean_latency_ci.csv` | Start-bootstrap intervals for fixed-cut mean latency |
| `paired_cut_vs_full_pd_ci.csv` | Paired-start ratios against the matched Full-Cut snapshot control |
| `request_latency_raw.csv` | All 1,208 measured map requests, with background/target flags |
| `e2e_component_shares.csv` | Per-request exclusive host-observed E2E percentages |
| `e2e_component_shares_aggregate.csv` | Component sums divided by summed all-request E2E |
| `migration_payloads.csv` | Actual NIXL bytes, completions, and payload accounting |
| `worker_memory.csv` | Weight, KV reservation, process-wide peak memory, and structural slab sizes |
| `native_pd_g1_vs_snapshot.csv` | Separate native streaming control, one paired start |
| `fixed_regime_selection_comparison.csv` | Frozen-choice homogeneous-regime composition, not Dynamic Serving |
| `holdout_regime_selection_ci.csv` | Three-start held-out composition uncertainty |
| `profiler_kernel_breakdown.csv` | Actual kernel counts, inclusive sums, and device busy unions |
| `profiler_directly_correlated_GEMM_time.csv` | Incomplete module mappings from External IDs and matrix shapes |
| `profile_kernel_timeline_bins.csv` | Clock-aligned profile visualization data |
| `main_nine_arm_status_matrix.csv` | All 270 main-serving cells, explicitly NOT_RUN |
| `experiment_matrix.csv` | Completed, failed, superseded, and unexecuted configurations |
| `stage_decisions.json` | Independent model and stage decisions |

The `raw/` directory preserves all outputs, token events, source archives, and profiler bundles.
The `logs/` directory preserves successful and failed experiment logs.
The `state/` directory preserves correctness, accounting, source, and index checks.
The `figures/` directory contains measured cut maps and profiler-only timelines.

Host-observed percentages are not GPU dispatch/GEMM/DMA percentages.
Finite cohort throughput is not sustainable serving capacity.
The fixed-regime composition is not a Dynamic Oracle or online-policy result.
Original-HF equivalence and downstream accuracy remain separate and unestablished here.

See [reproduction](reproduce.md), [correctness](correctness_and_execution_contract.md), and [profiling limits](profiling_and_limits.md).
