# Full-model baseline profile

Status: **clean wall measured; fine breakdown DIAGNOSTIC_HIGH_OVERHEAD**

## Clean input-ready wall

These are current physical EP4, BF16, full-model measurements from the frozen
baseline workload. Loading and dataset download are excluded.

| Workload | Batch 1 | Distinct-image batch 4 |
|---|---:|---:|
| First generated token / TTFT proxy | 2,863.995 ms | 3,069.171 ms |
| Forced-length 32 generation | 5,792.639 ms | 5,995.280 ms |
| Forced-length 128 generation | 14,826.975 ms | 15,001.826 ms |

The 32/128 runs use `ignore_eos` forced-length generation. They are timing
anchors, not exact teacher-forced replay of an identical token sequence and
not quality measurements. A full natural-EOS benchmark E2E distribution was
not established; the bounded sanity outputs frequently hit their token cap.

## Diagnostic component pass

Median request-accumulated CUDA spans over the matched 11-request component
cohort were:

| Component | Median accumulated span |
|---|---:|
| Attention | 34.452 ms |
| Router/layout/residual inside routed stage | 28.478 ms |
| EP dispatch | 25.004 ms |
| Routed expert | 24.287 ms |
| EP combine | 6.425 ms |
| Decoder other | 1.287 ms |
| Routed stage total | 82.969 ms |

These device-local critical-rank spans are not a complete additive E2E
decomposition: vision, scheduler/host, synchronization, overlap, and other
full-engine time remain outside these hooks. They must not be divided by an
unmatched request wall to claim a validated E2E share.

The matched full-category profiler increased weighted wall time by **13.550%**
(median per-request 17.254%), above the 10% rejection threshold. A lower-hook
retry stalled at a DP barrier and was safely stopped. Therefore:

- `ROUTED_STAGE_E2E_SHARE=NOT_VALIDATED`;
- clean wall values remain valid;
- the component table is diagnostic only and cannot gate an optimization.

Evidence: `artifacts/analysis/full_model_profile.json` and
`artifacts/profiling/ep4_clean_matched_20260924/`.
