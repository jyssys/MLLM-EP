# Evidence map — scheduling/overlap study only

This is **Layered Prefill / FastPP / NanoFlow**, not SERE/Libra/MoDES.
Main report: `poc_flashvep/reports/scheduling_overlap_successor_mining.md`.
Result root: `poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/`.

## Evidence tiers and primary inputs

| Evidence | Result-root-relative path | Scope |
|---|---|---|
| Layered original graph direction | `layered_runs/graph_screen_20260909_v1/` | 9 native TP2 engines; descriptive quality-limited request comparisons |
| Layered stronger existing-knob attack | `layered_runs/full_warmup_cap_screen_20260909_v1/` | 12 native graph engines; full same-trace warmup |
| Layered native layer/route mechanism | `layered_runs/mechanism_resume_20260909_v1/` | Instrumented eager traces, not clean performance |
| FastPP native MoE policy screen | `fastpp_runs/qwen3_screen_20260909_v1/` | 9 PP4 engines; no EP claim |
| FastPP actual ALP/rank trace | `fastpp_runs/qwen3_mechanism_alp_v3/` | 3,950 complete four-PP-rank identities |
| FastPP real partition intervention | `fastpp_runs/static_partition_control_20260909_v2/` | 3 pairs; same KV capacity/full warmup |
| FastPP five-knob final control | `fastpp_runs/qwen3_full_warmup_chunks_20260909_v1/` | 15 native PP4 engines complete; common KV/full-trace warmup |
| NanoFlow original entry correctness | `raw/nanoflow_official_smoke_correctness.json` | Independent HF smoke, not performance |
| NanoFlow stream overlap | `nanoflow_runs/nsight_resume_20260909_v1/` | CUDA/NVTX profiling, excluded from clean latency |
| NanoFlow large prefill portfolio | `nanoflow_runs/prefill_portfolio_20260909_v1/` | 18 native EP4 engines; fixed cohorts; manual plans |
| NanoFlow larger decode portfolio | `nanoflow_runs/decode_volume_portfolio_20260909_v1/` | 36 native engines; correctness-gated envelope insufficient |
| NanoFlow setup amortization control | `nanoflow_runs/long_decode_amortization_20260909_v1/` | Same natural inputs; output horizon 16 to 96; fixed cohorts |
| NanoFlow same-prefix/HF numerical control | `nanoflow_runs/numerical_resume_20260909_b16/`, `hf_numerical_reference_20260909/` | Teacher-forced; excluded from performance |
| NanoFlow resource/HF prefill diagnostic | `nanoflow_runs/numerical_prefill_resource_20260909_v1/`, `hf_prefill_resource_reference_20260909_v1/` | Near-tie and projection-shape evidence |
| Small actual-image VL observer control | `vl_transfer/lite_control_20260909_v1/` | Six real vLLM engines; not native paper ports |
| Larger exact-token matched VL | `vl_transfer/large_matched_control_20260909_v1/` | Six engines, 120 exact-volume visual/text pairs |
| Frozen workloads | `request_traces/` | Request IDs, content/token counts, arrival schedules, hashes |
| Runtime provenance | `cpu_analysis/environment_provenance.json` | Distribution metadata, source pins, diff/patch hashes, authorized GPU UUIDs |
| Experiment-time accounting | `cpu_analysis/interim_accounting.json` | Union of recorded intervals; not CUDA busy-time |

## Tables to read before raw logs

Native request screens provide `request_run_summary.csv`, `paired_comparisons.csv`
and `restart_aggregate.csv`. Effect sign is positive for latency reduction.
Bootstrap intervals resample independent restart pairs, not requests.
`existing_policy_envelope.json` is restricted to its explicit existing options.
`existing_knob_slo_envelope.json` is attained fixed-arrival goodput, not capacity.

NanoFlow uses `paired_results.csv` with a hard output-equality label and
`finite_plan_envelope.json`. An insufficient correctness-gated portfolio has
null headroom, not zero. No profiled or teacher-forced run is a clean gain.

Common `REQUEST_E2E_RESULTS.csv` is a searchable index with per-row provenance
and correctness scope, not a pooled cross-system benchmark. Different models,
precisions/topologies and workload pools cannot be compared by raw latency.

Large tensors, checkpoints, native build products, reference source checkouts
and Nsight binaries remain local. Versioned harnesses, compatibility patches,
audits and selected small summaries provide a reviewable Git handoff; their
absence from Git does not imply raw evidence was deleted. No burn time is
counted as research execution.
