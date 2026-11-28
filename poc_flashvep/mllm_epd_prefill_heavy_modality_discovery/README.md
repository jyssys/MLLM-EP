# MLLM EPD operating-region reproduction

[Research spec](../reports/mllm_epd_prefill_heavy_modality_discovery_poc_spec.md) · [Final summary](../reports/mllm_epd_prefill_heavy_modality_final_summary.md) · [Runtime prerequisites](../reports/mllm_epd_prefill_heavy_runtime_setup.md) · [Workload and photo provenance](../reports/mllm_epd_workload_modality_audit.md)

## Frozen execution

The runner uses the existing pinned H200 EPD environment and owner-aware lease. It starts and stops its own workers and temporarily pauses only verified approved burners for GPU indices 4/5/6. Plans name exact manifest files, cohorts, recurrence-free model settings, arrival schedule flavor, seed, offered rate or multiplier, and the output-length contract. All measured requests include failures in the denominator. Every start verifies the model and image hashes and freezes the executable Python source. Existing GPU roles, approved assets and runtime live in sibling workspace directories.

- Primary checkpoint `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Runtime Python 3.14.6, PyTorch 2.11.0+cu129, vLLM 0.26.0+cu129, NIXL 1.4.1, BF16, FA3, token budget 8192, S128, KV 42 GiB.
- The supplied original-case manifest and image SHA receipts are in `results/`. Original media and large prompt-token arrays live outside Git.
- Each retained epoch includes executed worker start options and a source hash census. Legacy client receipts contain a stale `mm_processor_cache_gb=1` summary field. Actual per-worker argv explicitly uses zero. Request occurrence UUIDs differ and APC is disabled.

To repeat a retained epoch, extract its curated archive into a new external directory. Read `configs/frozen_plan.json`, reconstruct its media via the workload report, and set its manifest paths to the newly restored manifests. Preserve the published cohorts, orders and case counts. Keep the model and media prerequisites accessible to the appropriate GPU owner and use a new empty run directory.

```bash
namespace=poc_flashvep/mllm_epd_prefill_heavy_modality_discovery
pkg=poc_flashvep.mllm_epd_prefill_heavy_modality_discovery
study="$(realpath ../h200_epd_modality_runs)"
runtime=../mllm_epd_runtime_env/bin/python

"$runtime" -u -m "$pkg.runner" \
  --plan "$study/baseline_plan.json" --run-dir "$study/calibration-reproduction" \
  --p-budget 8192 --d-budget 8192 --kv-gib 42 --max-seqs 128 --wall-seconds 3600
"$runtime" -u -m "$pkg.runner" \
  --plan "$study/deflection_inline_plan.json" --run-dir "$study/inline-reproduction" \
  --inline-reroute --d-budget 8192 --wall-seconds 3600
"$runtime" -u -m "$pkg.runner" \
  --plan "$study/dp3_plan.json" --run-dir "$study/dp3-reproduction" \
  --layout dp3 --wall-seconds 3600
```

The opt-in inline reroute uses the existing one-shot CUDA IPC descriptor. As soon as E finishes, one final P/D recipient is chosen. P consumes a native request's descriptor and D consumes a D-local request's descriptor. The vLLM consumer obtains an atomic claim before cloning and sends a clone-complete ACK before E release. Compare the matched native proxy, same-route on-demand control, and the original NativeStrong EPD route. The E helper uses a separately verified model process and native NIXL on the same physical E GPU in both E/D matching arms.

## Metrics and observer qualification

TTFT includes open-loop dispatch lateness. The unit of request goodput is the count passing TTFT ≤5s, TPOT ≤33ms, and every ITL ≤34ms divided by planned-first-arrival to drain. Decode-only absolute SLO requires TPOT and every ITL. Native metrics also contain `SLO_goodput` in output tokens/s. Finite-cohort drain throughput is reported separately from sustainable QPS.

The natural kernel ROI spans scheduled-arrival p20–p80. A strict same-cohort clean/profile/clean comparison checks duration, TTFT p50/p95, TPOT p95, ITL p99, and exact output tokens. The guard caps both clean-to-clean spread and profile deviation at 5% per primary metric. Clock fitting and activity loss are checked independently. Failed observers produce `DIAGNOSTIC` intervals. Hardware GPM SM activity refers to NVML hardware sample windows, while CUPTI union refers to time covered by at least one actual CUDA kernel. The clean runs can be used directly for end-to-end timings. An externally observed request-free interval is a conservative proxy view of engine idleness. Proving absence of every copy requires a reliable transport channel for the native NIXL copy, which can be peer DMA outside GPU kernel activity.

## Rebuild reductions and figures

Use Python with Matplotlib installed for plots. Primary figures A/B/C are named and captioned in the operating-region, joint-saturation and idle-fragmentation reports. Kernel traces in figures preserve measured temporal intersections. Figures distinguish diagnostic, measured, and oracle quantities.

```bash
export MPLCONFIGDIR="/tmp/mllm-epd-modality-mpl"
python -m "$pkg.archive" --raw-root "$study" --out "$namespace/results/raw_evidence"
python -m "$pkg.reduce_epd" --raw-root "$study" --out "$namespace/results/region"
python -m "$pkg.plot_epd" --reduction "$namespace/results/region/operating_region.json" \
  --out "$namespace/results/region/figures"
python -m "$pkg.stage_capacity" --epoch "$study/calibration-sweep" --out "$namespace/results"
python -m "$pkg.routing_results" --epoch "$study/deflection-smoke" \
  --epoch "$study/deflection-inline" --epoch "$study/dp3-retry" --out "$namespace/results"
```

The teacher-forced diagnostic uses an independent `--logit-check` start and pins one original E output across three serialized passes. The compact logit summaries contain true full-vocabulary comparisons and the external raw artifacts retain the tensors. Native READ state capture starts with `--nixl-ready` and reads existing request and transfer state without device synchronization or added polling. Both diagnostics scope their conclusions to their specific observed cohorts.

[Curated request events, transfer receipts and true CUPTI activity streams](results/raw_evidence/README.md) · [Online metrics and figures](results/region/) · [Stage reference CSV](results/independent_stage_capacity.csv) · [Paired online route evidence](results/whole_request_routes.csv)
