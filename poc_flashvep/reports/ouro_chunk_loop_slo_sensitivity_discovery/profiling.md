# Profiling scope and interpretation

## Separate lanes

Clean continuity and open-loop runs use light submission descriptors.
Heavy CUDA-event and Nsight jobs execute separately.
The heavy wall time does not enter headline serving latency.
`timing_contract.md` defines the clocks and denominators.

The light recorder captures scheduled request IDs, phases, row counts, depths, and loop groups.
Its host entry-to-return span is not GPU compute.
Its scheduled row count is not necessarily the padded CUDA Graph kernel row count.
Graph capture/replay/fallback counters are cumulative engine observations.
They are not a per-request graph-workspace allocation.

## Graph enablement is not universal graph execution

All ordinary arms enable the same native graph setting.
That does not put every prefill call inside a CUDA Graph.
The pinned executor marks decode recurrent spans with `emit_gate`.
Prompt-only spans do not set that flag.
`ModelRunner._prepare_forward` supplies recurrent graphs only when the batch has `emit_gate`.
The atomic prompt path prepares ordinary loop forwards without recurrent graphs.
Split and Grouped prompt-only recurrent calls also use the eager path.

Mixed can contain decode spans and therefore request a graph.
`RecurrentGraphs.select` rejects a query count above the configured graph maximum.
That maximum equals the resident sequence capacity, 4 or 8, in these ordinary configurations.
A mixed prompt/decode batch can exceed that count and fall back to eager execution.
This is native eligibility behavior under equal settings, not identical graph execution across policies.

For example, GPU4 `grid_repeat_s5_P8192_N8`, trial 10, records these cumulative counters.
Mixed C512 has 4 captures, 1021 replays, and 143 fallbacks.
Mixed C128 has 3 captures, 1021 replays, and 512 fallbacks.
Atomic C1024 has 7 captures, 1039 replays, and no counted fallbacks.
Prompt-only eager calls do not count as rejected graph selections.
These counters include warm-up and cannot serve as isolated per-request costs.
`graph_activity.csv` also records counter deltas where a previous cohort shares the same reused engine.

The source anchors are `worker/executor.py`, `request.py`, `worker/model_runner.py`, and `worker/cuda_graph.py` in the pinned RLT checkout.
The study does not attribute the full C/G effect to graphs alone.

## CUDA-event quantum surface

The separate profile queue covers C128/256/512/1024 for Atomic, Grouped, Split, and Mixed.
Primary profiling uses P≈8K/N8.
Another profile shape uses P≈2K/N4.
Staged G4 C1024 provides a repeated path control.
All prompt and decode work stays full R4.

`quantum_events.csv` records observed stream start/end and elapsed milliseconds.
It distinguishes core-containing prefill, separate prelude, and decode/coda submissions.
Atomic bundles prelude with recurrent work.
The staged modes can expose that prelude separately.
Complete request latency includes both.
The full quantum CSV includes background priming before the new long prompt arrives.
`long_quantum_response_surface.csv` filters the new long prompt by its actual request ID.
`profiling_tables.md` uses that subset.
This prevents short background prefills from entering the long-prompt quantum comparison.

The outer event interval contains staging, control, and dependency waits on its selected stream.
Nested forward intervals can overlap with that interval.
They cannot be summed into an E2E100% chart.
These traces have few independent process repetitions per shape.
They do not establish a production-ready held-out quantum predictor.

At P≈8K/N8, equal C×G does not imply equal stream elapsed time.
The following full-C means use the long-prompt subset, not background priming.

| C | G | C×G | Full-C mean stream elapsed, ms |
| --- | --- | --- | --- |
| 128 | 4 | 512 | 78.700 |
| 256 | 2 | 512 | 44.170 |
| 512 | 1 | 512 | 22.981 |

These are separate heavy measurements with few independent process repetitions.
The stream interval includes staging, waits, and host control.
The table does not identify a pure GEMM, HBM, or kernel-launch share.
It rejects a simple proportional C×G timing assumption for these observed execution paths.
It does not prove that a two-dimensional controller is necessary.
A calibrated one-dimensional deadline budget could still explain or recover part of this operating region.
That controller is NOT_RUN here.

## Nsight diagnostics

The installed profiler is Nsight Systems2024.6.2.225-246235244400v0.
Target capture starts after the declared engine warm-up through `cudaProfilerStart`.
CUDA Graph node tracing is enabled for this heavy lane.
The target stops capture through `cudaProfilerStop`.
CPU sampling and context-switch capture are disabled.
Only the selected worker process and its allowed CUDA device participate.

The portable exports include GPU activity, kernel/grid/block summaries, CUDA API totals, and NVTX ranges.
Full local captures remain under the ignored `private_profiles/` directory.
Export status and original logs preserve tool failures or unsupported behavior.
An absent trace is `NOT_AVAILABLE`, not an inferred service total.
All four target captures and CSV export commands completed successfully.
The local capture hashes appear in `run_manifest.json`.

| Heavy capture | Recorded device events | Union of device activity, ms | First-to-last activity span, ms |
| --- | --- | --- | --- |
| Atomic C1024, GPU4 | 800864 | 4173.219 | 7650.373 |
| Atomic C128, GPU7 | 1180380 | 5681.997 | 14073.664 |
| Grouped C1024, GPU5 | 831026 | 4259.312 | 7750.483 |
| Mixed C512, GPU6 | 836363 | 4297.341 | 8210.094 |

These captures include priming and drained generation, not only the long prefill.
Holes include host control and profiler effects as well as any runtime waiting.
They do not establish removable E2E percentages or a headroom oracle.
No clean E2E breakdown follows from the service totals.

`nsys_timeline_coverage.csv` measures the union of recorded device-activity intervals.
It separately reports the sum of recorded service durations.
The union can differ from the sum when streams overlap.
Neither value equals clean request E2E latency.
Holes between first and last profiled GPU activities are not automatically removable E2E cost.

The kernel-name summary uses explicit name heuristics.
It does not claim exact operator attribution for every generated kernel.
No NVLink, TP, EP, or context-parallel communication cost is invented for independent single-GPU runs.

## Memory

Every ordinary new engine reserves54GiB of KV.
Atomic does not use the staged-prefill hidden buffer.
At N8, staged C128/256/512/1024 uses4/8/16/32MiB respectively.
Peak allocated and reserved HBM are distinct fields.
Loaded parameter/buffer storage is another distinct accounting field in later startup records.
None of these byte counts measures physical HBM read/write traffic.
Actual loaded parameter/buffer storage is 2,869,305,602 bytes in the recorded startup audits.
This includes 1,434,652,673 BF16 parameters and a 64-element FP32 RoPE buffer.
It is not a per-token weight-read measurement.

Exact per-call padded graph geometry, HBM bandwidth, and achieved FLOP utilization remain unmeasured.
One-second `nvidia-smi` utilization cannot fill those gaps.
The source and event evidence supports repeated kernel invocations and scheduling explanations.
It does not justify a precise small-GEMM or bandwidth bottleneck percentage without further measurement.

## Observer controls

The final diagnostic queue compares light recording ON/OFF on every GPU.
The same warmed engine and trial inputs execute both conditions in alternating order.
`paired_observer_controls.csv` records output/depth parity and measured cohort-time ratios.
`control_summary.json` records the paired-block confidence interval.
All 12 matched cohorts preserved output tokens and depth records exactly.
The mean ON/OFF cohort-time ratio is 1.002225.
Its 95% block/cohort bootstrap interval is [0.997524, 1.007551].
The point overhead is 0.223%, and the interval includes zero overhead.
This control uses Atomic C1024 at P≈2K/N4.
It does not establish the same overhead for every policy or quantum count.
Earlier unpaired controls remain raw evidence but do not isolate observer overhead.
The reused legacy exporter shadows its `record` argument with a request dictionary.
Its raw `record_quanta` metadata field therefore does not identify observer enablement reliably.
Actual descriptor presence and the new paired control's explicit `observer_enabled` field identify the condition.
This metadata issue does not change the observer call or model execution.
