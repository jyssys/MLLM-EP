# Large, exactly token-matched Qwen3-VL transfer diagnostic

Completed 2026-09-09. `vl_transfer/large_matched_control_20260909_v1` contains
three randomized independent observer/clean engine pairs, 480 measured requests
(240 clean), with four real ChartQA images per visual request and natural-text
controls. Physical GPUs 4–7; BF16 TP2/DP2/EP4, verified DeepEP HT path, DBO off.
This is a common vLLM transfer diagnostic, not a native VL port of a base paper.

All **120 paired clean requests have exactly equal actual prompt token counts**.
No processor-count prediction mismatch. Short prompts have 1733/1821/1909 tokens;
long prompts add 2048 natural-text tokens. The input manifest retains original
image identities/resolutions and paired IDs. Images are not artificially upsampled.
Per-DP simultaneous cohort sizes are 1 and 4. This is controlled cohort serving,
not a claim of arbitrary open-loop native MLLM scheduling.

| Family | Median clean mean E2E across restarts | Entire-TTFT zeroing / E2E, median |
|---|---:|---:|
| Four charts + long text | 3.5191 s | 15.6647% |
| Four charts + short text | 4.1550 s | 13.9850% |
| Matched long natural text | 3.2033 s | 9.4381% |
| Matched short natural text | 3.0693 s | 6.9441% |

The last column is an intentionally unrealistic **fixed-request-timeline,
prefill-only bound**, not removable scheduling waste, not a feasible oracle,
and not a native counterfactual. It includes unavoidable image processing,
encoding, normal prefill and queueing. Do not subtract it from every overlapping
request or use it as an upper bound on all possible scheduling improvements.

## Common-state and correctness caution

The sparse observer does not yield a stable overhead estimate on this larger
screen: one block is 7–9% faster with instrumentation and another 52–58% slower
across both text and image families. Median effects hide that common-state
variation. Instrumented stage durations are diagnostic only; no stage-cost
projection is promoted as clean request improvement. A first-block short-image
TTFT outlier inflated its single-run bound to 28.79%; all blocks remain included.

Matching total tokens does not match expert histograms, image-encoder work or
task content. Thus the family difference is not a demonstrated modality-caused
failure of any of the three schedulers. Some long free outputs differ between
observer runs; no task-quality-equivalence claim is made. No successor is
promoted from these rows.

Reproduce CPU summaries with `analyze_transfer_cpu.py` per restart,
`analyze_observer_restarts.py`, and `analyze_large_matched_transfer.py`.
Their inputs and joins are preserved; rank rows are not summed as duplicate
request latency. All observed logical invocations have four-rank joins and
unknown request IDs are empty in each completed block.
