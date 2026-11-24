# MLLM EPD Processor-result Handoff Control

Evidence: CORRECTNESS_ONLY and DIAGNOSTIC_C3. This is a common runtime control, not a new token split method.
Main100 and the original Main400 remain unchanged. The experiment uses Qwen2.5-VL-7B in original BF16 and BI0.
All arms use the same GPU4/5/6 E/P/D servers. The primary LLM scheduler limits remain B8192/S1024.
Encoder B114688/S1024 and synchronous encoder scheduling remain the same in every arm.
No source credit or artificial D admission cap is added. Prefix caching stays disabled.

## Data path

- `none`: E, P and D run the original renderer. P/D require the original E visual embeddings through Strict EC.
- `full`: E processes the original request online. P/D consume its exact processed prompt, including CPU pixel tensors.
- `grid_only`: E is unchanged. P/D retain every prompt token, original image grid, placeholder, embedding mask and hash.
The grid-only consumer payload omits redundant `pixel_values`. It does not omit images, visual tokens or encoder computation.
Strict EC and an explicit export check require every original E embedding. Missing exports cause failure, not local fallback.
The handoff uses request-scoped trusted tmpfs files. It is not cross-request caching or precomputed embedding replay.
Input fingerprints exclude only stage-only output, transfer and request-ID fields.
Each consumer receives its own current arrival timestamp. It does not inherit the encoder admission time.
Original model weights, Vision Encoder operations, token order, MRoPE and full 28-layer LLM execution stay unchanged.

## Correctness

Thirty CPU regressions passed before the GPU control. A later streaming-reader regression also passed.
The actual API Schema test first exposed an automatically generated request-ID mismatch. The source correction is preserved.
N1 and N8 fixed16 comparisons matched all explicit token IDs across none/full/grid-only.
N8 Natural EOS also matched every token across the three paths. Each path generated 526 tokens across eight requests.
All reported request input counts matched the frozen manifest.
This is same-geometry evidence. It does not waive the earlier cross-geometry encoder gate or the full task-quality gate.

## Main100 equal-output pilot

Every arm uses QPS2, arrival seed2026112402 and order seed2026112404. Every arm generates the same 5269 output tokens.
The code loads only replay counts from the reference. It does not retain the full 400-request reference tree during timing.
The same count-only client applies to every arm. No timestamps or SLO thresholds are corrected after execution.
The sequence includes a no-handoff warmup. The measured arms use one initialization, not three independent starts.

| Mode | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | Max ITL ms | TPOT p95 ms/token | All-ITL PASS % | SLO output tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 3932.987 | 8783.446 | 25.282 | 57.984 | 9.255 | 68.00 | 28.473 |
| full | 3399.799 | 7647.737 | 20.460 | 91.935 | 7.379 | 91.00 | 44.431 |
| grid_only | 2590.690 | 6914.666 | 15.772 | 92.634 | 6.829 | 97.00 | 61.024 |

The grid-only point improves TTFT p95 and request-level all-ITL attainment in this sequence.
This is not an independent-start confidence interval or a qualified serving success.
It supports redundant CPU processing and payload handling as removable runtime costs.
The full arm separates processor reuse from removal of redundant downstream pixel payloads.
It does not assign an exact GPU/CPU causal percentage. Extra handoff I/O remains part of the measured path.
Serialized buffer bytes are logical CPU payload accounting. They are not measured NIXL, NVLink or GPU DMA bytes.
Actual batch geometry was not captured in this pilot. A separate CUPTI/scheduler CPC control follows.
No offloaded LLM prefill or decode-safe insertion was performed. Generic Token Split remains NOT_RUN.
Future token/deflection comparisons must share any validated runtime repair with the Native baseline.
The legacy no-handoff results remain an ablation. They must not become an intentionally weak primary Native baseline.
CSV/JSON: `mllm_epd_prefill_heavy_token_split_20261124/analysis/processor_handoff_B2_replay100_q2_controls.*`.
The source archives, run configs, raw token records, all failures and handoff hashes are retained.
