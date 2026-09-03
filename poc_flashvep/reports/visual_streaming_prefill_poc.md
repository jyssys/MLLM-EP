# Visual streaming prefill feasibility PoC

## Verdict

**FINAL STATUS: GO (bounded 2-image oracle; 4-image oracle not computable)**

`IMAGE_LEVEL_EQUIVALENCE=PASS`.  Twelve independent-vs-combined image
comparisons were repeated across two TP-visible worker shards and three
repetitions (36 comparisons total); every comparison passed the preregistered
cosine tolerance.  The clean two-image CUDA-event decomposition gives an
11.40% ideal streaming reduction (1.129x).  The four-image decomposition is
explicitly not claimed: cumulative prefix timings are non-monotone and the
causal segment increments include negative values.  No real streaming
prototype was run because the prototype is optional and would require a new
runtime handoff boundary; this branch contains no scheduler or model change.

## Configuration and workload

| Item | Value |
|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct (local snapshot) |
| Precision | BF16 |
| Runtime | vLLM 0.20.0 V1, eager |
| Parallelism | TP2 / DP2 / EP4 / PP1 |
| MoE backend | DeepEP high-throughput, TritonExperts, linear placement |
| DBO / prefix cache | off / off |
| Token budget | `max_num_batched_tokens=8192`, max model length 4096 |
| GPU mapping | `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1--4 only) |
| Images | local skimage astronaut, brick, camera, chelsea; RGB 448x448 |
| Visual tokens | 196/image (392 for 2 images, 784 for 4) |
| Repetitions | 1 warmup, 3 measured repetitions, one decode token |

The exact command and manifest are in the result directory.  Requests were
fixed and contiguous: four independent single-image encodes, one single-image
control, combined 2- and 4-image requests, and cumulative prefix requests
containing the first 1/2/3/4 images.  No image reorder, token pruning, route
change, or expert-placement change was used.

## Q1 — Image-level equivalence

The live `Qwen3_VisionTransformer.forward` output was saved for each worker.
For each combined request, its contiguous per-image slices were compared with
the corresponding independent image forward on the same TP-visible shard.

| Metric | Result |
|---|---:|
| Comparisons | 36 |
| Passed (cosine >= 0.999) | 36/36 |
| Minimum cosine | 0.999581 |
| Maximum absolute error | 0.765625 (BF16 quantization/outlier) |
| Median mean absolute error | 0.002779 |

`IMAGE_LEVEL_EQUIVALENCE: PASS`.  The greedy first token was stable across
all repeated calls (`A` for independent/single controls and `The` for the
multi-image/prefix prompts).  This is an output-agreement check for the
existing vLLM path, not a claim that a separately injected embedding tensor
has been validated: current vLLM does not expose an LM input boundary for
per-image embedding substitution.

## Q2 — Runtime boundary feasibility

The installed `qwen3_vl.py` path calls `encoder_eager_forward` ->
`Qwen3_VisionTransformer.forward(pixel_values, grid_thw)`.  Vision metadata
uses per-image `grid_thw`/`cu_seqlens`; `_process_image_input` later splits the
concatenated output by each image's merged-token count.  This provides the
structural independence needed for image-level work units.

The current call returns one concatenated tensor only after the complete
vision forward and then proceeds to LM prefill.  There is no image-ready
callback, per-image future, or supported API to inject one image's embedding
into a partial LM prefix.  Therefore:

`RUNTIME_CHANGE_REQUIRED: POSSIBLE_WITH_RUNTIME_CHANGE` — a small boundary
would need to expose per-image encoder completion and a safe embedding handoff
to a prefix-prefill queue.  This PoC does not implement that boundary.

## Q3 — CUDA timing decomposition

The hook records real CUDA events around the complete vision forward and each
decoder layer.  For prefix requests, `P_i` is the cumulative LM prefill layer
stack for the first `i` images; the per-image segment is the difference between
successive cumulative values.  Values below are medians over the six visible
worker/repetition samples (three repetitions x two TP-visible workers).

### Image encode (`E_i`)

| Image | E_i (ms) |
|---:|---:|
| 1 | 27.73 |
| 2 | 23.09 |
| 3 | 21.85 |
| 4 | 28.05 |

### Cumulative prefix prefill (`P_i`)

| Prefix images | P_i (ms) | Increment (ms) |
|---:|---:|---:|
| 1 | 104.27 | 104.27 |
| 2 | 151.76 | 47.49 |
| 3 | 116.04 | -35.72 |
| 4 | 106.83 | -9.22 |

Negative increments are retained rather than clipped.  They indicate that
the bounded layer-event trace is too noisy/non-monotone to support a
four-image causal oracle.  Clipping or choosing a favorable repetition would
be post-hoc tuning and was not done.

### Baseline wall observations (diagnostic only)

| Request | Median wall (ms) | p25--p95 (ms) |
|---|---:|---:|
| `multi_2` | 4924.26 | 4251.47--5224.13 |
| `multi_4` | 5277.09 | 4694.58--5490.12 |

These host wall values include per-layer CUDA-event synchronization, vLLM DP
engine scheduling, and one decode step.  They are not used as TTFT or as the
oracle denominator; the stage oracle uses paired CUDA-event work only.

## Ideal streaming oracle

For two images, the preregistered dependency is `E1 -> P1` while `E2` may run
in parallel with `P1`, followed by `P2`.  Using the measured segments:

| Quantity | 2 images |
|---|---:|
| Encoder sum | 50.81 ms |
| Prefix segment sum | 151.76 ms |
| Decomposed serial critical path | 202.57 ms |
| Ideal streaming critical path | 179.48 ms |
| Hidden time | 23.09 ms |
| Oracle reduction vs serial | **11.40%** |
| Oracle speedup | **1.129x** |

The 4-image oracle is `NOT_COMPUTABLE` because segments 3 and 4 are negative.
The generated `timeline_streaming_oracle.png` and
`baseline_vs_oracle.png` explicitly mark this rather than fabricating a
timeline.  The analyzer's gate is consequently scoped to the clean 2-image
case: equivalence PASS plus 10--15% oracle reduction gives `GO` under the
fixed policy.

`REAL_PIPELINE_PROTOTYPE: NOT_RUN`.  This is not evidence of an end-to-end
speedup: it is an oracle headroom result.  A prototype should only be built
after exposing the per-image callback/handoff and rerunning a clean repeated
timing protocol.

## EP segment scaling

No per-image partial LM call was introduced, so segment-level DeepEP
dispatch/expert/combine attribution was not measured.  The existing MoE
router, DeepEP collectives, expert execution, precision, and token order were
left untouched.  This PoC therefore does not assert that smaller prefix
segments are EP-efficient or EP-inefficient.

## Interpretation

The strongest positive evidence is numerical image-level independence: the
concatenated multi-image vision path yields per-image embeddings matching
independent forwards at BF16 tolerance.  The bounded 2-image CUDA-event
oracle also exposes 23.09 ms of useful overlap, exceeding the fixed 10%
`GO` threshold.

The strongest counter-evidence is that the same protocol cannot produce a
valid 4-image oracle: cumulative LM prefill work is non-monotone, and the
instrumented host wall is dominated by event synchronization and runtime
overhead.  The result is therefore not a production-ready streaming claim.

### Next single action

Add one minimal, read-only-compatible runtime boundary that emits an
image-ready embedding future (without changing model math), then rerun a
paired 2- and 4-image measurement with per-image LM prefix injection and
direct logits/token agreement.  Do not implement a general scheduler until
that boundary-level test is clean.

## Files

- Code: `poc_flashvep/visual_streaming_prefill_poc/`
- Result: `poc_flashvep/deepep_revalidation/results/visual_streaming_prefill_poc_20260903_150121_rep3/`
- Key outputs: `equivalence_results.csv`, `equivalence_summary.json`,
  `timing_results.csv`, `vision_timing.csv`,
  `decoder_layer_timing.csv`, `oracle_results.json`,
  `timeline_streaming_oracle.png`, `baseline_vs_oracle.png`
- Raw `.pt` tensors are retained locally under `vision_outputs/` but excluded
  by the repository's existing `*.pt` ignore rule; the equivalence CSV and
  summary are committed.
