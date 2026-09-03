# Real visual-streaming prefill pipeline PoC

## Final verdict

**FINAL STATUS: `NO_GO`**

`STREAMING_CORRECTNESS=PASS` for the completed real 2-image handoff pair
(8/8 greedy tokens agree), and the previous independent/concatenated encoder
check was 36/36 with minimum cosine 0.999581.  The correctness result does
not compensate for the latency and reliability result: the primary direct
prefill critical path was **0.33% slower** with streaming, not faster, and
three additional repeated custom-stream runs hung before the second image
could be consumed.  The fixed gate therefore fails the required `>=2%`
actual reduction and the runtime-reliability requirement.  No 4-image
prototype was attempted because the preregistered 2-image `>=3%` condition
was not met.

## Configuration and exact workload

| Item | Value |
|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct (local snapshot) |
| Runtime | vLLM 0.20.0 V1, eager |
| Precision | BF16 |
| Parallelism | TP2 / DP2 / EP4 / PP1 |
| MoE | DeepEP high-throughput, TritonExperts, linear placement |
| DBO / prefix cache | off / off |
| GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical 1--4 only) |
| Token budget | `max_num_batched_tokens=256` |
| Prompt | 483 tokens, two 448x448 local images, 196 visual tokens/image |
| Image spans | `[8,203]`, `[270,465]`; image 2 is outside chunk 1 |
| Decode | 8 greedy tokens for the primary pair |

The exact manifest and commands are in
`deepep_revalidation/results/visual_streaming_real_pipeline_poc_20260903_164000_analysis/`.
The code was based on commit `334e4128974435a125d4c0b5327e739b1416bf38`.

The fixed experimental path is deliberately narrow: on a request named
`streaming_*`, image 1 is encoded on the normal stream, image 2 is launched
on a side CUDA stream, and a CUDA event is waited on when the scheduler reaches
image 2's token range.  Baseline remains the stock multimodal path.  No token
reordering, routing, expert-placement, precision, or decode scheduling change
was made.

## 1. Correctness gate

| Check | Result |
|---|---:|
| Prior independent vs concatenated image embedding comparisons | 36/36 pass |
| Prior minimum embedding cosine | 0.999581 |
| Completed real handoff pair (8-token greedy sequence) | 8/8 exact |
| Current prompt-token count baseline vs streaming | 483 vs 483 |

The exact token sequence in the primary pair was
`[785, 1156, 2168, 4933, 264, 8778, 46633, 304]` for both modes.  Thus
`STREAMING_CORRECTNESS=PASS`; this is output agreement, not permission to
claim a speedup.

## 2. Direct segment timing

The layer hook records one CUDA event interval per decoder layer.  For each
contiguous prefill invocation, the critical path is the maximum sum over the
two TP-visible worker traces.  This is a direct measurement of each partial
prefill call, not a difference of cumulative prefix timings.

### Primary stablecheck (decode 8, one complete pair)

| Mode | Chunk 1 (256 tokens) | Chunk 2 (227 tokens) | Direct prefill critical path |
|---|---:|---:|---:|
| Baseline | 93.7248 ms | 94.1018 ms | **187.8266 ms** |
| Streaming | 93.4977 ms | 94.9497 ms | **188.4475 ms** |

`REAL_STREAMING_REDUCTION = -0.3306%` and `REAL_STREAMING_SPEEDUP = 0.9967x`.
The streaming first chunk is 0.24% shorter, but the second chunk is 0.90%
longer, so the end-to-end direct prefill path regresses slightly.

The host wall (prefill plus the requested decode suffix and engine overhead)
was 3442.40 ms baseline versus 3414.93 ms streaming, or +0.80% apparent
reduction.  This is one pair, includes decode, and is not the primary metric.

### Secondary handoff diagnostic (decode 1, one complete pair)

| Mode | Chunk 1 | Chunk 2 | Direct prefill critical path | Host wall |
|---|---:|---:|---:|---:|
| Baseline | 100.3654 ms | 95.0105 ms | 195.3759 ms | 3527.09 ms |
| Streaming | 93.6274 ms | 95.5260 ms | 189.1534 ms | 3394.31 ms |

This isolated one-token diagnostic showed +3.18% direct-prefill reduction and
3.76% host-wall reduction, but it uses a different decode suffix and one
sample.  It is retained as evidence of variance, not combined with the
primary gate.

There was no valid 20--30-repetition custom-stream sample: the intended
repeated run was attempted with 20 repetitions, while stock-control runs
completed 20 baseline/streaming pairs before the real hook was enabled.  The
stock-control medians (not real streaming) were 3403.77 ms and 3372.87 ms
including decode.  Their interpolated p25--p95 ranges were 3357.84--3803.29
ms and 3332.23--3859.37 ms, respectively; these are retained only as a
drift/control reference.

## 3. Image encoding and handoff

Direct CUDA-event observations from the completed primary pair:

| Unit | Median observed time |
|---|---:|
| Image 1 encode (`E1`) | 27.68 ms |
| Image 2 encode (`E2`) | 21.36 ms |
| Image 2 handoff | image-2 token range begins at 270; event recorded |

The side-stream event was consumed successfully in the stablecheck pair.
The event was not reliably consumable in repeated runs: the side-stream
vision path entered a TP collective/all-gather and the first LM chunk could
finish its gather call, but no decoder-layer rows followed.  Four aborted runs
(`162016`, `162547`, `162856`, `163153`) stopped at `streaming_2_r0` after
recording image 1 and the handoff; no image-2-ready event/decoder completion
was observed.  This is a runtime reliability failure, not a fabricated timing
sample.

## 4. Interference decomposition

The persistent serving process does not expose an isolated `E2 + P1` wall
without adding another request/scheduler path, so no value is invented.

| Measurement | Value |
|---|---:|
| `E2` observed alone before handoff | 21.36 ms |
| `P1` baseline first-chunk critical path | 93.72 ms |
| Streaming first-chunk interval | 93.50 ms |
| Isolated `E2 + P1` concurrent wall | not exposed by this bounded path |
| Repeated-run outcome | 4 custom runs hung before image 2 consumption |

`RESOURCE_INTERFERENCE=HIGH` operationally: although one pair completed,
side-stream vision execution is not a safe repeatable partner for the stock
TP/DeepEP engine.  The most likely mechanism is contention/interference from
the vision path's TP all-gather/collective on the same process/GPU group while
the LM/DeepEP path is making progress.  The logs show `gather_enter` and
`gather_exit` for the first 256-token chunk, followed by no decoder rows in
the diagnostic hang; this is consistent with a collective/stream dependency
problem, not evidence of useful hidden work.

## 5. EP segment scaling

The existing read-only hook surrounds the complete decoder layer only.  It
does not add DeepEP calls or expose dispatch/expert/combine boundaries.
Consequently `dispatch_ms`, `expert_ms`, and `combine_ms` for each partial
segment are explicitly `NOT_MEASURED` in `ep_segment_scaling.csv`; they are not
inferred from the layer total.  The evidence is therefore
`EP_SEGMENT_SCALING=INCONCLUSIVE`, not a claim that splitting is EP-benign.
The measured complete chunk totals are the segment-level primary timing shown
above.

## 6. Oracle versus real boundary

The prior decomposition oracle used 202.5724 ms serial and 179.4843 ms ideal
streaming, an 11.3975% reduction.  The current direct primary prefill result
is -0.3306%; therefore the oracle's hidden time was not realized and the
realization ratio is intentionally `null` (a negative measured result is not
reported as a positive fraction of oracle headroom).  The secondary one-token
diagnostic is not a comparable oracle validation.

| Quantity | Result |
|---|---:|
| Prior oracle reduction | 11.40% |
| Primary direct real reduction | -0.33% |
| Oracle realization ratio | not meaningful (primary reduction <= 0) |
| 4-image extension | **NOT_RUN** (2-image >=3% prerequisite failed) |

## 7. Decision and next action

The fixed decision gate is `NO_GO`: correctness passed, but the primary direct
prefill reduction is below 2%, the sample is not repeatable, and side-stream
collective interference caused multiple hangs.  The apparent single-pair host
wall reduction is not sufficient because it includes decode/engine variance
and conflicts with the direct prefill measurement.

**MAIN BOTTLENECK:** the minimal image-ready handoff is not a clean causal
boundary in the current vLLM execution.  Launching image 2 on a second stream
can interfere with Qwen vision TP collectives and prevents reliable LM/DeepEP
progress; when it completes, the partial-prefill boundary adds no measured
critical-path gain.

`NEXT: FULL METHOD DESIGN = NO`.  The single next action is to expose (or
prove impossible) a safe per-image encoder boundary that avoids concurrent
vision TP collectives with LM/DeepEP; only after a repeated 2-image direct
measurement is stable should any scheduler work be reconsidered.  No 4-image
or production integration should proceed from this result.

## Artifacts

- Code: `poc_flashvep/visual_streaming_prefill_poc/`
- Analysis result: [`visual_streaming_real_pipeline_poc_20260903_164000_analysis`](../deepep_revalidation/results/visual_streaming_real_pipeline_poc_20260903_164000_analysis/)
- Required artifacts: `correctness_results.json`, `paired_latency.csv`,
  `segment_timing.csv`, `ep_segment_scaling.csv`, `interference_results.json`,
  `oracle_vs_actual.json`, `gate_summary.json`,
  `baseline_vs_streaming.png`, `segment_timeline.png`
- Raw trace copies: analysis result `raw/`; full original run directories are
  preserved and are not overwritten.
