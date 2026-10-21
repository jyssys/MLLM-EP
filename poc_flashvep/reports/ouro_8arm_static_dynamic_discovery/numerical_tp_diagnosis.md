# TP numerical diagnosis

## Status

The historical strict P512 C2 gate remains **FAIL**.
The current TP task lane remains **NUMERICALLY_UNQUALIFIED_EXPLORATORY**.
The study does not raise the historical hidden relative-L2 limit of 0.03.
It does not use benchmark answers to select a new numerical threshold.

This failure does not mean that the repository lacks a real TP4 implementation.
The previous operator audit passed 192/192 operations. Real four-rank collectives and sharded attention/FFN weights exist.
The new task therefore measures actual exploratory accuracy instead of treating hidden error as an accuracy score.

## Five-engine comparison

The study compares HF Single, custom unsharded, custom TP4, native vLLM TP1, and native vLLM TP4.
Native numerical runs use the same vLLM version and attention backend.
The custom and native environments differ. `state/versions_*.json` records those differences.

The frozen synthetic inputs cover P128, P256, P512, and P2048.
Each length has two distinct inputs. P512 retains 32 aligned output positions.
The other lengths retain eight positions each. Each pair therefore contains 112 aligned token positions across eight synthetic requests.
These token positions are not independent serving or quality samples.

Custom unsharded matches HF bitwise on the captured seams and logits.
Every BF16 pair matches argmax at 112/112 aligned positions.
Native TP1 also differs from HF internally, although it does not shard the model.
Thus, the observed HF difference cannot be attributed entirely to TP sharding.

## Observed boundaries

`first_divergence.csv` records after-attention, after-MLP, and layer-output observations.
Each observation contains only the last token. It is not a full-prompt tensor comparison.
Native layer observations materialize the deferred residual for comparison. That observation is not fed back into the model.
Separate pre/post values for every normalization are not captured in this lane.

The 0.03 crossing in this CSV is a diagnostic level for intermediate components.
It is not a replacement gate for the historical final hidden-state contract.
The first crossing does not identify the operation that first introduced numerical error.

At P512 variant 0, HF versus native TP1 first crosses that diagnostic level at step 0, loop 2, layer 9, after attention.
HF versus native TP4 first crosses at the same sampled boundary.
Native TP1 versus TP4 first crosses at step 0, loop 3, layer 9, after attention.
These layer numbers are one-based. Decode-step numbers are zero-based.
The CSV retains the other input and all subsequent observations.

## Same-input FP32 operation references

The study measured 90 local attention/projection seams.
It includes the historical failed P512 input and zero-based steps 18, 19, 26, and 27.
The reference expands the same BF16 checkpoint values to FP32. It does not change the trained weight values.

Across these local seams, median attention relative L2 versus FP32 is approximately 0.001723 for both head geometries.
The maximum same-input 16-head versus four-head geometry difference is 0.0004283.
The median monolithic BF16 output-projection error versus its same-input FP32 reference is approximately 0.001652.
FP32 partitioned projection sums have median relative error approximately 1.39e-7 and maximum approximately 3.95e-7.

These results support a numerical-sensitivity explanation. They do not prove a unique full-model root cause.
`numerical/fp32_seams.json` contains the individual observations.
The local reference does not reproduce all native fused kernels or distributed execution.

## Bounded full-model FP32 reference

A separate diagnostic uses FP32 math attention and disables TF32.
It covers the same eight synthetic requests and 112 teacher-aligned positions.
It uses physical GPU5. It is not a BF16 serving baseline or a benchmark accuracy result.

All five BF16 variants match the FP32 argmax at 112/112 positions.
Maximum KL divergence, FP32 reference to candidate, is 0.0003156 nats for custom TP4.
The corresponding maxima are 0.0003674 for HF, 0.001230 for native TP1, and 0.0001808 for native TP4.
These maxima describe this calibration set only. They do not rank model quality.
The FP32 reference is a numerical reference, not a ground-truth answer oracle.
`logit_metrics.csv` preserves per-position logit errors, KL, JS divergence, margins, and top-five overlap.

## New masked Graph path

The initial new runtime had a separate Graph input-lifetime error.
It failed to retain a mask input tensor allocated before capture.
The fixed path retains that tensor. Graph and matched eager logits then agree bitwise at all eight bounded positions.
This error does not explain the historical eager TP4 failure.

The masked-capacity attention shape still differs numerically from exact-length BF16 attention.
An FP32 math diagnostic compares those two mathematical paths on identical inputs.
Its short-capacity run produces identical logits and key-cache values across eight positions.
A separate 4096-capacity diagnostic also records key and value errors explicitly.
These bounded diagnostics do not qualify long BF16 generation.

## Actual pilot quality is separate

The fixed native Single pilot scores 46/64 on GSM8K and 32/50 on MATH-500.
Native TP4 scores 45/64 and 34/50, respectively.
The raw outputs contain actual free-running generations, not frozen target tokens.
The GSM8K paired difference is -1.5625 percentage points. The MATH difference is +4.0 percentage points.
Question-paired intervals and answer changes are in `paired_accuracy_deltas.csv`.

This pilot does not show a uniform accuracy decrease across the two domains.
It also does not establish accuracy equivalence or full-benchmark fidelity.
Long outputs can diverge despite agreement on short, teacher-aligned numerical probes.
Full evaluation status must remain explicit in the final table.
