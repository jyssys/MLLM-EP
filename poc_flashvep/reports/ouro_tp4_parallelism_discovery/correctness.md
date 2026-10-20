# Correctness and numerical qualification

## Frozen limits

These limits preceded the first GPU probe. No limit changed after a failure.

| Quantity | Frozen limit |
| --- | --- |
| Logit maximum absolute error | 0.75 |
| Logit mean absolute error per step | 0.08 |
| Hidden relative L2 per step | 0.03 |
| KV relative L2 | 0.03 |
| Operator relative L2 | 0.03 |
| Stable reference margin | 0.5 |
| Stable argmax mismatches | 0 |

Relative L2 is `norm(observed-reference) / norm(reference)`.
Raw files also retain maximum relative errors.
Elementwise relative errors near zero can be large. They are not the primary gate.

## Gate results

| Gate | Scope | Result |
| --- | --- | --- |
| C0 | Protected prior code, physical mapping, two locked legacy R4 cases | PASS with timing drift |
| C1 | Layer-0 attention, MLP, norm, full layer, K/V head slices, all loops and ranks | 192/192 checks PASS |
| C2 short | P128/O8, two prompt streams, aligned numerical and real greedy feedback | PASS in final eager and graph variants |
| C2 longer | P512/O32, full aligned logits/hidden/owner KV | FAIL against frozen internal limits |
| C2 P2048 | Custom implementation | NOT_RUN after P512 failure |
| C3 PP | Four concurrent KV namespaces and physical cross-request kernel overlap | PASS on bounded probes |
| C3 TP | Four-rank sharding, collective participation, short graph replay | PASS on bounded probes, full serving unqualified |

C1 contains eight operator inputs: four loops and prefill/decode positions.
The prefill length is 16. Decode adds position 16 with populated KV.
Each input checks six component values on each rank.
These checks do not prove full-depth numerical equivalence at every prompt length.

The MP correctness probe uses C4/P128/O8.
It duplicates two gold prompt streams into four distinct request namespaces.
Both PP placements match final logits, hidden values, tokens, and every owner KV element bitwise.
This is not four independent benchmark-quality samples.

## Failure and diagnosis

The first BF16-row-reduction probe failed on the second P128 request.
Its maximum hidden relative L2 was 0.034708, above 0.030000.
All eight aligned argmax IDs still matched the reference.

FP32 row accumulation and reduction corrected the short-probe gate failure.
The P512/O32 probe still failed.
Its maximum logit absolute error was 5.210938.
Its maximum per-step logit mean error was 0.802111.
Its maximum hidden relative L2 was 0.469359.
The full owner-KV relative L2 was 0.024168 on rank 0.
All 32 aligned argmax IDs matched on all four ranks.
The largest hidden deviations occurred at steps 18, 19, 26, and 27, using zero-based indices.

An unsharded custom-cache control matched the HF reference bitwise for all 32 P512 steps.
That result does not support a common cache-index or reference-adapter error.
FP32 column accumulation did not remove the long-shape failure.
A diagnostic SDPA head-geometry control reduced the maximum hidden L2 to 0.098947.
It still failed. That diagnostic repeats local heads and is not a performance baseline.
The results implicate numerical execution geometry as a candidate. They do not establish a unique root cause.

Native vLLM also passed the short probe and failed the longer internal limits.
Its bounded free-running outputs matched all 72 reference tokens across three shape probes.
The internal values below summarize rank 0. All four rank records remain available.

| P/O | Max logit absolute error | Max logit mean error | Max hidden relative L2 | Sampled KV relative L2 | C2 |
| --- | --- | --- | --- | --- | --- |
| 128/8 | 0.250000 | 0.045335 | 0.027809 | 0.009527 | True |
| 512/32 | 10.562500 | 0.764408 | 0.498974 | 0.009012 | False |
| 2048/32 | 1.937500 | 0.265983 | 0.176108 | 0.008571 | False |

This does not prove a model-quality regression or a mathematically incorrect TP decomposition.
It proves that numerical equivalence under this frozen contract remains unresolved.
We did not broaden the limits because tokens happened to agree.
The custom P512 natural-generation lane did not execute after the aligned numerical assertion failed.
The external natural probes permit EOS. All three reached their 8/32/32-token caps.
No GSM8K, MATH, or long-reasoning accuracy claim follows from these capped probes.

## Evidence

- `correctness/*rank*.json`: rank-local component, numerical, KV, and token observations.
- `tp_numerical_diagnostics.csv`: compact rank-0 numerical summary.
- `external_numerical_comparison.json`: all four external ranks and sampled KV comparisons.
- `correctness/unsharded_cache_probe.json`: bitwise unsharded control.
- `logs/`: assertions, failed attempts, and observer failures.

Full native tensor captures remain in the ignored `.tensor_cache/` directory.
`native_reference.json` records their hashes and reproduction source.
The repository contains numerical errors, token arrays, shapes, and raw timing evidence.
It does not commit the multi-GiB redundant tensor dumps.
