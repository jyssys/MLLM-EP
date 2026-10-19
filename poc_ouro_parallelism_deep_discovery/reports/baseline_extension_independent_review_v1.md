# Independent review of the baseline extension

## Blocking issues first

No model-semantics or sample-exclusion launch blocker remains in the inspected sources and frozen metadata.
The launch contract specifies a 14400-second outer timeout per arm.
This resolves the earlier concern about the guard's 1800-second default.
The 1200-second alarm covers generation only.
The outer timeout also bounds loading, warmup, and synchronization.

One analysis condition remains explicit.
Freeze the grader and completion rule before outcomes.
The base scorer marks every non-EOS response as truncated.
The existing campaign analysis instead marks only the length cap as truncation.
It treats EOS and frozen stops as completed.
This review pins both source hashes before launch.
The old analysis script cannot process the new IDs and gzip files unchanged.
This is an analysis requirement, not a generation-policy change.

## CPU and metadata checks

Independent reconstruction exactly matches the 64 frozen rows and their order.
There are 32 unique requests per domain.
Overlap is zero with all 80 original campaign IDs, earlier pilot IDs, and protected quality/speed metadata IDs.
All three metadata hashes and both model revisions match the frozen protocol.
All 64 gzip filenames are unique.
I opened no protected result file.

The three existing metadata/import tests pass.
Tokenizer-only CPU checks pass for both pinned Base checkpoints.
Each tokenizer gives prompt lengths from 33 to 384 tokens for all 64 requests.
No prompt plus the 8192-token cap exceeds 10240 tokens.
Both tokenizers report EOS 0.
Every configured stop sequence encodes to a nonempty sequence.
No model weights loaded, no model executed, and no CUDA context initialized in these checks.

“Held-out untouched” must refer to the original campaign's frozen held-out IDs.
It must not imply that every upstream metadata split remains unused.
The exclusion check supports the narrower campaign-specific statement.

## Semantic and artifact review

The new runner matches `quality_pilot.py` for the inspected generation path.
Both use the existing loader with graphs enabled, batch 1, 640 blocks, and model length 10240.
Both use the same four-token warmup, prompt encoding, stop encoding, EOS, cap, and request-wall boundary.
The exit callback records depth, then calls the original bound callback.
Both require one fewer decode-depth record than output tokens.
Fixed mode permits depth 4 only.
Adaptive mode permits depths 2, 3, and 4.

The loader preserves BF16, full-R4 prefill, checkpoint gate identity, and `last_exited` KV.
It disables synthetic replay, delayed gate consumption, and asynchronous batching.
The pinned CDB coda source uses argmax in both graph and eager paths.
No new sampler, gate, threshold, engine, candidate, or graph comparison enters this extension.

Each gzip record preserves prompt IDs, output IDs, text, stop classification, depth events, statistics, wall, model pin, and protocol hash.
The copied frozen protocol supplies the source prompt and answer metadata.
The current runner closes raw gzip before its depth assertions.
Thus, a depth-validation failure retains the returned tokens and events.
An exception inside generation cannot preserve a result that the engine never returned.
A failed compression write can leave a partial gzip file.

Caught exceptions preserve failure records and keep arm success false.
External termination need not execute Python's finalizer.
Analysis must therefore require both runner and outer guard success.
It must also check all 64 expected IDs, valid gzip content, unique IDs, and consistent hashes.
Partial files are diagnostic evidence, not accepted benchmark rows.

## Interpretation boundaries

Use request-level intervals and paired adaptive-minus-fixed outcomes on the same IDs.
Do not treat tokens or depth events as independent quality samples.
The selected benchmark requests do not establish a new natural-population sample.
The extension does not establish noninferiority or candidate quality.

Label the sum of recorded depths as decode core-pass work.
It excludes full-R4 prefill, the first output token's production, warmup, and non-core costs.
It is not total FLOPs, total latency, or a simultaneous causal frontier.
Keep request-weighted and token-weighted summaries separate.
Independent one-GPU walls cannot establish architecture speedup or component attribution.

An outer timeout is operational censoring, not answer truncation at the token cap.
Do not count unexecuted requests as incorrect answers or silently remove them from the denominator.
Do not construct a paired claim from incomplete arms.
No post-outcome parser, tolerance, cap, or sampling change is admissible.

The [structured review](../artifacts/baseline_extension_independent_review_v1.json) pins source paths and CPU observations.
This review changes no runner, protocol, prior report, or raw evidence.
No GPU ran during this review.
