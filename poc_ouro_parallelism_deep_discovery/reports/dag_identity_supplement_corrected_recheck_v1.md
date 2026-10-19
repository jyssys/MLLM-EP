# Supplemental-checker corrected recheck v1

The coordinator corrected both findings from the [independent review](dag_identity_supplement_independent_review_v1.md). This recheck records the corrected state separately. The earlier findings remain pinned to their original source hash.

The corrected checker hash is `ef39e4a6f859bcd3427461bc9ce24b38b17e6a5eceff1cb6ee58bc2a23bd1ae1`. It rejects external keys that also have an in-graph writer. It explicitly checks integer identity indices and request/committed field types.

I reran both minimal cases. The namespace collision raises `External KV cannot also have an in-graph writer`. The fractional range raises `KV identity indices must be integers`.

All 30 combined CPU tests pass: seventeen supplemental, five retained characterization, and eight original tests. The original checker and characterization-file hashes remain unchanged. The [coordinator's red record](../artifacts/dag_identity_review_red_v1.json) preserves eight failures or subtest failures before correction. I inspected that record. I did not execute its historical failing command.

These corrections close the two observed static declaration gaps. They do not prove original-gate CDF arithmetic, complete model edges, runtime values/publication, allocator lifetimes, aliases, resource feasibility, or validated speculation. No model or GPU executes. No larger validator follows from this review.

The [recheck record](../artifacts/dag_identity_supplement_corrected_recheck_v1.json) preserves hashes and actual times. The complete worker interval through validation is 16:07:36–16:10:33 UTC, or 177 seconds. The earlier 99-second cutoff is a prefix of that interval, not additional time. Final record formatting follows the cutoff.
