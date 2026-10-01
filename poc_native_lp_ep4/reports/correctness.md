# Native correctness

## Final native revision (v4)

PASS: **30 Full-versus-LP comparisons**: AUTO/G8/G16/G24/G48, visual and
matched-text inputs, three repeats per input/configuration. Final normalized
prefill hidden states and first-token logits have max-absolute and relative-L2
error **0**. First tokens and all **eight actual unforced greedy continuation
tokens** match. Prompt IDs, MRoPE positions and actual multimodal input
embeddings match. The continuation check resolves the injected request ID
explicitly; it does not infer identity from output length.

Every EP rank records stages 0..G-1 in identical order. Each partition covers
the 48 prefill layers once. All 48 decode layers execute with four resident
rows per DP at every stage, and each resident emits one actual token per stage
iteration. G48 therefore verifies the one-layer-per-stage boundary too.

Acceptance source: `artifacts/correctness/release_validation_v4.json` and the
compact raw export of `release_correctness_v4`. This low-overhead correctness
lane deliberately does not download router IDs: its route-identity fields are
N/A, not false PASS. Exact per-source-rank route ID/weight hashes are separately
checked in the rich route lane; previous completed rich checks below also
passed. No correctness/trace latency is used as the clean serving headline.

## Earlier independent rich checks (retained)

PASS: G4/G8/G16, three paired deterministic repeats each.

All 2029 final normalized prefill hidden rows and the first-token logit vector
were bit-exact versus ordinary full prefill in every comparison: max absolute
and relative L2 errors 0, row cosine 1, logit JSD 0. First-token equality and an
eight-token unforced greedy continuation both passed.

Four-rank stage logs confirm contiguous ranges, monotonically advancing stage
state, 48 prefill layers covered once, and 48 decode layers executed on every
stage with four residents per DP. Stage ordering matches across all EP ranks.
The same image, prompt, processor and MRoPE/KV mappings feed both arms; only
scheduling differs. Stored GPU hidden/residual state supplies the next stage.

Raw bounded hidden/logit tensors are local and git-ignored. Compact errors and
checks are in `artifacts/correctness/validation.json`. The first smoke was
interrupted while tightening implementation and teardown; the later complete
three-repeat run is the acceptance evidence. No timing gate is based on an
interrupted run.
