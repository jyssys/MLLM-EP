# Native correctness

## Native model revision (v4; unchanged in v6/v7)

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

After the trace-only event-pool and rendezvous repair, a fresh v7 three-repeat
bimodal Full/AUTO/G8 run passed **12 additional comparisons**. Hidden/logit
max-absolute error, relative L2 and logit JSD remain 0; input identity,
unforced eight-token continuation, stage order and real resident-token emission
pass. See `artifacts/correctness/pooled_validation_v7.json`. A separate v5
Full/G48 boundary recheck passed six comparisons after atomic admission changed.
Neither runtime repair changes the native attention/router/expert model code.

Resident forced-token parity is a **system-control integrity check**, not an
unforced task-quality score. The injected request's eight-token continuation
is unforced greedy and is checked separately. Historical full benchmark scores
are reused rather than replaced with a one-token quality claim.

The completed **current v7 rich lane** passes 30 exact Full-versus-LP route
comparisons (AUTO/G8/G16/G24/G48, both inputs, three repeats). Each comparison
has all 96 layer/source-rank records and identical selected expert-ID and
router-weight SHA256. All 72 measured rich waves preserve the expected
sum(k)=prompt_rows×8×48 and complete four-rank active-prefill frames.
Loaded ownership and original token order are checked separately from timing.
See `final_metrics.json` and the compressed raw rich runs for these checks.

The numerical comparison reports max-absolute/relative-L2 error zero and
logit JSD zero. Self-cosine calculations on large FP32 logit vectors can be
slightly below 1 from rounding even for identical tensors; the measured
minimum (about 0.999986) is retained, not rounded into a perfect cosine claim.

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
