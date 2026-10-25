# Correctness and quality interpretation

## Native baseline checks

All three native adapters passed exact no-op logit equality.
They also passed the q=0 logit check and short greedy token parity.
The latest P0 runs use the `native_v5` suffix.
Earlier failed P0 attempts remain in the archive.

The observers count calls at each physical site.
They require 4, 8, or 16 passes at the corresponding recurrent sites.
They require one call at each nonrecurrent outer site.
This prevents prelude or coda layers from inflating temporal overlap.

The parameter check compares storage pointers, dtypes, and tensor versions.
It does not permit parameter mutation during an intervention.
All learned parameters remain BF16.
Native router reductions, logits, and positional buffers retain their original precision.

## Backend qualification

The BF16 explicit-attention comparison failed its original whole-model L2 threshold for both IFM models.
The threshold remains 0.05.
The failed results are not relabeled.

Separate operator checks use actual post-RoPE Q/K/V from the pretrained models.
The original independent operator checks stayed below a 0.01 relative-L2 limit.
They also preserved eight greedy output tokens.
The later strict diagnostic disables TF32 for its reference matrix products.
Its records remain separate from the original P0 artifact.

The strict diagnostic passes the 0.01 operator threshold for both IFM models.
Its largest operator relative-L2 errors are 0.00200237 for Ouro and 0.00199615 for Huginn.
However, its eight-token output differs at token 4 for both models on that different prompt.
The corresponding whole-model relative-L2 errors are 0.04163234 and 0.04061466.
Neither the operator result nor the smaller whole-model error establishes generation equivalence.
Both IFM P0 outcomes therefore retain a backend-fidelity qualification.

All three official generation APIs match their adapters on a separate eight-token smoke prompt.
That comparison uses the same native backend on both sides.
The final strict Foil loader reports no missing, unexpected, or mismatched tensors.

Fourteen legacy q=0 rows record top-1 agreement as 0.9999999403953552.
A CUDA reduction of 230 all-true values reproduces that exact value.
All affected rows have zero NLL and KL differences.
The integrity test allows one FP32 epsilon for this aggregate only.
It does not relax the model-logit criterion.
New metric code uses an integer agreement count and an explicit denominator.
The raw records remain unchanged.

These checks establish an internally controlled native FA4 baseline.
They do not prove FA3, BF16 explicit attention, and FA4 have identical whole-model outputs.
They do not reproduce the paper's full benchmark evaluation.

## Intervention semantics

The omission path compacts rows before calling the native expert GEMMs.
It does not compute every expert output and then call that omission a speedup.
The output-reuse path bypasses fresh expert rows and uses cached raw outputs.
The current router weight remains active.

Removing rows does not certify a reduction in padded GPU instructions.
The grouped kernel can retain its launch count and tile geometry.
The native-weight operator microbenchmark measures that limitation separately.

The q=0 path must preserve exact native logits.
Matched random controls use the same realized teacher-forced row budget.
The cache tests prevent token or expert-slot confusion.
Every full forward resets prior request state.

## Scoring

The NLL evaluator shifts targets explicitly.
It never uses Foil's auxiliary-inclusive model loss.
It passes no padding mask to the model.
Teacher-forced NLL, KL, and top-1 agreement do not substitute for generation quality.

The short rollout grid runs all primary arms without forced output tokens.
Its 16-token cap only measures short-run fidelity.
The separate downstream pilot permits EOS and declared next-question stops.
It preserves token IDs, parsed answers, raw text, and truncation flags.

Eight questions per benchmark cannot certify a 1 percentage point quality margin.
The final score tables must retain that limitation even when point estimates match.
A weak or truncated baseline cannot establish candidate quality preservation.
