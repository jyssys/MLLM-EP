# Correctness summary

The original model weights, BF16 precision, RoPE, and Static R4 remain unchanged.
Native generation asserts exit depth four for every returned token.
Complete GPU windows contain 96 attention executions.
The cache manager uses four distinct depth planes.
Initialized fixture hashes preserve the original BF16 K/V values.

The exact layout probe passes twelve sampled physical-layer/depth comparisons.
Its Q, K, V, and attention outputs are byte-exact.
The two-token output and depth arrays also match.
The full timing cohorts retain output tokens for paired analysis.
The cohort comparison must not replace full-answer accuracy evaluation.

The distributed operator has 4464 retained paired rows.
Its maximum relative L2 is 0.0034308868926018476.
Its maximum absolute error is 0.0078125.
`distributed_attention_correctness.md` states the batch and mask limitations.
The edge results remain in their own archived rank records.

No integrated distributed natural generation was performed.
GSM8K/MATH-500 accuracy is `NOT_RUN`.
There is no claim of statistical quality equivalence.

The retained layout cohorts match tokens in 90/90 request pairs.
These are bounded hardware outputs with the same static depth, not benchmark answers.
