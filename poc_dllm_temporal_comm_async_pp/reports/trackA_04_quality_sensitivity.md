# Track A4 — quality sensitivity gate

No full-trajectory compressed-dispatch quality run was performed.  This is a
deliberate gate outcome, not missing evidence: the best zero-cost gross request
oracle is only 0.72%, far below the required 8% threshold for modifying the
expert path.

The local codecs establish only numerical feasibility:

- FP8 reconstructed-hidden relative L2: about 0.27%.
- Row-scaled INT8 reconstructed-hidden relative L2: about 0.062%.

These values are not a trajectory-safety claim.  No credit is taken for expert
output, logits, acceptance, NFE, sequence, GSM8K, or HumanEval quality.  The
unmodified clean baseline remains the correctness reference.
