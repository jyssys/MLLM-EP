# Layer scope and overlap oracles

Vanilla batch-one refinement has a strict dependency chain from the current
decoder logits to the next accepted position. No next-iteration exact work is
independent before that decision. This report therefore counts overlap only
when a timeline exposes real independent slack; moving a wait or assuming
parallel execution of dependent stages is not a saving.


The scaled upper bound from making every refinement dispatch and combine free
is 6.23% of clean request wall in the observer-heavy trace. This is an
unattainable cap, not an overlap opportunity. The source dependency remains:
current global MoE output feeds later layers and final logits; logits select the
accepted position; only then is the next refinement state known. No independent
batch-one exact slack was exposed.

Layer-wise state reuse is also unsafe as a generic inference: at lag 1 the
combined MoE update has only 0.829 cosine and 49.1% relative L2, despite 82.2%
routed-branch persistence. Removing all layer-global work has a large arithmetic
cap, but no correctness-preserving oracle and direct collision with Epoch/DICE.

**Decision: KILL overlap (<5% feasible) and KILL layer-gated reuse (no
quality-safe exact contract).**
