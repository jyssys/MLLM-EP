# Exact dependency contract

Task `F[r,j]` executes all 48 physical layers for loop r and contiguous causal chunk j.
It includes the native pass-boundary normalization and residual reset.

Each task needs two predecessor states:

- `F[r−1,j]` supplies the same chunk's preceding-depth hidden state.
- `F[r,j−1]` supplies earlier positions' same-depth KV in all 48 layers.

Transitivity supplies every earlier chunk. Within each chunk, causal masking preserves token order.
RoPE uses original global positions. KV index remains `48*r + physical_layer`.
Weights are immutable. Read-only sharing adds no new causal edge.

Flattening `(r,layer)` to `48*r+layer` gives an ordinary tied-unrolled 192-layer Transformer.
The layer-level hidden and prefix-KV edges are identical under that mapping.
Mapping four contiguous 48-layer groups to four GPUs therefore gives generic chunked PP.
Relabeling the stages as recurrent depths does not change the legal schedules.

The operator-level check separates KV production, contiguous KV publication, and layer output.
It also retains every native pass-boundary normalization as an explicit operator.
The relabeled edge set exactly matches an independently constructed generic tied-unrolled graph.
The macro task graph is a sufficient whole-stack schedule. It is not the finest possible operator schedule.
Finer layer overlap would also exist in the generic operator graph. It is not a recurrence-only advantage.

Each stage owns its depth-local KV. Prefill requires interstage hidden activations, not remote KV reads.
Any later migration to native TP4 decode needs a separate cache-layout transition.
The schedule cannot overlap unknown future autoregressive tokens without speculation.

The graph verdict is `GENERIC_PP_EQUIVALENT`.
Physical timing can still characterize generic engineering value. It cannot establish a new loop-specific mechanism by itself.
