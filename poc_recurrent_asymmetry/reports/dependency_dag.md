# Exact dependency DAG

## Audited operations

The model executes a recurrent 24-layer core four times.
Each physical layer contains attention, MLP, normalization, and residual operations.
The terminal normalization occurs after every 24-layer pass.
There is no independent prelude or coda transformer block.
Embedding precedes the recurrent core, and the LM head follows the selected exact R4 output.

For a prompt chunk, define:

- `Q(r,l,c)`: prepare the current layer's normalized input and Q/K/V.
- `K(r,l,c)`: publish the current depth/layer KV for that chunk.
- `O(r,l,c)`: complete attention, residual operations, and MLP.
- `N(r,c)`: apply the pass-terminal normalization after layer 23.

`Q` is a dependency abstraction, not a claim that one existing fused kernel exposes all these boundaries.
The actual runtime currently publishes full operation outputs.

```text
Previous layer O ──> Q(r,l,c) ──> K(r,l,c) ──> O(r,l,c)
                                      ^
                                      |
                              K(r,l,c-1), when c > 0

O(r,23,c) ──> N(r,c) ──> Q(r+1,0,c)
```

The K publication chain preserves past-position KV at the same logical depth.
It does not require the earlier chunk's MLP to finish before attention can use its published KV.
The coarse scheduling model uses the stronger full-task boundary for safety.
It does not claim to expose partial buffers from an unfinished kernel.

## Legal concurrency

Prompt embeddings are known before prefill starts.
The first-layer QKV preparation can therefore become ready for several prompt chunks.
After the first chunk publishes KV, later chunks can use it at the same depth and layer.
A later depth still requires that chunk's completed earlier-depth hidden state.
No task can substitute a previous loop's stale state for its required input.

For autoregressive decode, the next token's embedding depends on the current token's prediction.
The fixed teacher inputs in hardware trials do not authorize parallel future-token generation.
This study restricts the wavefront model to known causal prefill chunks.

## Generic unrolled control

Map `(r,l)` to logical depth `d = 24*r + l`.
Retain separate KV slots and a terminal normalization after each group of 24 layers.
The mapped edge sets exactly match an ordinary causal unrolled transformer graph.
The checks cover 1, 2, 4, 8, 16, 32, and 64 chunks.
They retain shared-parameter storage as a separate ownership constraint.

This identity establishes generic causal-pipeline equivalence for the tested task graph.
It does not establish that every future recurrent architecture is equivalent.
Weight reuse could affect a different mechanism, but this study did not demonstrate such a mechanism.

## Tests and limitations

`tests/test_dag.py` checks the following conditions:

1. Recurrent and generic unrolled edge sets match.
2. Ready sets include only tasks with completed dependencies.
3. An illegal future-KV order fails.
4. A two-chunk, two-loop, two-layer float64 toy matches serial causal attention.
5. Each scheduled compute task executes once without overlapping compute on one GPU.

The toy tests check graph logic, not pretrained BF16 model accuracy.
Full-model serial chunking fails its frozen numerical gate for smaller chunk shapes.
That separate result prevents a physical pipeline correctness claim.
No new engine or hidden-state predictor changes that failed qualification in this PoC.
