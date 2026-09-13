# Fine-grained dependency DAG

## Current execution path

Source inspection and runtime validation show this exact order:

```text
local hidden
  -> router GEMM -> grouped top-k
  -> DeepEP dispatch completion
  -> owner-rank fused routed expert completion
  -> DeepEP reverse combine completion
  -> rank-local shared expert
  -> routed + shared add
  -> TP all-gather
  -> next dense layer
```

The coarse serialization is real: dInfer waits for dispatch before owner expert,
waits for the complete fused expert output before combine, and computes the
shared expert only after routed combine. DeepEP nevertheless exposes an
asynchronous completion event, so independent work can legally run before
`current_stream_wait()`.

## Pair classification

| pair | dependency | reason | outcome |
|---|---|---|---|
| router -> dispatch | strict | exact top-k and layout are inputs | confirmed |
| remote dispatch -> owner expert | strict | remote hidden rows must arrive | confirmed |
| remote dispatch / local source expert | independent after routing | local rows and weights are already known | live tested |
| routed EP / shared expert | independent until the final add | both consume the same local hidden | live tested |
| dispatch wave i+1 / expert wave i | independent complete waves | distinct input and handle | live tested |
| expert wave i / combine wave i-1 | independent complete waves | distinct output and handle | live tested |
| completed expert tile / same-wave combine | partially dependent | legal per ready tile, but current fused API exposes only full completion | prior-art collision |
| request A EP comm / request B attention or expert | independent | distinct requests and tensors | live tested |
| routed+shared sum -> TP gather -> next layer | strict under current dense layout | next layer needs exact replicated hidden | confirmed |

The important distinction is that algebraic independence is necessary but not
sufficient: remote dispatch/local expert is legal yet slower concurrently.

Machine-readable evidence is in `DEPENDENCY_DAG.csv`. Source locations are
`modeling_llada2_moe_sglang.py:704` for the sparse path and
`exact_overlap_probe.py` for diagnostic replay.
