# Exact state and ownership dependencies

The diagrams describe legal execution. They do not claim measured speedup.

## A: one prefill migration

```mermaid
flowchart LR
  I[Prompt IDs and positions] --> P[P: prelude and loops 1..r]
  P --> S[Completed loop event]
  S --> K[Valid KV planes and page mapping]
  S --> H[Full prompt recurrent carrier]
  S --> R[Request RNG state and loop frontier]
  K --> T[NIXL transfer and final commit]
  H --> T
  R --> T
  T --> D[D: restore and loops r+1..R]
  D --> C[D: coda and sample]
  C --> N[D: next token with full fixed depth]
```

The prompt's next loop depends on the preceding loop's completed hidden state.
Causal attention at depth `d` reads the same depth's valid prefix KV.
The destination cannot replace that prefix with another depth's KV.
Huginn and Dory also require their fixed input-injection carrier.
Huginn additionally requires its request RNG generator state.

The source retains registered buffers and cache leases until transfer completion.
The destination restores valid page frontiers before it schedules remaining work.
The adapter forbids prefix-cache migration and unfinished-token-chunk migration.
These restrictions are explicit implementation limits.

## B: unchanged compute ownership

```mermaid
flowchart LR
  P1[P: prompt depth d] --> E[CUDA completion event]
  E --> X[NIXL write of immutable depth-d range]
  P1 --> P2[P: later prompt depths or chunks]
  X --> A[D: transfer completion]
  P2 --> F[Final prompt state and commit]
  F --> A
  A --> D[D: coda and full-depth decode]
```

The transfer and later prompt computation may overlap.
This diagram proves independence, not achieved overlap duration.
The same source bytes must remain immutable until the copy finishes.
The last-loop shared layout does not satisfy that condition for intermediate values.

## C: legal information boundary

```mermaid
flowchart LR
  W[GPU-completed frontier event] --> Q[Observed worker state]
  L[Known prompt length and output cap] --> M[Frozen calibrated cost model]
  Q --> M
  M --> P[Initial P/D worker choice]
  P --> B[Conventional full-prefill PD]
```

The policy does not use future token IDs, EOS positions, or future GPU durations.
Loop progress is a state feature, not a dynamic-depth prediction.
The standalone C comparison does not enable A or B.
