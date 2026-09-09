# Tensor Layout Contract

## Ring/pass-KV CP

Each rank retains its local sequence/query shard with full hidden width after attention. Remote K/V blocks circulate, but there is no mandatory post-attention full-hidden gather. A one-collective-fusion claim therefore does not apply. The only relevant counterfactual is exact query-block readiness followed by block-wise Router/EP execution.

## Ulysses/A2A CP

For CP degree `P`, the relevant layout is:

```text
sequence shard × full QKV heads
  -> sequence-to-head all-to-all
full sequence × head shard
  -> attention
full sequence × attention-head shard
  -> return all-to-all
sequence shard × full attention hidden
  -> dense o_proj + residual + RMSNorm
  -> router/top-k
  -> EP dispatch
```

The replay implements this contract for `P=4`.

## Exact direct-transport algebra

Qwen's dense output projection mixes every input attention head into every output hidden coordinate. Before the Ulysses return, rank `p` can form only an additive partial output:

```text
y = sum_p y_p,  where y_p = x_p W_o,p
```

Residual addition and RMSNorm require the exact `y`. Router logits and exact top-k then require the exact normalized full-hidden vector. Expert destination ranks are therefore unknown until the exact cross-rank assembly/reduction has completed.

The alternatives do not remove structural communication:

1. Return/gather head shards to the token owner, then project/norm/route and dispatch: the baseline.
2. Row-shard `o_proj`, materialize `H` partials, and reduce: still communicates an exact `N×H` result before routing and creates larger partial buffers.
3. Feature-shard the exact result, add distributed norm/router reductions, then send all `H/P` shards to every selected expert owner: retains the CP reduction and the full EP hidden transport, plus metadata/reduction/assembly overhead.
4. Send shards before route identity is known: must broadcast toward possible owners and increases traffic.

## Byte lower bound

For BF16, hidden size 2048, CP4:

- Ulysses return network bytes/token: `2048 × 2 × 3/4 = 3,072 B`.
- Measured mean remote EP destinations/token: approximately 2.68–2.69.
- EP dispatch network bytes/token: approximately 10.995–11.011 KB.
- Baseline exact lower bound: approximately 14.07–14.08 KB/token.
- Candidate exact lower bound: the same 14.07–14.08 KB/token, before new metadata/reductions.
- Exact structural bytes saved: 0%.

The tempting 11.96–17.19% layer-time oracle obtained by deleting both return A2A and dispatch is algebraically invalid because it assumes route identity before exact hidden-state construction. The valid exact direct-transport oracle is 0%.

## Decision

`ALGEBRAICALLY_BLOCKED_ROUTE_DEPENDENCY`. No direct-transport implementation was justified by the analytical gate.
