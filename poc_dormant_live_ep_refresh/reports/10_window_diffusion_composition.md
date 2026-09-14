# Window-Diffusion composition analysis

This composition was not benchmarked as an integrated model because the local
Window-Diffusion checkpoint was unavailable and neither component crossed its
independent implementation gate.

The mechanisms are conceptually separable:

- Window-Diffusion removes/prunes far-field token computation and periodically
  refreshes buffer KV/state.
- The proposed candidate would keep attention/shared work fresh but reduce only
  routed-MoE refreshes for still-MASK rows.

In practice the overlap is large.  Rows pruned from the transformer by a
Window-Diffusion-style method cannot also yield routed-MoE savings here.  Only
buffer rows that remain physically executed but have routed outputs safely
reusable are residual.  Our finding that DORMANT routed outputs are less stable
than ACTIVE outputs, plus the sub-1% route-safe oracle, gives no evidence that
this residual is material.

No Window-Diffusion speedup is attributed to this PoC and no composition gain
is claimed.
