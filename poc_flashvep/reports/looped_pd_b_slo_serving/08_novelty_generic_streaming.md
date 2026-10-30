# Generic layerwise streaming and novelty control

Status: structural audit complete. A separate GenericLayer serving runtime is NOT_MEASURED.
No loop-exclusive novelty GO is permitted.

B changes transfer readiness inside a token chunk, not computation ownership or total KV bytes.
The existing PD_G1 control can transfer a full-depth chunk while the next chunk computes.
B can transfer a completed immutable depth plane while later loops of the same chunk compute.
The consumer still waits for all required KV and the final hidden/commit state.

Ouro's physical layers and fixed recurrence can be mapped to an unrolled logical-layer index.
Huginn adds prelude and coda logical layers to its repeated core.
Under the qualified depth-indexed cache, each valid logical layer has an immutable prompt-prefix frontier.
A generic immutable-layer streaming algorithm can consume those same frontiers.
If all layers in one loop share one completion event, the generic algorithm can coalesce that event's ready descriptors.
Its dependency graph then can match B at the engine's exposed completion granularity.

This is a structural novelty counterexample, not a measured equivalence of two independent serving engines.
The pinned runner does not expose a ready-made, independently qualified per-physical-layer streaming connector.
Adding callbacks inside every physical layer would change the readiness granularity and profiling cost.
That would require another correctness and fairness qualification.
The follow-up therefore does not label a B alias or ordinary PD_G1 as a new GenericLayer baseline.
The new worker explicitly rejects an unimplemented GenericLayer live label.

The performance comparison against PD_G1 remains meaningful even without a novelty GO.
The final conclusion must distinguish `NOVELTY_UNRESOLVED` from a measured system-level capacity result.
Source evidence: frozen `frontier.py`, `boundary.py`, RLT `pd/connector.py`, `pd/transport.py`, and the model runner's completion events.

Prior systems establish generic phase separation and inter-GPU state transfer.
Read [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) and [Splitwise](https://arxiv.org/abs/2311.18677).
[DuetServe](https://proceedings.mlr.press/v306/gao26m.html) is a strong generic interference-control reference, not this PoC's implementation.
None of those paper speedups are imported as measurements for Ouro or Huginn.
The generic layerwise counterexample above is our structural analysis. It is not an independently timed prior implementation.
