# Libra bounded execution reproduction contract

**SUPERSEDED / PAUSED:** User supplied supplementary source at ~16:00 KST.
Do not execute the paper-only probe before comparison with the actual code.
The limitations below describe the pre-supplement fallback, not current access.

The official SNU-ARC/Libra repository returns HTTP 404. A paper-derived port is
therefore necessary, but it is not evidence that the published system fails.

## Preserved

- Actual next-layer router evaluated on current normalized MoE input.
- Original, exact routing determines expert execution; prediction never changes
  token routing probabilities or substitutes an expert.
- Eight additional experts per rank, half reserved for local-demand replication.
- Local work before remote work; remote-only adaptive sharding (Appendix C).
- AllGather token dispatch, independent of the remote sharding decision.
- Real Qwen BF16 expert weights, copy-engine peer transfers, grouped Triton MoE.

## Explicitly unresolved or changed

- EP4/H100/Qwen30B rather than original EP8/H200/Qwen235B or GLM355B.
- Initial remote ownership and ties are underspecified. Use the linear expert
  home and stable integer-ID ties. Primary planning follows Appendix B literally;
  a stronger sharding-between-replication interpretation is a sensitivity only.
- Python planning is NOT the unavailable Cython planner. Its measured host time
  cannot establish an algorithmic overhead failure.
- Isolated replay first establishes exact work conservation, BF16 output parity,
  actual local/remote compute, AllGather and replica-copy costs. It does not count
  as request-level serving speedup or full overlap reproduction.
- A perfect-prediction placement comparison tests only the cost of the diagnosed
  prediction error; it does not grant free replication to an E2E successor oracle.

## Order of evidence

1. Routing/planning invariants and no-approximation output equivalence.
2. Four-GPU real-input isolated compute/communication/copy measurements.
3. Measured gap between actual-prediction and ground-truth-prediction placement.
4. Request-level critical-path upper bound / online port if the gap is material.

No successor implementation or candidate ranking before all three screening
milestones have comparable, honestly labeled evidence.
