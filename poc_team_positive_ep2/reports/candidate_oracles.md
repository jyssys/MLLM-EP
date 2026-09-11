# Candidate oracles from measured TEAM residuals

## Gate result

No independent serious candidate survived.  One independent perfect oracle is
above 5%, but it is only 6.28%, has no demonstrated feasible path, and overlaps
existing speculative dLLM control.  It does not justify EP4 validation.

| Candidate | Evidence | Source / upper bound | Request E2E | Decision |
|---|---|---:|---:|---|
| Existing fused expert packing | **Measured** trivial fix | 4,413.5 ms vs Python TEAM | 56.94% | Reject as novelty |
| Perfect early speculative-branch commitment | Perfect future oracle | 209.7 ms | **6.28%** | Weak; no follow-up |
| Same-route metadata delta reuse | Generous perfect oracle | 160.5 ms | **4.81%** | Kill (<5%) |
| Eliminate all router/gating | Physically unrealistic upper bound | 323.5 ms | **9.69%** | Semantically invalid / prior-art risk |
| Eliminate all dispatch + combine | Physically unrealistic upper bound | 330.4 ms | **9.90%** | Required transport / generic EP prior art |

## Candidate 1 — early speculative-branch commitment

**Measured source.** TEAM executes 528 fused decoder-layer calls at M=128,
corresponding to four speculative branches, versus 192 M=32 calls.  M=128 costs
4.267 ms and M=32 3.869 ms per layer.

**Oracle.** Knowing the winning branch before each layer and charging every
M=128 call at the M=32 mean saves 209.7 ms / 3,337.7 ms = 6.28%.  This is more
optimistic than a realizable policy: it assumes perfect prediction, immediate
commitment, zero verification, and no quality loss.

**Risks.** Correctness risk is high because branch scores become meaningful
only after computation.  TEAM already introduces speculative exploration, and
ODB-dLLM explores jump/share speculative decoding.  The residual is too small
to support an independent paper direction.

## Candidate 2 — same-route delta plan reuse

**Measured source.** 51.00% of assignments reuse the same logical
position/expert route across speculative branches, while route preparation is
9.43% of clean request time.

**Oracle.** Multiplying those values gives a deliberately generous 4.81%
request upper bound.  It assumes every duplicate route removes its proportional
preparation cost with zero bookkeeping.

**Causal falsification.** Exact input equality is only 0.0295%, so neither
dispatch payload nor expert compute can be reused exactly.  Epoch's compiled
block plan and fresh worklist also make routing/metadata reuse a direct
adjacency.  Kill.

## Candidate 3 — router or communication elimination

Completely free router/gating and completely free dispatch+combine are 9.69%
and 9.90% upper bounds.  Neither is a counterfactual executable system:
iteration/branch hidden states change, and remote experts require their inputs
and outputs.  Epoch explicitly recomputes gate logits for live values, while
DeepEP/COMET-class systems already target communication execution.  Their
feasible residual must be materially below the quoted upper bounds.

## Trivial-fix conclusion

The largest measured opportunity—56.94%—was recovered by the existing
`vllm.model_executor.layers.fused_moe.fused_experts` primitive.  It attacks the
reference substrate, not TEAM or dLLM structure.  Once applied, no independent
candidate has a credible >=10% request oracle.

Figures: `plots/13_candidate_oracles.png`, `plots/14_candidate_risk.png`.
