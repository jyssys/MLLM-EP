# Acceptance gates

## Evidence classes

| Class | Meaning |
|---|---|
| MEASURED | Actual GPU execution or recorded host timing |
| TRACE_REPLAY | A counterfactual uses a frozen real trace |
| MODEL_PREDICTION | A cost model estimates a future cost |
| ORACLE | An analysis removes a declared cost or uses future information |
| NOT_MEASURED | The experiment did not obtain the value |

Structural slot counts are accounting, not measured speedup.
Microbenchmark bandwidth is not serving handoff bandwidth.
A source-derived independence graph is not evidence of achieved overlap.

## Fidelity

The current internal RLT pilots require exact output IDs on 100 prompts per model and control pair.
They also require exact valid KV and hidden-buffer transfer checks.
The tests use greedy generation with EOS and a declared cap.
They do not establish complete-answer accuracy.

The original-reference pilots are a separate gate.
Their current agreement is Ouro 88/100, Huginn 89/100, and Dory 80/100.
Those values cannot establish official benchmark quality or bit-exact original-reference equivalence.
No timing win alone removes that limitation.

The additional Ouro audit checks a specific source-derived hypothesis.
BF16 cumulative gate probabilities may reach one before the final state.
The hypothesis is not a measured checkpoint result until the audit completes.
The audit compares the original q=1 path with explicit zero-based `exit_at_step=3`.
Both paths still evaluate all four loops. This is not actual early compute skipping.
The experiment preserves the original q=1 reference and records any new protocol separately.

A CPU counterexample shows that this selection rule can choose an earlier state in BF16.
Four synthetic gate logits of 2.2 produce a third-step cumulative probability of 1.0 in BF16.
The same FP32 calculation remains below 1.0 until the fourth step.
This example does not use checkpoint-generated gate values.
It does not establish the frequency or cause of any actual reference mismatch.
The queued model audit must supply that evidence separately.

That audit completed on 16 prompts, including all 12 known mismatches.
It observed no early post-hoc selection on their default paths.
The explicit-last-state and default outputs matched on 16/16.
The gate-rounding hypothesis therefore does not explain those observed mismatch cases.
The original 88/100 reference agreement remains unresolved.

Batched trials also require per-request output and length comparisons.
An earlier B1 match cannot conceal a later concurrency failure.
Internal parity and official-reference fidelity remain distinct even when both have 100 samples.

## Frozen runtime

`configs/runtime_freeze_final.json` identifies the final execution and telemetry sources.
Each actual attempt contains its own source archive and RLT patch.
The initial measurements include telemetry development epochs.
`frozen_runs.py` repeats those completed trials under the final runtime before the C comparison.
The original trials remain in the master table.
`configs/frozen_serving_replacements.json` identifies the primary replacement runs.

This replacement rule depends on source hashes, not observed speed or output quality.
It does not remove a slow trial because its result is inconvenient.
Accepted comparisons require at least three matched frozen starts.
Any new source change after this freeze requires another explicit qualification decision.

## Track A

A must preserve the complete recurrent snapshot and continue without prefill restart.
It must compare against decode-local prefill, ordinary PD, and the strongest equal-budget colocated and static-ratio controls.
Its matched source control uses the same whole-prompt chunk, staged quantum, and generic descriptor optimization.
Ouro has an additional explicit coalesced A control.

The target is at least 10% lower p95 E2E or 10% higher SLO goodput.
An individual request retains its sequential recurrence dependency.
A queue benefit must not become a claim of less prescribed model computation.
Generic logical-depth migration remains a novelty counterexample.

## Track B

B must preserve compute ownership, depth, chunk size, and scheduling quantum in its matched comparison.
Upstream overlapped transfer remains enabled in the main control.
The control also receives the same generic descriptor coalescing where needed.

The detailed specification permits a p95 TTFT or SLO-goodput target of at least 10%.
The report also evaluates p95 E2E and TPOT penalties.
A TTFT gain with an unacceptable decode penalty is not GO.
The strongest equal-budget colocated result remains visible.

The implementation currently waits for complete transfer before D admission.
It does not implement every consumer-deadline streaming alternative.
A tail-only transfer-removal estimate is not the strongest possible distributed-attention or streaming oracle.
Generic immutable layerwise transfer remains a novelty counterexample.

## Track C

C uses conventional full-prefill ownership and upstream transfer.
A and B remain disabled.
The primary ablation is calibrated service-time placement versus loop-state placement.
Both receive the same legal completed-frontier telemetry and cost calibration.

The target is at least 10% lower p95 E2E or 10% higher SLO goodput.
C must beat the strongest measured queue, token, service, static-ratio, and colocated controls.
The loop-state ablation must explain the additional benefit.
A one-prefill/one-decode pool has no initial routing freedom.
The experiment does not invent a placement gain there.

The current score is a pending-work heuristic, not a validated completion forecast.
Its live comparison does not validate a global queue simulator automatically.
C2 migration and dynamic-role hybrid control remain separate unimplemented capabilities.

A positive C result also requires a stronger generic completed-phase control.
That control can use the existing full-prefill completion notification without any recurrent frontier.
The current service model subtracts elapsed service once per worker queue.
It does not remove every completed request's compute term separately.
Loop-frontier information could otherwise receive credit for ordinary phase-completion information.
If C shows a gain, that competing explanation must be tested before GO.
No additional runtime is required if C already loses to the existing controls.

## Scale and scope

Repeated baseline scaling uses 2/4/8 GPUs.
Track-specific scaling and high-pressure natural workloads begin as one-start discovery.
Those points cannot support an accepted speedup confidence interval.
Promising points require independent frozen repetition before GO.

A negative result applies to the tested exact runtime and workload.
It does not prove that every possible PD implementation lacks headroom.
The report must state missing ratios, unimplemented controls, reference mismatches, and quality limits.
Each track receives an independent decision. One failure must not cancel another track.
