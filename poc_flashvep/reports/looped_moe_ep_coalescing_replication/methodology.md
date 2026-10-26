# Measurement and decision contract

## Native and EP paths

The experiment uses the three pinned checkpoints from the previous study.
The official IFM loader remains unchanged.
Foil uses its official remote code and Foil-1 subfolder.
The thin bridge replaces only the routed expert expression.
It preserves attention, normalization, residuals, gate scores, and recurrent execution.
No expert output is skipped or reused.

Each process sees one physical GPU as cuda:0.
Each EP group has four or eight real NCCL ranks.
The non-expert model components remain replicated.
Source token rows use strided ownership.
Expert weights have unique owners before replication controls.
Unequal owner counts handle IFM's 25 experts without adding logical experts.

The execution sequence is:

1. Compute the unchanged native router.
2. Partition source rows and exchange actual count metadata.
3. Dispatch BF16 expert rows with NCCL All-to-All.
4. Run only owner-local, weight-identical expert GEMMs.
5. Return those outputs with NCCL All-to-All.
6. Apply the original top-2 weights and restore source rows with All-Reduce.

Source restoration is not free.
It remains in all full-model timings.
This replicated-dense research baseline is not optimized DeepEP serving.
DeepEP is absent from the validated PyTorch 2.11 environment.
The experiment does not present NCCL control overhead as a new loop-specific bottleneck.

## Fidelity

The same FA4 backend runs on both sides of every IFM native-versus-EP comparison.
FA3 equivalence remains outside the established baseline.
The previous FA4-versus-explicit-attention numerical limitation remains unchanged.

The distributed gate freezes these numerical limits before the first EP run:

- Relative logit L2: at most 0.02.
- Maximum absolute error divided by maximum reference magnitude: at most 0.04.
- Top-1 agreement: at least 0.99.
- Absolute shifted-NLL difference: at most 0.02 nats/token.
- Eight-token greedy smoke output: exact agreement.

The original limits are not changed after inspection.
Engineering quality checks cover 32 examples per domain where completed.
The domains are language, GSM8K, and MATH-500.
These teacher-forced checks are not downstream answer accuracy evaluations.
Long-shape checks cover B1/B2/B4 at 128/1024/2048 tokens.
Failures remain separate from completed runtime measurements.

Huginn's official evaluation initializer includes a batch-index term.
Native B4 therefore does not define the same initial state as four independent B1 calls.
The experiment does not change that published initializer.
Native-versus-EP fidelity always compares identical batch geometry.
The independent-request observer instead compares against sequential B1 reference calls.
Its wall time must not be compared with native B4 as an identical-output serving benchmark.

## Timing

Clean timing encloses the complete native model forward and CUDA completion.
A barrier precedes the timed interval.
The maximum rank-local elapsed duration defines the coordinated cohort duration.
No unsynchronized absolute GPU timestamps are compared across ranks.
Each primary cell uses ten warmups and twenty timed repetitions.
Prefix 128 receives three independent process starts.
Those starts use prompt-cohort offsets 0/4/8.
The confidence interval combines restart and prompt-cohort variation.
It does not isolate GPU-group variance.
Corresponding EP4 and EP8 starts use matching prompt IDs.
Long-prefix cells receive one start, so their repetition intervals are exploratory.

Generation uses 32 and 128 greedy output tokens with a fixed hardware cap.
EOS is disabled only for this hardware matrix.
Those runs are not natural-generation accuracy results.
Both native runtime families recompute the full prefix without a KV cache.
No reported TPOT is cached-decode TPOT.

CUDA-event component spans come from separate observer runs.
They include API control work and host launch gaps.
The selected recurrent capture site also clones diagnostic expert inputs.
Its expert API span includes that copy.
The report selects one cumulative-critical rank for a forward.
It does not add unrelated component maxima into a 100% E2E partition.
Observer overhead and unresolved host/rank waiting remain explicit.
Heavy profiler wall time never becomes headline latency.

## Readiness and Track A

The synchronized cohort baseline keeps requests at the same logical frontier.
That baseline cannot alone establish asynchronous cross-pass opportunity.
A separate bounded P1 observer runs independent native request threads.
It does not coalesce expert work.
Only the coordinator issues EP collectives, in one globally consistent order.
CUDA events protect input readiness and output buffer lifetime.
The observer logs actual device-ready items before each scheduling choice.
Its extra control broadcast prevents a clean-serving performance claim.

The scheduling model maintains every request's layer and recurrent dependencies.
It includes the measured non-expert work between expert calls.
It never releases a request's future recurrent pass early.
Wait windows include 0/50/100/250/500 microseconds and larger sensitivity values.
The best generic same-pass window is a required comparison.
A gain over a deliberately delayed generic window is not sufficient.

The simulated scheduler is a bounded policy search, not an exact global optimum.
The prefill model is not a full generated-request E2E speedup measurement.
Only a validated actual implementation can establish measured acceleration.

## Replication and Track B

Replicas are immutable BF16 copies allocated before a measurement control.
The allocation control verifies copied tensor hashes across ranks.
It measures retained memory and one-time initialization cost.
It does not claim that the candidate replica-aware dispatcher ran.

Placement calibration uses eight engineering examples per domain.
The remaining examples and separate hardware cohorts do not tune replica sets.
Generic hot-expert, generic cost-aware, and pass-conditioned placements share copy budgets.
Current exact routing remains authoritative in every assignment model.
The static-placement search is greedy, not an optimal perfect oracle.
Row-proportional expert-time estimates are optimistic.
They do not remove collective launch latency or assume free replica selection.

Pass identity and current exact routing are different information sources.
With a fixed resident set, a generic selector can use the same current demand counts.
The same-resident-set ablation tests that distinction.
Generic replication benefits do not establish loop-specific novelty.

## Decision discipline

O0 is an unconstrained analytical diagnostic.
O1 preserves modeled readiness and memory constraints.
O2 uses a causal policy with measured costs and reported calibration error.
M denotes an actual distributed implementation.
Operator microbenchmarks are not request-level M speedups.

GO requires at least 1.15× overhead-constrained E2E headroom against a strong matched baseline.
It also requires a material incremental benefit from recurrence information.
The minimum heuristic increment is 5% beyond generic controls.
An incomplete or unvalidated model receives CONDITIONAL or BLOCKED, not invented performance.
