# Acceptance and stop gates

## Status discipline

Each future phase reports PASS, CONDITIONAL, NO_GO, BLOCKED, or NOT_RUN.
Only the source/planning work in Phase 0 is complete now.
No code-audit result qualifies GPU performance, numerical accuracy, or race safety.

The numerical and performance limits below are proposed registrations for review before implementation.
Do not modify a failed limit after observing held-out candidate results.
Historical frozen limits remain immutable.

## Global correctness requirements

1. Pin base model, tokenizer, parameters, BF16, and R4 semantics.
2. Confirm 24 physical layers, 96 logical KV namespaces, owner-local tensors, and actual rank participation.
3. Keep RoPE positions, causal masks, residual order, loop norm, and sampling unchanged.
4. Compare natural outputs independently from replay outputs.
5. Preserve all failed probes and locate the first divergence.
6. Require race-free page publication, cancellation, preemption, and graph-buffer reuse.

The old P512 custom TP hidden relative L2 `0.469359` exceeded `0.03`. That result remains FAIL.
The old logit and KV criteria also remain in the historical report.
Passing task accuracy does not erase those numerical failures.
Floating-point sensitivity is not evidence of a sharding bug by itself.

## New numerical registration

Before candidate tuning, use eight disjoint calibration requests across P128/P256/P512/P2048.
Archive raw hidden, logits, and KV differences for same-backend TP1/TP4 and repeated TP1 runs.
Normalize relative L2 with an explicit denominator floor. Record that floor before evaluation.
Separate per-layer maxima, final outputs, and token-position quantiles.

For each tensor class, propose a tolerance from the calibration envelope and a stated numerical rationale.
Freeze the numeric tolerance file before the 24 held-out natural probes.
An independent review must approve the values. This plan does not invent a successful tolerance.
If baseline numerical variation is unexplained, mark numerical qualification unresolved.

Placement-only PP should preserve native operations closely. Require exact greedy output, exit-depth, EOS, and NFE parity on bounded same-backend probes.
Any discrete mismatch triggers investigation. Do not automatically excuse it through a wider hidden tolerance.
TP4 may remain an explicitly exploratory accuracy lane while numerical attribution continues.
It must not serve as the only qualified reference for an architectural GO.

For pure KV repair, copied BF16 row values must match their source exactly.
Historical target rows must remain unchanged.
Current-row readiness must precede every dependent attention read.
If a layout conversion changes values, treat it as a correctness failure, not a performance trade-off.

## Quality and confidence requirements

Report correct/N, accuracy%, and paired bootstrap 95% CIs for GSM8K and MATH separately.
Use at least 5,000 request-level paired bootstrap draws with a fixed seed.
Report EOS, truncation, parsing failures, and baseline-correct/candidate-wrong examples.
Do not compare architectures using different problem selections or post-hoc prompt changes.

Proposed practical regression margins are 1.0pp on GSM8K and 2.0pp on MATH-500.
Use the same-mode Single reference for placement and repair changes.
For a quality noninferiority claim, the paired CI's lower bound must exceed the negative margin.
If the 300/100 cohort lacks power, report CONDITIONAL and expand toward the full 1,319/500 suite.
Zero observed loss on the pilot is a smoke result, not proof of noninferiority.

Dynamic versus Static accuracy measures the exit policy's effect.
Dynamic PP versus Single Dynamic measures the new execution mechanism's effect.
Do not stop all architecture discovery solely because the unchanged Adaptive baseline scores below Static.
Do stop a new mechanism that creates unexplained or meaningful additional quality loss.

## Phase gates

| Gate | PASS or continuation condition | NO_GO / BLOCKED condition |
| --- | --- | --- |
| G0: bounded scope | Native-kernel linear PP path plus a bounded cyclic reentry proof are plausible | Near-total scheduler or serving rewrite. Narrow scope before implementation |
| G1: native baseline | Stable natural TP1/TP4 runs, correct ownership, valid timing, and numerical attribution | Wrong model, wrong cache, hangs, wrong GPU, or unexplained suspected operator bug |
| G2: Static PP | Repeatable natural-serving benefit or measured useful capacity benefit against strong controls | Replay-only advantage, graph-policy artifact, no competitive serving region, or failed cyclic scope proof |
| G3: Dynamic Single/TP | Original gate, real skipped loops, exact row repair, coherent TP survivor ordering | Post-hoc exits, all-loop replay, divergent collectives, or invalid history |
| G4: Dynamic PP | Bounded natural parity, correct coda routing, measured repair and imbalance | New correctness failure or measured overhead with no feasible recovery |
| G5: async repair | Legal GPU overlap, reduced dependent wait, and repeatable client improvement | Nonblocking API only, early read, unsafe buffer reuse, or clean benefit within noise |
| G6: main evaluation | Qualified same-engine quality, latency/goodput, capacity, and uncertainty | Missing controls or exploratory-only evidence. Do not issue a broad GO |

### G2 numeric continuation rule

Pre-register two prompt regimes and C1/C4/C8 before PP results.
Use at least three independent server starts and ten paired cohort repetitions for key cells.
Require one of the following:

- At least 7% lower E2E or TPOT against the strongest qualified same-P baseline in two pre-registered cells.
- At least 10% higher SLO goodput at a frozen SLO, with repeatable open-loop evidence.
- A measured safe capacity frontier that supports an otherwise infeasible target request at acceptable latency and quality.

For latency improvement, the paired 95% CI must exclude no improvement.
For goodput, report interval estimates across runs and arrival traces.
A C1/C4 regression above 5% requires an explicit throughput-only claim and a scope review.
Do not hide low-concurrency harm behind a high-concurrency average.
These limits guide continuation. They are not measured achievements.

If Loop PP equals generic contiguous logical PP, report that equivalence.
Do not advance solely to produce a loop-specific label.
A clearly valuable generic deployment result may continue only after a separate scope decision.

### G4 economic stop rule

Measure synchronous repair's critical-path wait, not only its bytes or service-time sum.
Estimate the maximum removable wait under the legal DAG.
If that bound cannot recover the observed loss against the strongest same-resource baseline, stop Dynamic Loop PP optimization.
Do not implement refill or aliasing merely to rescue an uneconomic copy path.

### G5 numeric continuation rule

After race and parity tests, require a repeatable clean improvement, with a paired 95% CI excluding no improvement.
Prefer at least 5% lower TPOT/E2E or at least 10% higher SLO goodput in the pre-registered exposed regime.
Compare sync and async within the same barrier and admission mode.
Report copy/compute contention, extra memory, preparation, and graph effects.
If benefit falls below these practical targets, report CONDITIONAL rather than a broad architecture GO.

## Timing validity gates

- Heavy and clean runs must use the same request IDs and policy.
- Report output/depth/work parity and observer overhead.
- GPU timing must include source packing, P2P, scatter, waits, and graph staging where relevant.
- Client E2E must include normal graph preparation, queueing, output processing, and transport.
- The strongest native control must keep its available optimized kernels and reductions.
- A timeline must distinguish producer completion, publication, and consumer readiness.
- Do not count one overlapped interval as a saving in multiple stages.
- Insufficient samples for tails must remain exploratory.

Failing these gates produces TIMING_UNRESOLVED or BLOCKED evidence, not a speedup claim.

## Runtime safety and resource gates

Use only physical GPUs4–7. Resolve UUIDs at launch.
Stop only recognized task burners when a future approved GPU test starts.
Do not stop unrelated processes, even on an allowed GPU.
Reserve at least 10% driver-visible HBM headroom for initial capacity probes.
Freeze the final safety reserve from independent calibration and report actual bytes.
Abort unsafe capacity sweeps rather than causing repeated OOM or disrupting other jobs.

## Final interpretation

GO means a qualified practical improvement against strong controls, with a clearly stated novelty scope.
CONDITIONAL means some evidence supports continuation, but a named gate remains unresolved.
NO_GO means the evaluated premise failed under the measured scope.
BLOCKED means a required supported path or correctness condition could not be established.
This planning package concludes CONDITIONAL feasibility, not an experimental GO.
