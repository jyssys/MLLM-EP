# Completion-aware implementation and limits

## Source scope

The optional implementation is `poc_flashvep/completion_aware_loop_pd/`.
It reuses the qualified Native-path connector, packed exact KV descriptors, NIXL transport, and Native ACK.
It does not change the external runtime checkout or model weights.
Per-run archives identify the actual source version.
The feature-disabled arm remains Native Atomic PD.

## Candidate coverage

The source separates checkpoint eligibility from D admission.
`source_monitor_boundaries` permits later boundary offers after a transient rejection.
The source cannot recover a boundary inside an atomic graph that already completed.
Checkpoint-capable execution can cost more than an unmodified atomic graph.
The primary comparison includes that cost.

## Ownership protocol

The source offers a GPU-complete boundary with an epoch and expiry.
The destination verifies its receiving state, KV reservation, deadline budget, and continuation occupancy.
The destination prepares a lease but does not execute the suffix yet.
The source requests commit with the same epoch.
The destination recomputes completion feasibility before commit.
Only a matching committed reply lets the source export that frontier.

The source retains its KV and hidden buffers until Native ACK and transport drain.
The destination executes the committed suffix exactly once.
A stale preparation permits source continuation before commit.
An uncertain commit does not permit duplicate source execution.
The implementation does not provide D-to-P recovery after commit.

## Completion model

The model prices legal token widths, recurrent steps, terminal work, handoff, and already accepted work.
It inserts a Decode reserve between predicted future opportunities.
It does not pack a fraction of an indivisible quantum.
The model records a predicted finish and a completion deadline.
The executor prioritizes eligible continuations by deadline.

These records are empirical reservations, not guaranteed future GPU slots.
Unknown future arrivals can invalidate a forecast.
The primary handoff estimate also understates some observed handoff intervals.
A reservation miss stops new admission.
It does not discard committed work or release live state.

The primary fix separates admission suspension from execution permission.
Previously accepted work can still execute within a live safe budget.
This removes one cause of persistent starvation.
It does not prove starvation-free scheduling under every arrival trace.

## Execution budget

The executor verifies a quantum before each new submission.
It does not preempt a running CUDA kernel.
Decode-ready work retains its existing priority and deadline reserve.
The primary model uses individual CUDA Event costs and a boundary guard.
The recovery model also includes complete Decode-cycle enclosures and their host gaps.
Neither cost is an exclusive kernel-union measurement.

The recovery diagnostic separately limits the next cumulative request-average TPOT.
It uses only the first output time and the number of delivered tokens.
It does not assume a future output length.
The same execution-budget rules apply to Token and Loop.

## Recovery diagnostics

The secondary Token baseline chooses a prefix from the current advertisement.
The destination still revalidates the offer at the actual boundary.
After rejection, the source computes the whole remaining suffix.
It does not repeatedly create small source chunks after that rejection.
This policy differs from the primary repeated-boundary Token arm.

All secondary arms prohibit cold packed-prefill graph capture during measurement.
They use a cached graph or the qualified eager path.
The graph audit distinguishes those choices.
The policy does not claim control of every CUDA graph in the runtime.

The secondary model includes the future Decode work of known accepted continuations.
A past tail suspension can recover only after active Decode and committed continuation work both disappear.
A completion-model failure does not permit that recovery.
The frozen absolute SLOs remain unchanged.
Native-relative non-inferiority remains a separate result.

## KV limitation

The original Huginn D pool contains 23552 blocks.
A 2048/16 R32 request reserves 4128 blocks.
The pool therefore holds five requests, despite eight logical slots.
The Native coordinator reserves destination KV before source Prefill begins.
Those receiving reservations can prevent P ingress while D has no Decode-ready request.

The secondary 24832-block pool holds six requests.
Every secondary Huginn arm uses that same pool.
Startup statistics require at least 3 GiB of free HBM.
This diagnostic does not implement late destination reservation or compact unused KV planes.
Any benefit from the larger pool belongs to the Native control too.

## Correctness scope

CPU tests cover stale epochs, repeated decisions, quantum packing, deadlines, and lossless evidence recovery.
N8 and N32 diagnostics compare exact output, valid KV planes, hidden state, recurrent work, and wire bytes.
The comparison reuses an identified frozen Native state reference.
The report does not describe that reference as a new Native GPU run.

Marker-free N128 checks compare exact output and full work.
They also check Native lifetime and credit release.
N128 does not hash every intermediate state during clean timing.
The corresponding N32 diagnostic qualifies that state path separately.

## Measurement limits

Host submission and completion records are not GPU kernel intervals.
CUDA Event enclosures include waits and submission gaps.
CUPTI kernel unions describe the instrumented execution only.
A failed 5% interference gate blocks a clean Native capacity claim.
Missing physical measurements do not prove absent headroom.

## Novelty limit

An equivalent logical-layer cut can reproduce these recurrent boundaries and transfers.
Boundary naming alone does not establish Loop-specific novelty.
A fair Token comparison must also exclude asymmetric graph and scheduling costs.
The final verdict must distinguish generic offloading from a Loop-specific benefit.
