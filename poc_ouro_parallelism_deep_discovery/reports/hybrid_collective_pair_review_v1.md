# Hybrid collective pair: bounded prelaunch review

## Blockers first

No source blocker remains after four remedies.
Runtime graph capture and every-rank output checks remain mandatory.
This is a CPU/source review, not GPU correctness or a launch authorization.

The [structured record](../artifacts/hybrid_collective_pair_review_v1.json) preserves initial and current hashes.

| Initial finding | Current remedy | Review result |
|---|---|---|
| The mask check admitted other P2 subsets and reordered P4 masks. | `check_mask` requires ordered GPUs 4/5 or 4/5/6/7. | Resolved. |
| Setup preceded the cleanup protection. | Setup now enters the protected block. Close and group destruction have separate cleanup attempts. | Resolved for ordinary local exceptions. |
| The script pinned only the deprecated CUDA allocator alias. | Both allocator aliases now specify `expandable_segments:False` before any Torch import. | Resolved by source checks. |
| A mid-cell failure discarded completed local timing rows. | Each rank now retains available CPU samples, phase, failure, and cleanup records. | Resolved by source checks. |

The initial script hash is `3dfdc35b3698c3e78e72176e96e9d9f200b57d2bdbbcd4bab1d1e53c0c5d2d08`.
Main made the remedies.
This reviewer edited no script or package.

## Capture, groups, and primitive selection

The [probe](../scripts/hybrid_collective_pair_probe.py#L61) creates Gloo WORLD and a separate NCCL group.
Gloo handles control traffic and CustomAllreduce metadata.
NCCL handles stock All-Gather and Reduce-Scatter.

The installed CustomAllreduce constructor requires a non-NCCL metadata group.
Its capture context registers graph buffers after the CUDA capture.
The probe places `ar.capture()` outside `torch.cuda.graph`.
It replays only after registration.

The captured native calls use the registered custom path.
A disabled communicator or either None return fails the probe.
The direct class call does not establish the complete native TP dispatch path.

The installed stock wrappers preserve collective ordering through their default synchronous work interface.
Reduce-Scatter defaults to SUM.
Actual NCCL capture support and stream completion remain runtime gates.

Both allocator variables now specify the same value before the first Torch import.
The installed generic allocator header gives `PYTORCH_ALLOC_CONF` higher priority.
A previously imported Torch module now causes rejection.

## Outputs, padding, and lifetime

Four current CPU unit tests pass.
A separate integer model checks all 12 frozen P/C combinations.
The model includes equal-count padding and rank order.

The first and second All-Reduce outputs are 3 and 6 at P2.
They are 10 and 40 at P4.
These integer values are exact in BF16.

All-Gather retains padded zero rows.
Each rank receives the expected complete row order.
SUM Reduce-Scatter returns its owned rows multiplied by P.

All frozen AR payloads have byte counts divisible by 16.
The largest is 524288 bytes.
Every payload is below the class's default 8 MiB threshold.

The source retains the input tensors, graphs, and final outputs throughout each cell.
The inputs remain unchanged during replay.
The graphs do not share an explicit allocation pool.
A completed cell does not replay after the next shape or teardown.

These are source-level lifetime checks.
They are not measured pointer identity or GPU graph evidence.

## Timing and cleanup

Each event interval contains one graph replay with 16 complete serialized pairs.
The summary selects each trial's maximum rank-local duration.
It does not add independently timed component maxima.
It does not claim globally aligned cross-GPU timestamps or global makespan.

Host time remains separate.
All 31 trials retain alternating arm order.
Warmup, capture, correctness checks, and control barriers remain outside the event interval.

A fake setup-failure test checks failure-record retention and process-group destruction.
The test uses fake objects only.
It executes no CUDA or real Torch call.
An AST check finds no CUDA query in failure logging.

The first fake harness omitted its fake distributed-module attribute.
The corrected harness passes.
That failure concerned the review harness, not the probe.

Rank-0 success appears before final cleanup.
Thus, admission must also require successful outer-guard completion and error-free rank-local cleanup records.
A local cleanup attempt does not prove recovery from fatal CUDA errors or peer failures.

The script requires a guarded campaign output path.
The launcher must still supply a fresh directory and the frozen physical mask.
The output-path check alone is not an independent process-ownership proof.

## Claim boundary

The [contract](hybrid_collective_pair_contract_v1.md) describes communication service only.
It distinguishes this probe from the old remote-full-MLP/NCCL component.
Its scope is appropriate.

The pair arms do not implement identical complete mathematical layers.
They omit model operators, packing, gate decisions, KV ownership, and row scheduling.
Their tensor capacities are not measured wire bytes or HBM traffic.

A faster stock pair exposes a component opportunity only.
A slower stock pair concerns the named implementation and shape.
Neither result alone establishes serving speedup, recurrence-specific novelty, or a universal rejection of hybrids.

No model weights, actual outcomes, protected files, or GPUs entered this review.

## Actual work

The actual interval is 2026-10-19 18:20:06 UTC through 2026-10-19 18:25:41 UTC.
The active interval through this checkpoint is 335 seconds.
No prior pause interval enters this work.
Concurrent worker time does not add campaign elapsed time.

