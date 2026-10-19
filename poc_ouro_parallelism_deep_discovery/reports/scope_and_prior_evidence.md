# Scope and verified predecessor evidence

## Current scope

The empirical targets are dense Base Ouro-1.4B and Base Ouro-2.6B.
The campaign considers fixed R4 and the existing original-gate adaptive path separately.
BF16 weights and model execution remain unchanged.
No training, gate replacement, reduced precision, or target-model substitution is permitted.
Only physical GPUs 4/5/6/7 are eligible.
P8 remains an explicitly unmeasured future analysis.

The independent worktree prevents edits to the original staged or dirty files.
`artifacts/preservation_snapshot.json` records the source specification and predecessor Git diffs.
The frozen snapshot is `reports/spec_snapshot.md`.

## Predecessor evidence ledger

The following numbers are previous project measurements, not new measurements from this campaign.
The coordinator read the actual reports and inspected the associated implementation paths.

| Predecessor | Verified local evidence | Finding | Restriction on this search |
|---|---|---|---|
| Predictive asynchronous PP | `poc_async_recurrent_pp/reports/final_report.md`, `scripts/baseline.py`, emulator and quality artifacts | Thinking 1.4B R4 pilot: exact GSM8K 15/16 and MATH 14/16. Stale: 8/16 and 3/16. Delta: 7/16 and 3/16. W0: 6/16 and 1/16 | Do not repeat whole-stage stale/delta substitution without a changed assumption and a new quality gate |
| Memory–compute asymmetry | `poc_recurrent_asymmetry/reports/final_report.md`, `scripts/physical_microbench.py` | Tied weights 2.672249 GiB. B1 depth-KV 0.75 GiB at 1K and 12 GiB at 16K. Tested CP operator was slower | Weight size, KV footprint, HBM traffic, and exposed latency are different quantities |
| Exact wavefront | The same report and dependency checks | The tested causal DAG maps to a generic aliased unroll | A depth/chunk diagram alone supplies no new execution axis |
| Decode-depth DP v3 | `poc_decode_first_depth_aware_dp_v3/reports/final_report.md` and raw timing audit | Historical C64 model predicts 12.770% P95 reduction. Live point estimate is 4.410%, CI [−8.298, 13.341]% | A fitted or selected hindsight schedule is not a latency lower bound or a causal method |
| Completed CDB factorial | Published commit `9726dc30729f160f7772ac484bc5ee01d596351c`, five-arm reports and ownership audit | All 125 speed cohorts and 1280 quality generations completed. Verdict: QUALITY_CONFOUNDED | Do not rerun the separate experiment. Do not assume portable outputs or future depth traces across batch layouts |

The last row differs from the specification's preparation-time description.
The factorial experiment completed before this campaign's recorded start.
Its finished reports and raw evidence are read-only inputs.
This is a verified status update, not an assumed unfinished result.

## Limits of the existing negative evidence

The stale-state pilot rejects the tested predictors, not every possible asynchronous architecture.
The attention result applies to its measured operator, backend, and contexts.
The generic wavefront identity applies to the tested exact dependency graph.
The historical DP result does not solve the optimal placement problem.

The previous reference fidelity remains partial across some backends and batch shapes.
The new campaign must retain that limitation until a matched probe resolves it.
Neither activation similarity nor a small smoke establishes reasoning-quality preservation.

## Resource interpretation

The final factorial ownership record identifies utilization burners on physical GPUs 4/5/6/7.
The initial live GPU inventory matches those context PIDs and UUIDs.
No new research GPU job has started at this audit point.
Before a probe, the coordinator must verify both process ancestry and device ownership again.
Only a verified owned burner may yield its device under the user's explicit burner-management instruction.
An experiment process or a foreign process must remain untouched.

The shared burner script defaults to another GPU set.
Any scoped restoration must set both `CUDA_VISIBLE_DEVICES` and `VLLM_UTILIZE_GPUS` explicitly.
This campaign must not launch a duplicate burner on an occupied device.
