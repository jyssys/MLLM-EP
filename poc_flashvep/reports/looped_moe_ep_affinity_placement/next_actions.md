# Resume instructions

## Current work

Read `research_state.json` before any launch.
Inspect the exact worker command lines and GPU UUIDs.
Do not launch a second dispatcher while the existing dispatcher owns devices.
The durable dispatcher runs `python -m poc_flashvep.looped_moe_affinity.full_campaign`.
Its current log is `runs/logs/full_campaign_v2.log`.

The dispatcher performs these remaining GPU stages:

1. Preserve the completed 90-cell TRUE EP performance matrix.
2. Finish official IFM benchmark anchors on EP4 and EP8.
3. Evaluate full available Foil anchors on EP4 and EP8.
4. Evaluate the official Foil held-out corpus on native, EP4, and EP8.

The separate `post_campaign` watcher waits for these jobs.
It then runs matched native hardware measurements and generic controls in independent processes.
Its resumed log is `runs/logs/post_campaign_v4.log`.
The first native attempt stopped at allocator fragmentation, after fourteen complete cells.
`native_oom_diagnosis.md` records the actual failing and passing regressions.
The resume recovers those fourteen cells into a new attempt without changing the original files.
The final generic ablation removes redundant count collectives using current replicated router outputs.
It preserves real expert sharding, dispatch, compute, combine, and restoration.
Its GPU results remain pending until those jobs complete.

The `attribution_campaign` watcher waits for `post_state.json` to report `GPU_CONTROLS_COMPLETE`.
It then measures complete E2E percentage breakdowns requested by the user.
Its current log is `runs/logs/attribution_campaign_v6.log`.
Do not launch these GPU stages manually while their watchers remain active.

The analysis watcher runs `python -m poc_flashvep.looped_moe_affinity.finalize --wait --publish`.
It waits for all three GPU campaigns.
It then checks result counts, writes reports, preserves compact evidence, and tests scoped publication.
Its resumed log is `runs/logs/finalization_v2.log`.
`release_state.json` is the local proof of a real commit and verified remote SHA.
It is not present before a real release.
`COMMITTED_PUSH_PENDING` does not mean a successful push.

Every attempt has an immutable run identifier.
Completed attempts have `complete.json`.
The dispatcher skips those completed attempts during resume.
An interrupted attempt requires a new identifier or an explicit per-item official-cache resume procedure.
Do not delete failed or incomplete attempts.

## CPU analysis

The route corpus has completed on all three native models.
The held-out split uses different prompts from calibration.
Run the following commands after the current analysis process finishes:

```bash
cd /home/esjung/MLLM-EP-github
python -m poc_flashvep.looped_moe_affinity.affinity
python -m poc_flashvep.looped_moe_affinity.placement
python -m poc_flashvep.looped_moe_affinity.aggregate_quality
python -m poc_flashvep.looped_moe_affinity.aggregate_timing
python -m poc_flashvep.looped_moe_affinity.aggregate_generic
python -m poc_flashvep.looped_moe_affinity.paired_quality
python -m poc_flashvep.looped_moe_affinity.aggregate_attribution
python -m unittest poc_flashvep.looped_moe_affinity.test_cpu -v
```

The analytical placement code does not measure a new placement runtime.
The current-top1 source control is generic, not loop-aware.
Do not call its gain recurrent novelty.
The bridge restores replicated dense states after each MoE invocation.
An inter-loop routing transition therefore does not automatically represent a communication edge.

## Whole-E2E percentages

Read `e2e_attribution_contract.md` before interpreting the component percentages.
The pilot uses complete 32-output-token rollouts, not extrapolated single-forward spans.
Fine and coarse markers must preserve the uninstrumented greedy token IDs.
The timing gate checks observer overhead and event-to-host alignment.
Only a passing pilot expands to the full workload matrix.
The report retains failed observer pilots with explicit warnings.

The attribution campaign refreshes `e2e_breakdown.md` and the percentage CSVs after each completed run.
`PENDING_MEASUREMENT` means that no whole-E2E attribution run completed yet.
Do not replace pending values with earlier kernel service-time sums.
The phase CSV separates the first completed output from the remaining output tokens.
The full breakdown uses one endpoint timeline and sums to 100% without overlapping intervals.
Communication spans include rank waiting. Expert spans can include launch gaps.
These spans are not pure wire time or pure GEMM service time.

## Remaining release work

Add paired native-versus-EP task comparisons and sequence-level NLL intervals.
Complete the constrained oracle analysis, figures, and required reports.
Audit all three models, both EP degrees, all five batch sizes, and all three required prefix lengths.
Mark unavailable, failed, or unsupported metrics explicitly.
Preserve the original staged tree outside this study.
Commit only this study's code, reports, and specification.
Push the branch and compare the remote SHA with the local commit.
Restore owned burners on every unused authorized GPU.
The finalizer performs these release steps only after its completeness and test gates pass.
Do not run a second release process while that watcher remains active.
