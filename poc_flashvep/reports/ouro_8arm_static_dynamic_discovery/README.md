# Ouro eight-arm discovery

This directory contains an active experiment, not a final positive architecture result.
Read `status.md`, `analysis.md`, `benchmark_summary.md`, and `correctness.md` first.
`reproduce.md` gives interpreter paths, GPU rules, staged commands, and payload exclusions.

The user-approved evaluation target is 400 questions per arm: 300 GSM8K and 100 MATH.
The study preserves earlier pilot results and samples only the additional question IDs.
The original 1819-question manifest remains unchanged.

`state/continuation_controller.json` identifies the background experiment controller.
`state/continuation_stage.json` identifies its current stage after the Static pilot exits.
`state/continuation_error.json`, if present, preserves an execution failure.
Do not run another campaign on the same GPUs while this controller owns them.

The controller restores authorized burners between experiments and after its GPU stages.
It uses physical GPUs4,5,6,7 only. Single uses physical GPU4.
It refuses unrelated GPU jobs.

Generated report snapshots remain IN_PROGRESS until the final review.
Missing cells remain NOT_RUN or BLOCKED. Historical strict TP numerical failures remain visible.
