# Experiment Log

- 2026-09-12 23:42 KST: created the isolated worktree and copied the complete working specification.
- 23:47: verified the trace additions with a submitted-batch-4 smoke test on physical GPUs 0--3.
- 23:49--00:18: ran the clean submitted-batch-32 sweep for mini 1/2/4/8/16/32. Every point completed at least three independent process restarts.
- 00:18--00:23: extended mini16 and mini32 to five restarts because these were the only competitive static settings.
- 00:23--00:35: captured one observer-heavy trace for every mini size, including per-wave denoising state and representative-layer route/timing data. A mini32 launch failed before model execution with a TCPStore nonce error; it was preserved and excluded, then rerun successfully.
- After GPU work: regenerated all CPU analyses, validated 2,203 logical waves and 6,609 representative-layer rows, restarted the requested repository-owned burn on physical GPUs 0--3, and did not launch further task experiments.

The raw logs are ignored from Git but retained locally. Clean outputs, analysis tables,
scripts, figures, and the trace patch are committed for reproduction.
