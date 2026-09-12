# Experiment log

All times are KST on 2026-09-13. Every GPU launch used `CUDA_VISIBLE_DEVICES=0,1,2,3`.

- 02:19--02:29: six independent clean mini32 runs, three GSM8K and three HumanEval.
- 02:29--02:39: first shape/stage/temporal probes. These established the trace schema but some preceded the discovery-worktree `PYTHONPATH` fix and are excluded where fields were incomplete.
- 02:40--02:41: a GSM8K shape run hit an empty-live confidence-quantile instrumentation bug. It was observer-only, produced no performance result, and left task-owned ranks waiting; only those identified task processes were stopped. The partial artifact is retained and excluded.
- 02:43--02:45: corrected GSM8K shape trace (`shape/gsm8k/r3`).
- 02:45--02:47: corrected GSM8K stage trace (`stage/gsm8k/r2`).
- 02:47--02:50: corrected GSM8K temporal trace (`temporal/gsm8k/r2`).
- 02:51--02:53: corrected HumanEval shape trace (`shape/humaneval/r2`).
- 02:53--02:55: corrected HumanEval stage trace (`stage/humaneval/r2`).
- 02:55--02:57: corrected HumanEval temporal trace (`temporal/humaneval/r1`).
- After 02:57: no further task measurement. The task-owned GPU0--3 burn was restarted while CPU analysis proceeded.

Instrumentation correction: the editable environment originally resolved the older dInfer worktree. The task scripts now prepend `/home/esjung/external/dinfer-llada2-flash-discovery/python` to `PYTHONPATH`. No old or failed trace is silently mixed into the final analysis; exact selected paths are hard-coded in `scripts/analyze_discovery.py`.

