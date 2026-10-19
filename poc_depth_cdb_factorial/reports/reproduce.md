# Reproduction

## Existing environment

Start from parent commit `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950` and the published experiment files.
Use the pinned environment from `poc_decode_first_depth_aware_dp_v3/reports/reproduce.md`.
The existing model cache and CDB checkout must match `environment_and_reuse.md`.
No GPU outside physical 4/5/6/7 is permitted.

Run commands from the new worktree root.
`artifacts/benchmark_commands.sh` contains the full ordered command list.
The runtime passes an explicit one-GPU mask to each independent worker.
The outer ownership guard also receives the exact physical GPU set.

## Fresh experiment execution

The freeze commands create new files and refuse to replace an existing frozen manifest.
Use a new experiment directory for a fresh campaign.
Do not delete or reset the published raw evidence to rerun a command.
For continuation, `campaign.py` skips only completed successful run directories.
It refuses to reuse an incomplete output directory silently.
Preserve a failed directory and create a versioned replacement when a repair is necessary.

The execution order is:

1. Freeze the arm manifest, speed cohort, quality cohort, seeds, and placement rule.
2. Build the execution plans and all-C trace plans.
3. Record environment metadata.
4. Run the five-arm heavy functional probe.
5. Run the three initial paired speed repeats.
6. Run A0 on all 256 quality questions.
7. Freeze quality hindsight placements from A0's actual depth work, without grading answers.
8. Run the other four quality configurations on the same questions.
9. Run the trace lane and the matched historical-placement sensitivity.
10. Run the frozen two-repeat extension for every arm and C.
11. Analyze quality, latency, interaction, parity, and observer effects.
12. Compress raw traces losslessly and run the final audit.

The `speed`, `quality`, `trace`, `sensitivity`, and `extra` phases select campaign stages.
All five arms run even when measured gains are negative.
There is no gain-based termination branch or RL training step.

## Analysis of the published evidence

The published raw rank traces and controller case logs use `.json.gz` files.
The originals remain on the research server.
`contract.read` accepts either representation.
`raw_compression_manifest.json` records original sizes, hashes, and compressed paths.
`rank_timeline.jsonl.gz` and `rank_timeline_summary.csv.gz` preserve the detailed trace timeline.

Use the following analysis modules with the reference Python environment:

```bash
python -m poc_depth_cdb_factorial.scripts.analyze --quality
python -m poc_depth_cdb_factorial.scripts.trace_analysis
python -m poc_depth_cdb_factorial.scripts.audit_v3
python -m poc_depth_cdb_factorial.scripts.figures
python -m unittest -v poc_depth_cdb_factorial.tests.test_contract
```

The final ownership audit requires access to the original server and current burner processes.
It is not a portable scientific test for a different host.
Its role is to confirm scope preservation and final GPU cleanup on this server.
