# Reproduce the A1/A2 diagnostic

## Scope

This diagnostic reads archived results. It does not change the serving runtime.
The analysis uses CPUs only. It does not start a CUDA context.

Base commit: `9726dc30729f160f7772ac484bc5ee01d596351c`.
Prior v3 commit: `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950`.
Diagnostic branch: `flashvep/hindsight-depth-diagnostic`.

Run commands from the repository root.
The local worktree is `/home/esjung/MLLM-EP-hindsight-diagnostic`.
The Python interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
Required CPU packages are NumPy, SciPy, and Matplotlib.
No package installation was necessary.

## Commands

1. Reproduce the symptom from the original measurements.

```bash
env CUDA_VISIBLE_DEVICES='' \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  poc_depth_cdb_factorial/diagnostic/reproduce_symptom.py \
  --output /tmp/hindsight_symptom_fresh.json
```

Use a new output path. The script refuses to overwrite its output.
An exit code of 1 means `SYMPTOM_REPRODUCED`.
It tests whether the archived gain reaches 1.05×.
The result is 0.997449×, so this diagnostic deliberately remains red.
This test does not define a significance threshold.

2. Analyze the archived requests, ranks, and batches.

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  poc_depth_cdb_factorial/diagnostic/analyze_diagnostic.py
```

3. Build the Korean report from the derived tables.

```bash
env CUDA_VISIBLE_DEVICES='' \
  /home/esjung/.venvs/llada2-mini-reference/bin/python \
  poc_depth_cdb_factorial/diagnostic/build_report.py
```

4. Test the accounting and replay identities.

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest discover \
  -s poc_depth_cdb_factorial/diagnostic -p 'test_*.py' -v
```

The seven tests check population variance, the real graph bucket at 12 rows, placement capacity, and trace identities.
They also check direct A1/A2 output parity counts and the source hashes.
Synthetic test fixtures are unit checks only. They are not research evidence.

## Inputs

The original `placements.jsonl`, `rank_timeline.jsonl.gz`, and `metrics_raw.csv` exist.
The aggregate timeline is compressed in the published commit.
The detailed rank JSON files retain more fields than that aggregate timeline.
The analysis therefore uses those rank files for the primary calculations.

Clean speed inputs:

```text
poc_depth_cdb_factorial/outputs/speed_DP4_r{0,1,2,3,4}_v1/
  speed_C{8,16,32,64}_A{1,2}_r{repeat}.json.gz
  rank_{0,1,2,3}/speed_C{8,16,32,64}_A{1,2}_r{repeat}.json.gz
```

Trace inputs:

```text
poc_depth_cdb_factorial/outputs/trace_DP4_v1/
  trace_speed_C{8,16,32,64}_A{1,2}_r0.json.gz
  rank_{0,1,2,3}/trace_speed_C{8,16,32,64}_A{1,2}_r0.json.gz
```

Quality inputs:

```text
poc_depth_cdb_factorial/outputs/quality_DP4_v1/
  rank_{0,1,2,3}/quality_A{1,2}_part{0,1,2,3}.json.gz
```

The prior v3 depth trace contains actual token-level records.
Its request summaries supply the historical Hindsight work values.
The new manifest records SHA256 values for every input used by the analysis.
The source audit records the pinned CDB revision and external file hashes.

## Definitions

- **LIVE_MEASURED:** a value from an archived GPU execution. It is not a new execution.
- **TRACE_REPLAY:** a counterfactual calculation that holds recorded tokens and depths fixed.
- **SIMULATED:** a fitted bracket cost or an explicitly stated unit-cost model.
- **NOT_AVAILABLE:** an absent measurement or a counterfactual that was not executed.

Variance uses `ddof=0`. CV is standard deviation divided by mean.
Decode work excludes the first token from full-depth prefill.
The reported speedup is A1 time divided by A2 time.
The timing statistic is the ratio of median cohort makespans.
Timing confidence intervals use 2000 paired whole-cohort resamples.
Depth and accuracy intervals resample requests or paired questions.
Layers, ranks, and tokens are not independent serving samples.

The no-refill replay assumes one resident cohort per rank.
It reproduces all recorded wave counts for C8 through C64.
It does not model live output changes after a new placement.
The 128-position sensitivity selects real recorded windows after observing their depths.
It is a mechanism diagnostic, not a new held-out quality result.

The fitted model predicts recurrent-core trace brackets only.
It excludes prefill, terminal KV copies, and host scheduling.
Its small predicted gains do not establish E2E gains.
The additive-work and unit-call bounds apply only to their stated surrogate models.
Neither bound is an achievable physical oracle.

## Failures and corrections

The first analysis run rejected a false power-of-two bucket assumption.
The assertion failed on the real 12-row graph bucket.
The pinned source and recorded padding values specify `[1,2,4,8,12,16]`.
The updated analysis checks every recorded batch against that list.

An intermediate sensitivity used rank-concatenated output order for its round-robin control.
That order differed from the original manifest order.
The final analysis restores the original manifest order for the natural cohort.
The final replay is deterministic at seed 20261023.
These corrections affect the diagnostic only. They do not change the original experiment.

The diagnosis workflow stops before a runtime fix.
The user requested causal analysis, not scheduler modification.
Thus the symptom test remains red. The accounting tests must pass.

## GPU and preservation policy

No extra inference or P2P benchmark was necessary.
Existing utilization workers on physical GPUs 4,5,6,7 remained active.
No process on GPUs 0,1,2,3 was changed.
The original factorial and v3 worktrees remained unchanged.
Only this diagnostic directory and the new Korean report belong to the new commit.

The original `QUALITY_CONFOUNDED` verdict remains unchanged.
This diagnostic adds a separate `DEPTH_COST_MISMATCH` finding.
