# Bottleneck Crossover Study: Reproduction

## Source and preservation

Branch: `flashvep/looped-pd-bottleneck-crossover-abc`.
Base commit: `d2050debde78c33c4a77fce5fe3ef342fc5f3694`.
The private RLT HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The initial private diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

The new package is `poc_flashvep/looped_pd_bottleneck_crossover/`.
The evidence root is `poc_flashvep/reports/looped_pd_bottleneck_crossover/`.
Protected earlier source and report hashes remain in `configs/initial_contract.json`.
The index guard preserves 122,717 unrelated entries.
Each dispatcher checks these records before starting a worker.
Existing staged changes and old raw data are outside this study's commit scope.

Every run contains `source.zip` and its configuration reference.
The archive includes the private engine diff.
`commands.jsonl` records the exact command, GPU mask, source archive hash, and parent Git SHA.
Use the archive for a specific run if later study code changed.
Do not assume the checkout HEAD alone identifies an earlier uncommitted worker.

## Environment

Python: `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The environment resolves the existing vLLM 0.26 stack.
Recorded versions include Python 3.14.6, PyTorch 2.11 with CUDA 12.9, and NIXL 1.4.1.
The native AR control and RLT use different attention paths.
The complete software and GPU audit remains in `configs/` and `state/initial_gpu.json`.

Eight physical H100 80GB GPUs are available.
Each request uses one model replica per worker.
The primary PD layout is 1P+1D, not TP or PP.
Clean timing leases all eight GPUs and runs one isolated lane.
No managed burner runs during those measurements.
This policy avoids uncalibrated fabric and host interference.

## Checkpoint pins

| Model | Revision |
|---|---|
| Ouro Base | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` |
| Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

Do not change precision, weights, recurrence, or the qualified numerical fixes.
Huginn retains its 4,096-token context limit.
All configured Prompt and output pairs fit that limit.

## Data and SLOs

`data/source_prompts.json` contains the frozen ShareGPT source records.
The workload uses explicit tokenizer prefixes.
It is `LENGTH_MATCHED_CONTROLLED_SHAREGPT_PREFIX`, not an Azure text trace.
`data/test64_*.json` extends the test set without calibration overlap.
Distinct source records can still have identical truncated prefixes.
See `configs/calibration_test_prefix_overlap_audit.json` for the exact duplicate counts.

The serial calibration uses eight separate calibration prompts per model.
`configs/frozen_SLO.json` contains the Ouro and Huginn thresholds.
`configs/frozen_SLO_qwen.json` contains the native AR thresholds.
The primary criterion combines TTFT and request-average TPOT.
Token-level ITL sensitivity is separate.
Never replace a frozen threshold after reading candidate results.

## Commands

Run commands from `/home/esjung/MLLM-EP-github`.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.bootstrap
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.looped_pd_bottleneck_crossover.test_contracts -v
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.campaign --plan /home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_pd_bottleneck_crossover/configs/phase_b_screen_plan.json
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.summary_tables
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.classify
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.profile_summary
```

The queue skips completed namespaces.
It preserves failed namespaces instead of replacing their evidence.
Use a new configuration name for a retry.
Do not start a second clean dispatcher while another lane owns the GPUs.
The campaign records its original start time and remaining budget.
Resuming the queue does not reset the twelve-hour boundary.

The calibration and configuration generators intentionally reject existing output files.
Do not rerun them to overwrite frozen settings.
Use existing generated configurations for reproduction.

## Output interpretation

`capacity_summary.csv` reports finite independent service cohorts.
`serving_runs.csv` reports arrival-to-drain Serving results.
`bottleneck_screening.csv` reports finite pressure categories.
`profiled_phase_kernel_busy.csv` reports CUPTI activity in separately profiled ranges.
`e2e_component_shares_aggregate.csv` reports exclusive host intervals as percentages of summed request time.
These percentages are not exclusive GPU computation or DMA shares.
Do not add nested CPU spans or overlapping device durations.

`dispatcher_result.json` records the worker return code.
`status.json` records the namespace status.
A zero return code alone is not a correctness qualification.
Read the associated gate result and raw comparison evidence.

## Resource restoration

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.resources status
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.looped_pd_bottleneck_crossover.resources restore
```

Restore burners only after all clean GPU measurements end.
The resource helper checks ownership and active GPU processes.
It must not terminate unrelated jobs.
The final report must distinguish a requested restore from verified running burners.
