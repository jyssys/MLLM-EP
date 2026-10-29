# Active session handoff

## Final closure, 2026-10-29

The GPU queue is complete. The final scientific review is complete with explicit limitations.
Read `09_final_summary.md` and `configs/reviewed_decisions.json` before the historical notes below.
A: CONDITIONAL overall. B: CONDITIONAL overall. C: NO_GO_CORRECTNESS overall.
No original-model-equivalent lossless serving GO is accepted.
No inference controller needs a resume. Two optional DP Mixed timelines remain failed and documented.
Owned burners run on physical GPUs 0–7 after the experiment releases its quiet lane.
Future fidelity, capacity, generic-control, or long-context experiments require a new follow-up scope.

## Latest user override

Read `user_scope_override.md` first.
Only Ouro and Huginn remain active. Dory is deferred by the user.
`configs/model_scope.json` controls future job admission.
Do not resume Dory without a new user request.
The original three-model specification and all Dory evidence remain preserved.
Controller PIDs listed below are historical after this scope transition.
Read the current `state/main_campaign.json` and `state/dispatcher.json` instead.
The scope-aware resume entry is `resume_campaign.py --from-phase C`.

The active task is the integrated Dense Looped PD A/B/C PoC.
Read the authoritative `../looped_pd_abc_disaggregation_poc_spec.md` before further work.
Read the reporting skill and its required references too.
Do not resume old MoE, diffusion, or decode Loop PP experiments.

## Live controllers

| Process | PID | Role |
|---|---:|---|
| `run_benchmarks` | 2521142 | Main GPU campaign |
| `analysis_watch` | 2549242 | CPU analysis after the GPU campaign |
| `failed_worker_guard` | 2606599 | Cleanup of verified owned FAILED job sessions only |

These PIDs are historical identifiers. Check `/proc` and the current state files before using them.
`state/dispatcher.json` records current child jobs and pending names.
The main campaign currently measures repeated serving trials.
The controller does not stop when the assistant stops sampling.
Do not start a duplicate controller while it remains active.

## Runtime lock

The private RLT clone is `/home/esjung/external/looped-pd-abc-rlt`.
Its upstream commit is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
The main Python is `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The official HF reference Python is `/home/esjung/.venvs/looped-pd-abc-hf457/bin/python`.
That reference uses Transformers 4.57.6. The main environment uses 5.16.1.

The final runtime patch hash is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
`configs/runtime_freeze_final.json` locks the execution files.
`frozen_runs.py` repeats earlier telemetry epochs under that lock before the C phase.
It retains the original trials and writes a logical-to-actual replacement map.
Do not change execution files again without a concrete failure and explicit requalification.
Analysis and reports can still change.

## Model facts

Ouro uses its pinned Base 1.4B checkpoint at R4.
Huginn uses its pinned 0125 checkpoint at the original default R32, not the earlier R16 configuration.
Dory uses the official BF16 4B checkpoint at R2 with per-loop KV.
The Dory reference source is the official-card-linked vLLM PR head `876edc971507f602b18d1ec523c174a068cc02d7`.
It is not stock vLLM 0.26 support.
The local adapter retains the original sublayers, reinjection, nGPT operations, and sliding attention.

All three models passed 100/100 bounded internal RLT PD/A/B output comparisons.
Initial exact-transfer checks passed 18/18 imports, including warmup.
Tuned-cut qualification added nine passing exact imports.
Each newly selected cut also passed 100/100 bounded outputs.

| Model | Short-prompt A cut | 2048-token A cut |
|---|---:|---:|
| Ouro | 3 | 1 |
| Huginn | 1 | 16 |
| Dory | 1 | 1 |

`configs/chosen_cuts.json` records selection and qualification.
The sweep used a separate output-cap-8 workload.
The repeated main cases use cap 32 and a separate cap-128 regime.

Official-reference sequence agreement is Ouro 88/100, Huginn 89/100, and Dory 80/100.
These are not accuracy scores.
Full GSM8K test and HumanEval accuracy remain NOT_MEASURED.
The natural pilot uses 128 available GSM8K training-validation questions without answer labels.
Its maximum chat-tokenized prompt lengths are Ouro 178, Huginn 164, and Dory 174.

## Additional reference audit

Ouro's checkpoint sets `early_exit_threshold=1.0`.
The official implementation still selects an output state through a BF16 cumulative gate probability.
Source analysis suggests that rounding could select an earlier state despite q=1.
The audit completed. The hypothesis did not explain the observed mismatches.
All four loops execute before that post-hoc selection.

`ouro_gate_audit.py` compares selected diagnostic prompts under q=1 and explicit zero-based `exit_at_step=3`.
If it observes earlier selection, it collects a separate 100-prompt explicit-R4 reference.
It must not overwrite or relabel the original q=1 outputs.
Diagnostic prompt selection includes previous mismatches and cannot estimate population accuracy.
The audit evaluated all 12 known mismatch prompts and four controls.
No default path selected before R4 in these 16 prompts.
Default and explicit-R4 outputs matched on 16/16 prompts.
The original official-reference agreement remains 88/100.
The conditional 100-prompt explicit-R4 expansion did not run because its trigger was false.

## Critical timing correction

Read `timing_correction.md` before interpreting any serving result.
The original recorder performed foreground `fsync` during serving.
An Ouro B run recorded 768 P-side appends with 7167.440053ms cumulative wall time.
These durations are not additive E2E components.
Old timing evidence is invalid for clean method comparisons, including the positive Dory SLO signal.
Do not subtract I/O durations to invent corrected latency.

The new private-environment hook installs a nonblocking background recorder before future worker imports.
`configs/evidence_protocol_epoch.json` records the source hash.
The request-thread regression passed after failing on the old implementation.
Multiprocessing durability passed for 200 parent and 200 child records.
GPU requalification remains pending.
Existing healthy inference processes are not restarted or signaled.

Before C, `async_qualification.py` checks natural outputs and exact state imports.
`clean_cut_sweep.py` first repeats A tuning under the nonblocking recorder.
It preserves the old sweep and writes `configs/clean_chosen_cuts.json` after qualification.
`frozen_runs.py` repeats completed legacy timing cases under the locked protocol.
`clean_calibration.py` then runs 27 isolated cost-calibration jobs.
The model/RLT runtime remains unchanged.
Do not edit `telemetry.py` without registering another protocol epoch and repeating qualification.

## Campaign order

The main controller executes A cut sweep, repeated serving, C, scale, and short profiles.
The C manifest generator first performs these prerequisites:

1. Repeat A cut tuning without foreground I/O, then qualify the new protocol on all three models and transfer tracks.
2. Run explicit generic-coalesced Ouro A controls and the reference-gate audit.
3. Confirm earlier telemetry epochs under the final runtime lock.
4. Refresh the frozen service-cost calibration from 27 separate nonblocking isolated pilots.

The main C comparison then uses three starts of P2:D2 and strong four-GPU controls.
The cost model includes prefill/decode batch effects and uncovered handoff.
Each policy saves its actual initialization-time cost coefficients in its run directory.
The loop feature uses CUDA-completed frontiers, not submitted future work.
The placement score is a work heuristic, not a validated ETA or queue simulator.

Scale includes repeated 2/4/8GPU DP, Mixed DP, and selected static ratios.
It also includes one-start A/B/C scale discovery and C32 natural-prompt pressure cases.
One-start results cannot support accepted gain intervals.
Promising points require frozen repetition.
P6:D2 and P7:D1 at eight GPUs are not measured in the initial matrix.
Do not call those ratios empirically dominated.

## Failures and process cleanup

An evidence-statistics hook briefly omitted an `os` import.
It failed Huginn A/B at P128/C1 and DP at P512/C4 in start zero.
The import is fixed. A CPU regression test covers it.
One failed DP process retained a child GPU context for 1491.796 seconds after its failed status.
The verified failed session alone received SIGTERM. The main controller continued.
This interval is not productive GPU compute or measured latency.

The failed-worker guard now checks only FAILED statuses and verified owned sessions.
It never signals a healthy job or a burner.
Raw failure logs and the cleanup record remain under `logs/` and `archives/`.
Retry status must be checked before declaring those configurations complete.

Two extended CPU tests initially failed because instrumentation added statistics keys to the upstream API.
The extended schema is now opt-in through `ABC_RUN_DIR`.
The corrected extended suite passed 97 tests and deselected 86 GPU cases.
The PoC-specific suite passed 11 tests.
Do not erase the earlier failed test log.

## Evidence interpretation

The persistent two-process NIXL probe passed 40 byte checks.
Its 64MiB median was 0.227632ms, approximately 294.813GB/s.
The earlier single-process 188.238ms result is not a valid hardware bandwidth floor.
Neither probe is serving E2E.

The phase table partitions host coordinator time and also includes the harness residual.
It is not a pure GPU attention/MLP breakdown.
Transfer-observation durations can overlap and cannot be added to E2E.
NVML memory peaks are sampled at 0.2 seconds.
PyTorch allocator peaks include warmup and remain separate.
Evidence-I/O time is a CPU measurement, not an additive request component.

The original short profiles lack every matched staged control.
Add matched profiles if a positive claim needs them.
The scale manifest now queues 15 additional matched long-prompt profiles.
They use P2048/C8, eight requests, and output cap eight.
Each model compares Mixed DP2, overlapped staged PD, whole-prompt staged PD, A, and B.
Their latency remains profiled diagnostic evidence, not clean serving data.
C2 migration, dynamic-role hybrid PD, and a validated wall-time oracle remain unimplemented.

## Git and final work

The branch is `flashvep/looped-pd-abc-disaggregation-poc`.
The latest intermediate pushed commit is `2b31a247799d1f289c387cd2de4ce9c824ea0608`.
The remote SHA matched that commit.
This is not the final experiment commit.
Later code, logs, and measurements still require a scoped commit and push.

The outside-index hash is `c162445edb25097abbf18627e41ba8dffc2a15f9a35e6d11385b37a9eb449a22`.
Preserve all unrelated staged and dirty changes.
Use quiet scoped Git commands. Raw logs contain preserved whitespace and must not be reformatted.

After the GPU campaign, the analysis watcher generates comparisons, profile tables, and draft final-file names.
Those drafts do not approve a verdict.
Review all three models, scales, output parity, and uncertainty before writing `configs/reviewed_decisions.json`.
Then regenerate the reports, complete any justified confirmation, and commit only the two owned PoC directories.
Verify the pushed remote SHA.
Finally restore owned burners on every idle GPU with an explicit 0–7 mask.

## Latest resume notes

The controller reached the cap-128 serving regime after completing the main cap-32 matrix.
The remaining C prerequisites include the corrected 42-job cut sweep, recorder qualification, canonical replay, and 27 calibration jobs.
The scale generator also includes 15 matched long-prompt timeline jobs.
The latest analysis/control CPU suite passed 25 tests.

The first corrected Dory P2048/C8 start does not reproduce the old B goodput advantage.
Matched G1 PD qualifies 24/24 requests at the frozen 100ms ITL cell.
B qualifies 17/24. Both methods match all 24 output sequences.
Their p95 E2E values are 2995.06ms and 3820.73ms, respectively.
This is one start only. No final gain interval or architecture verdict follows from it.
The raw names end in `_P2048_C8_s2`.

Long startup gaps can contain active FlashAttention compilation.
Check child CPU activity and log changes before declaring a hang.
The observed Dory compilation workers used approximately 180% CPU each while GPU utilization was zero.
That observation does not classify every later idle interval.

The user received an estimated additional 12–18 hours for the full remaining matrix and review.
The estimate assumed no new correctness or resource failures.
It was not a completion promise.
The main controller and its GPU workers were not stopped for that status response.
