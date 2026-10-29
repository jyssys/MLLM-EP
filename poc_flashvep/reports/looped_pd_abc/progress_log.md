# Progress

## User scope change: 2026-10-29

The user restricted future work to Ouro and Huginn in vLLM-RLT.
Dory is deferred. Existing Dory evidence and the original specification remain unchanged.
The controller update sent signals only to verified owned scheduling and analysis controllers.
The current inference job completed normally without a signal.
Owned burners occupied all idle GPUs during the controller update.

The scope and regression suite passed 31 CPU tests.
The locked model execution files and private RLT patch remain unchanged.
The scope-aware controller resumed the existing canonical confirmations.
Its first jobs are Ouro B and Ouro A on physical GPUs 6 and 7.
The active manifest contains 128 cases: 66 Ouro and 62 Huginn.
Completed cases are skipped. The remaining workload is not restarted from zero.
The A/B/C tracks and 2/4/8GPU budgets remain in scope.

The experiment read the complete authoritative specification.
The experiment preserved the existing index and created a separate branch.
The experiment downloaded the pinned official Dory BF16 checkpoint.
Ouro native generation, Huginn native generation, Ouro NIXL PD, and NIXL byte copies completed.

Huginn PD has an explicit upstream rejection.
Dory requires distinct sublayer indexing, sliding attention, nGPT normalization, and input reinjection.
Neither model has a qualified PD adapter yet.

## Qualification update

All three models now have native RLT, full-prefill PD, recurrence-split PD, and frontier-handoff implementations.
Each model passed 100/100 bounded internal outputs for PD and A.
Each B implementation matched its staged control on 100/100 bounded outputs.
All 18 heavy KV/hidden transfer checks passed byte identity.
These counts include warmup imports. They are not accuracy scores.

The official-reference comparison completed on 100 prompts per model.
Sequence agreement is Ouro 88/100, Huginn 89/100, and Dory 80/100.
The report retains this unresolved reference-fidelity limitation.

The experiment corrected Huginn EOS and initialization RNG semantics with an explicit compatibility flag.
It preserved the earlier upstream-protocol raw outputs.
It also preserved both Dory adapter versions and their reference differences.

Three isolated starts and three four-pair simultaneous starts completed for contention calibration.
Clean performance comparisons therefore reserve a quiet node interval.
Owned burners remain off during that interval. The dispatcher restores them when it releases the resource pool.

The recurrence-cut sweep is active.
Repeated independent A/B/C comparisons, scale baselines, and short profiles follow it.
No accepted serving speedup or GO verdict exists at this stage.

## Repeated-serving update: 2026-10-28 21:52 UTC

The cut sweep and tuned-cut qualification completed.
The additional cuts passed 100/100 bounded outputs and nine exact-transfer imports.
The total exact-import count is now 27, including warmup.
The repeated-serving campaign is active. C and scaling remain pending.
The runtime execution files and private RLT patch remain frozen.

The corrected extended CPU suite passed 97 tests and deselected 86 GPU cases.
The four new host-accounting tests also passed.
They verify E2E partition sums, percentage denominators, and explicit oracle labels.
They do not validate GPU model output.

The original import failure affected three Huginn jobs.
One failed DP child delayed the controller for approximately 24.86 minutes.
The experiment cleaned only that verified failed session.
The failure record and original logs remain available.
The live guard now checks failed owned sessions without interrupting healthy work.

The CPU analysis now separates measured host E2E percentages from a frozen-timeline handoff counterfactual.
Neither quantity is a pure GPU DMA or compute percentage.
The C phase also queues a separate 24-job heterogeneous-length discovery matrix.
No result is assumed for those queued jobs.

## Evidence-protocol correction: 2026-10-29

The original recorder performed synchronous flush and fsync in the serving path.
One Ouro B source recorded 768 append calls and 7167.440053ms cumulative CPU wall time.
This finding invalidates old timings for clean method comparisons.
It applies to positive signals too, including the Dory strict-ITL pilot.
All raw records remain unchanged. Correctness evidence remains separate.

The actual request-thread regression failed before the correction and passed afterward.
The background recorder also preserved 400 multiprocessing records in order.
The analysis/control suite passed 21 tests.
The private environment activates the locked recorder only for future PoC workers.
No healthy inference worker was restarted or signaled.

The campaign queues new GPU output/state qualification before canonical timing replay.
A separate isolated calibration refreshes C's shared service-cost model.
No accepted architecture speedup or final verdict exists yet.
