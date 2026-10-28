# Progress

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
