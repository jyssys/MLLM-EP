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
