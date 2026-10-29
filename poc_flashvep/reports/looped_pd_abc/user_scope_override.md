# User scope override: Ouro and Huginn

The user changed the active model scope on 2026-10-29.
This instruction supersedes the original specification's three-model requirement for future work.
The original specification remains unchanged.

## Active work

| Model | Engine | Fixed depth | Status |
|---|---|---:|---|
| Dense Ouro-1.4B Base | Private pinned vLLM-RLT | R4 | Active |
| Dense Huginn-0125 | Private pinned vLLM-RLT | R32 | Active |
| Dense Dory BF16 | Existing RLT adapter | R2 | Deferred by user |

Both active models retain BF16 and their original checkpoint revisions.
The A, B, and C experiments remain independent.
The GPU budgets remain 2, 4, and 8 on physical GPUs 0–7.
The engine-matched performance baseline remains vLLM-RLT.
No new Native vLLM performance campaign starts for this scope change.
Existing original-reference comparisons remain evidence and limitations.

## Preservation and scheduling

`configs/model_scope.json` controls new scheduling.
The dispatcher checks this scope before it starts a job.
It also checks the scope between completed jobs.
Completed valid cases remain skipped on resume.

The previous scheduler loaded its pending list before the user instruction.
Changing its manifest alone could not remove jobs from that in-memory list.
The experiment therefore replaced the scheduling controllers.
No inference worker received a termination or pause signal.
The already-started Dory job could finish normally.

All Dory code, checkpoints, source archives, outputs, and failure logs remain available.
Deferral is not a quality failure or a NO-GO decision.
Historical Dory measurements can remain in the raw inventory.
They do not enter the active two-model decision board.

`archives/pre_two_model_scope_20261029.tar.gz` preserves the earlier derived reports and manifests.
The new primary decision matrix contains two models, three tracks, and three GPU budgets.
It therefore has 18 model/track/budget rows, not the original 27.

## Resume

`resume_campaign.py --from-phase C` resumes the current controller checkpoint.
It first waits for any existing inference worker to finish.
It then runs the remaining C prerequisites, the C comparison, scaling, and short profiles.
It does not repeat completed model qualification or valid timing cases.

The new CPU scope tests cover model filtering, preserved raw results, and all three GPU budgets.
They also check that an empty resume does not change GPU or burner state.
Model execution and the locked evidence recorder remain unchanged.
The final GO thresholds and correctness requirements do not change.

## Verified transition

The scope and regression suite passed 31 CPU tests.
The log is `logs/two_model_scope_tests.log`.
The model execution files still match `configs/runtime_freeze_final.json`.
The private RLT patch remains `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

The already-started `dory_DP_P512_C4_s3` job completed normally.
The transition did not signal its coordinator or GPU workers.
Owned burners ran on all eight idle GPUs during the controller update.
The new dispatcher stopped those burners before clean measurement resumed.

The new controller PID is 2921922. The new analysis watcher PID is 2921925.
Check current state files before using these historical identifiers.
The first resumed jobs are Ouro B and Ouro A on physical GPUs 6 and 7.
No Dory job was launched after the scope-change record.

The canonical confirmation manifest now has 128 active cases instead of 170.
It contains 66 Ouro cases and 62 Huginn cases.
Completed valid cases remain skipped. These counts are not the number of remaining jobs.
`state/model_scope_transition_before.json` and `state/model_scope_transition_after.json` preserve the controller transition.
