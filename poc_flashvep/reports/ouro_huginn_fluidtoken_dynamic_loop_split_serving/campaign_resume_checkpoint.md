# Campaign checkpoint at 2026-11-04 18:06 KST

This is an intermediate checkpoint. It is not a terminal verdict.
The user requests continued monitoring with few chat updates.

## Fixed budget

- Start: 2026-11-04 14:05:34 KST.
- New-job cutoff: 2026-11-04 23:35:34 KST.
- GPU stop: 2026-11-05 00:05:34 KST.
- Final deadline: 2026-11-05 01:05:34 KST.
- A resumption does not reset this budget.

## Live work

The static controller still finishes additional Huginn cut qualification.
The primary screen reorders the same planned N128 cells.
The transport screen reorders the same medium-load controls.
All GPU jobs use the same exclusive all-GPU quiet lane.

The old static controller retains an obsolete post-run analyzer in memory.
A one-shot supervisor waits for its natural exit before reloading it.
It does not stop a GPU worker.
The fresh controller reuses completed immutable namespaces.

Dynamic calibration waits for both models' cut qualification and the primary N128 screen.
It does not require every additional static serving cell to finish.
Marker overhead controls and bounded dual-stage captures wait for the primary screen.
Independent confirmation waits for completed static and dynamic stages.
It expands only promising, exact, Decode-safe candidates.

## Verified results

Both models pass the primary Token and Loop N8/N32 output, KV, hidden-state, and full-depth gates.
Both also pass the N8 unaligned-boundary and smaller-chunk checks.
The corresponding native ACK/release audits pass.

All six medium-load primary N128 runs completed and passed output/work/lifetime checks.
The Ouro static splits fail the ITL p99 limit in that condition.
Huginn Loop31 passes the Decode screen but worsens TTFT and joint goodput.
Huginn Token1984 slightly exceeds the ITL p99 limit.
Marker-overhead acceptance remains pending.
Read the actual `serving_metrics.json` files before reporting any performance conclusion.

The latency and work analyzers now exclude startup records before the measurement marker.
They still reject genuine measured duplicates.
The original completed GPU runs and raw startup records remain unchanged.

## Optional runtime diagnostic

The `m_` namespace tests `contiguous_prompt_loop_burst=True`.
It uses native grouped Core submissions for complete legal prompt regions.
Prelude and final-depth Coda handling remain separate and unchanged.
Existing configurations omit this flag and retain their original execution.

This is a generic submission optimization for both split axes.
It is not Loop-specific novelty.
Only an N8 exactness pilot is currently authorized by its controller.
Separate N32 qualification remains mandatory before performance expansion.
The CPU frontier replay and GPU pilot have separate evidence files.
Do not combine candidate and original execution epochs into a same-runtime speedup.

## Analysis and closeout still required

The metrics now distinguish scheduled-trace QPS from actual coordinator submission QPS.
Earlier derived summaries need a final quiet-lane refresh to include the new fields.
Their raw arrival and send timestamps remain unchanged.

Generate the configuration inventory, memory inventory, latency figures, and any valid activity figures.
Missing or perturbed GPU evidence must remain explicit.
Complete the required final report after reviewing all stage ledgers.
The final report does not yet exist.

The publication branch is `flashvep/full-dynamic-loop-split-serving`.
The publication base is `a4c9e4a69386b0cd9b50d8b3e5be3af05aa0b751`.
No campaign commit or push exists yet.
The publisher uses a temporary index and ordinary pushes.
Preserve the real staged index and external runtime diff.
Restore and check approved burners on GPUs 0 through 7 after GPU work.
Check the remote SHA after publication.

## Update at 18:28 KST

The optional grouped-Core GPU candidate failed the native Wavefront progress guard.
Its original failed namespace and full log remain intact.
The pilot controller stopped before further grouped-Core GPU candidates.
The corrected CPU replay checks the actual Wavefront callback for both fixed depths.
Read `native_grouped_submission_failure.md` before any future use of that flag.
The main serving configurations remain unchanged and continue.

Fine-Token controls now test Ouro Prefix3840 and Huginn Prefix2016.
They require N8/N32 exactness before serving and online use.
The dynamic controller waits for their completed screen before building the measured action set.
This strengthens the Token baseline below the minimum whole-loop offload granularity.
# Checkpoint at 2026-11-04 20:20 KST

The campaign remains active. Do not treat this checkpoint as a final result.
The original 14:05:34 KST start and all three deadlines remain unchanged.
The new-job cutoff is 23:35:34 KST. GPU execution stops by 00:05:34 KST.
Final analysis, publication, and restoration must finish by 01:05:34 KST.

Primary N128 screening finished: 18 of 18 cells passed output, work, and lifetime checks.
The fine-Token screen also finished: six N128 cells passed those checks.
These checks do not imply Decode SLO compliance or accepted measurement overhead.
Both models have Static N32 output, prompt KV, hidden-state, and full-work qualification.

The active Dynamic controller PID is `2595274`.
Its log is `logs/dynamic_controller_reloaded_v2.log`.
It started Huginn isolated calibration after both fine-Token screens finished.
Seven Huginn calibration plans completed by this checkpoint.
No Dynamic N128 performance expansion occurred yet.
Forced acceptance and rejection now require actual branch execution, not only equal final output.
The Dynamic branch tests and controller source explain that additional gate.

The active Static controller PID is `2585811`.
Its log is `logs/static_controller_reloaded_v4.log`.
The namespace race fix reuses a legacy controller's finalized receipt under the quiet lane.
It does not stop, duplicate, or overwrite a live GPU run.
The primary controller finished naturally. Its supervisor reported that no reload was needed.

Measurement controller PID `2493288` remains active.
Ouro marker controls: Native passes at -1.345713% wall change.
Ouro Token3072 passes at +2.073214%.
Ouro Loop3 fails at -10.769746%. Do not relabel it as clean timing.
Huginn Native passes at -0.212339%. Other Huginn controls remain pending.
The new marker-free analyzer can evaluate the existing OFF controls separately.
Those executions have no measured Service TTFT. Do not import timestamps from another run.

Confirmation PID `2493916` waits for terminal Static and Dynamic ledgers.
Monitor PID `2416222` and the fixed deadline guard remain active.
All GPU model jobs retain one isolated pair and the common eight-GPU quiet lane.
Approved burners remain stopped during these measurements.

The sampled source-slot audit found no dominant transfer-only state.
For Ouro Token3840, three of 87 fresh D-clear P-slot samples were transfer-only without an unretired submission.
Most other samples retained an unretired model submission.
These are sample counts, not elapsed-time fractions or GPU-idle measurements.
The draining-slot runtime did not change.

The grouped-loop submission diagnostic remains blocked by the native wavefront progress guard.
Do not enable it in main runs or remove the guard.
The final derived refresh, figures, final summary, archive, commit/push, and burner restoration remain pending.

# Checkpoint at 2026-11-04 21:18 KST

The campaign remains active. All original deadlines remain unchanged.
The current Dynamic controller PID is `2649860`.
Its log is `logs/dynamic_controller_reloaded_v5.log`.
Static controller PID `2585811`, measurement controller PID `2493288`, and confirmation PID `2493916` remain active.
No final report, publication, or resource closeout occurred yet.

Huginn Dynamic Token completed serial N8 acceptance and rejection with exact output, state, work, and lifetime checks.
Its concurrent forced-accept N32 also passed all checks.
That N32 case accepted 32 actual migrated requests. It is not an online performance result.
The unforced Online N32 gate and all Huginn Loop and Ouro Dynamic gates remain pending.
The forced concurrent tests use `_concurrent_state32` namespaces.
Their diagnostic-only forced decisions do not bypass real online SLO checks.

Dynamic startup no longer cycles short-prompt cut geometries.
This decision preceded the first Dynamic GPU branch test.
The ordinary startup budget and all first-use costs remain in force.
The isolated cost cache validates the immutable actual contract and output reference.
It reuses earlier completed calibration evidence. It does not claim new measurements.

All six event-marker pairs completed.
Five pairs passed the unchanged 5% wall screen.
Ouro Loop3 failed at -10.769746%. Its failed status remains unchanged.
`marker_free_analysis.py` produced six separate N128 marker-OFF comparisons.
Those controls have no measured Service TTFT.

New activity captures:

- Ouro Native: +5.705602% wall change. REJECTED as a clean-timing proxy.
- Ouro Loop3: +16.433804% wall change. REJECTED as a clean-timing proxy.
- Huginn Native: +0.783856% wall change. Both device captures pass the matched screen.
- Huginn Loop31: GPU capture run in progress at this checkpoint.

Huginn Native records P kernel union 91.081505% and D kernel union 59.971565% in their respective capture windows.
The common-window figure must use their actual window intersection.
Kernel union is not physical SM or FLOP saturation.
The new profile does not contain a D-local probe or migrated continuation.
Do not combine it with another run to certify a joint offload success.

Dataset inventory confirms 128 measured distinct records per primary start.
The preserved Huginn source file has 384 cyclic records, representing 128 unique prompt IDs.
Use its first 128 records for the primary-cohort source distribution.
The Ouro primary pool contains 128 records.
Both primary cohorts include 20 ShareGPT and 108 LongBench records.
All input lengths are controlled prefixes with explicit truncation.

Final tasks still include the remaining Dynamic gates and Serving jobs, derived refresh, figures, final report, archive, commit/push, and burner restoration.

