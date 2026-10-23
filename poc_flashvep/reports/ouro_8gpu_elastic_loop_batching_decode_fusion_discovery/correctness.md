# Grouped decode correctness

P1 status: **CONDITIONAL — numerical limits remain explicit**.
The measured primitives preserve Full R4 and pass the unchanged relative-L2 threshold0.03.
There is no detected cache/lifecycle corruption in the completed probes.
The probes do not exhaustively prove every concurrent buffer lifecycle or provide a complete event-identity lineage.
This study does not claim production async/preemption support.
These probes do not establish population task-quality equivalence.

## Pretrained numerical grid

The grid covers N1/2/4/8 and initial contexts256/2048.
It crosses graph on/off and native sync/async execution.
Each reference D1 compares with D2, D3, and D4.
Every request generates four real greedy output tokens.

| Evidence | Executions or pairs | Output IDs exact | Sampled KV exact | Numerical threshold |
| --- | ---: | ---: | ---: | --- |
| Main grid |128 executions,96 comparisons |96/96 |84/96 |96/96 PASS |
| Aligned-frontier N8 diagnostic |16 executions,12 comparisons |12/12 |12/12 |12/12 exact |

The maximum relative L2 in the main grid is0.01825784705579281.
The original0.03 threshold did not change.
Comparisons inspect CODA hidden/logits and sampled values from every depth/layer KV namespace.
They capture the first three output frontiers. Token-ID checks cover all four generated outputs.
Sampled positions are0, the preceding position, and the current CODA position.
The cache reader also checks initialized-prefix bookkeeping.
This is not a full elementwise comparison of every long-context cache value.

The twelve non-exact main KV comparisons occur at P2048/N8.
The first recorded non-exact main-grid diagnostic occurs at request1, output index1, position2048.
It follows a real decode step, not the initial prompt-only output.
When all eight requests start decode from the same host-committed output frontier, all twelve comparisons become exact.
This supports a batch-geometry explanation.
It is not a complete first-operator localization proof for every ordinary batched trajectory.
No failed cache read is excused as BF16 rounding.

## Actual work and legal groups

The patched executor binds every intermediate depth to native KV metadata.
The crossed grid records and checks actual prompt row-loop work against `N×P×4`.
All observed output depth traces equal4.
The event profile separately counts executed row-loop work and graph calls.
It does not infer work only from the final issued counter.

The legal common group clips to every member's remaining depth.
CPU tests reject direct illegal `[0,3]` plus D2 execution.
Pretrained lifecycle probes construct `[0,1,3]` with actual prior GPU computation.
They select common group1 and drain successfully in both sync and async modes.
No test manufactures a depth counter without its hidden/KV producer work.

## Lifecycle and natural stops

Six pretrained group/mode configurations test active cancellation and request-ID reuse.
They also test heterogeneous frontiers, bounded output, engine drain, and zero used KV blocks after completion.
Native upstream software tests cover additional EOS, stale-output, and submission ownership paths.

Twelve small natural chat generations test D1/D2/D4 in sync and async modes.
They use two prompts, not a benchmark dataset.
All stop at the native chat-end token through the existing frontend stop convention.
Outputs contain three or four tokens.
All eight non-reference comparisons match D1 exactly.
Every observed output executes R4, and every finished request releases its cache.
Native EOS0 does not occur in these small probes. It must not be reported as a pretrained EOS0 test.

The first natural-stop attempt passed a Transformers BatchEncoding container instead of its token-ID list.
It failed at input validation before inference.
The reviewed v2 extracts the unchanged `input_ids` field.
Both failed attempts remain preserved.

## Software and observer scope

Eight new CPU invariant tests pass.
The upstream CPU regression passes62 tests and skips35 unsupported/environment-specific cases.
Random tiny software fixtures are not pretrained quality measurements.

The separate event-observer audit has exact bounded token outputs for all nine observed cells.
Those cells contain63 matched requests across N1/4/16 and D1/2/4.
The ON/OFF wall ratios range from0.9362× to1.0072× in that single process.
These are not pure overhead estimates because scheduling and lazy work can differ.
Clean serving excludes the event observer.

GSM8K/MATH-500 quality remains NOT_RUN.
See `quality.md` for the conditional benchmark protocol.
