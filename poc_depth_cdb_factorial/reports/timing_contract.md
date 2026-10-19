# Timing contract

## Scope

The experiment measures five live execution configurations.
Every configuration uses the original online gate.
No run uses synthetic exit-depth replay.
Hindsight changes only the GPU assignment before prefill.
Requests never move between GPUs after that assignment.

## Clean measurements

The primary speed cohort contains historical discovery requests.
Each C denotes the number of requests in a simultaneous burst.
C is not the number of resident requests on each GPU.
The local resident limit is 16 requests.
The cache reserves space before admission.
Excess requests remain in the local waiting queue.

The controller records the first actual submission timestamp.
The final completion timestamp comes from the native request lifecycle.
The difference is the primary cohort makespan.
Controller drain time includes output serialization and is a separate diagnostic.
Neither model loading nor initial graph warmup belongs to the makespan.
Native lazy graph captures during evaluation remain in the measured time and counters.
Loop-token counters describe scheduled model work.
Graph-capture setup can execute extra kernels beyond those loop-token counters.
The separate capture counters retain that limitation.

The initial three repeats use the same cohort at each C.
Their time variability triggered a frozen extension to five complete paired repeats.
The extension covers every arm and C, not only positive cells.
`artifacts/repeat_extension.json` records that decision before the additional measurements.
The outer run alternates the order of DP1 and DP4 blocks.
The DP4 block alternates the order of the four factorial configurations.
No simultaneous A0 and DP4 timing run shares these GPUs.
Other allowed GPUs retain owned burners during A0.
Their power and host-resource effects are limitations of this requested operating policy.

Speedup uses the ratio of median cohort makespans.
Each C has its own A0 denominator.
Paired bootstrap resampling uses the five complete trial IDs.
It never treats tokens or layers as independent trials.
The bootstrap uses 2000 resamples.
Five repeats still cannot establish a precise serving-tail distribution.

## Trace measurements

The trace run records CUDA events around native runner calls.
Both scheduling modes use the same recurrent, prefill, prelude, and coda runner methods.
The events use the engine's actual compute stream.
Active batch counts exclude graph-padding rows.
The trace separately records the native padded recurrent shape.
The code does not synchronize after each stage.
A final synchronization makes event durations available.

The trace records a common host-clock anchor for each GPU event clock.
The synchronization bracket records the anchor uncertainty.
Cross-rank alignment is approximate within that recorded bracket.
Stage intervals include internal launch and host delays between their events.
They are not pure kernel service times.
Stage sums are not an exclusive 100% breakdown of cohort wall time.

Trace GPU gaps are not automatically hardware idle.
Queue samples establish whether local or remote requests remain.
An idle-with-backlog diagnosis needs both a device gap and an eligible local queue.
Ambiguous intervals remain unresolved.
Completion tails establish when a rank has no remaining local request.
They do not establish an undocumented CUDA synchronization wait.

## Quality measurements

Each configuration generates the same 128 questions per domain.
The output cap is 8192 tokens.
Native EOS and frozen stop sequences can end generation earlier.
No run forces tokens after EOS.
The grader preserves the earlier answer-extraction rules.
The report distinguishes EOS, a configured stop, and output truncation.

The quality questions form four separate 64-request bursts.
Their full-answer quality is not the 256-token speed cohort's accuracy.
The main table places these two measurements beside each other with separate denominators.
It does not claim that their latency denominators are identical.

Question-level paired bootstrap intervals describe accuracy differences.
They do not establish statistical equivalence from an interval that includes zero.
Token equality and exit-depth equality remain separate tests.
