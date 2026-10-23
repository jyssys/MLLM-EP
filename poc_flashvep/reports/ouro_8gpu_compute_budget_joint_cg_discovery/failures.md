# Failures, corrections, and unresolved evidence

## Native Mixed observer parity

Clean/event controls matched 143 of 144 output sequences.
The single mismatch occurred in native Mixed, P2033/N4, trial 5.
Its first numerical cause remains unresolved.
Timing instrumentation can change native batching, but that possibility is not proof of this case's cause.
Raw outputs remain unchanged under `raw/cb_p1_observer_control_gpu0_v1/`.
This control does not justify a universal output-parity claim.

## Fairness changes natural token sequences

Stock/completion-fairness output sequences matched in 426 of 480 pairs.
All 480 pairs matched output length and actually executed decode-loop work.
Thus the recorded fairness timing contrast has no observed length/work mismatch in this cohort.
Task accuracy is a different question and remains untested for these new policies at this checkpoint.

## Cost bound is not established

The frozen v1 lookup exceeded its upper estimate on 5.786% of blind Atomic samples.
The process-block 95% interval was [2.343%, 8.646%].
Staged Core exceeded on 3.862%, with interval [2.237%, 6.578%].
Staged Prelude exceeded on 4.481%, with interval [3.097%, 8.960%].
These intervals do not establish a universally valid 5% risk bound.
The live budget pilots therefore use the explicit EXPERIMENTAL_BUDGET_RISK label.
They do not claim deadline safety or guaranteed occupancy.

## Planned group versus immediately completed work

The first TRAIN pilot processes retained a misleading selector diagnostic field.
A new staged chunk's planned G4 could label `complete_tokens_now=C` before its separate prelude executed.
The optimizer's U2 denominator still included prelude and every remaining R4 pass.
Measured completion credit always came from actual observer R4 spans, not that planned field.
The issue did not change model depth, KV ordering, or the measured prompt-work assertion.

The definition correction sets immediately completed credit to zero for a new staged prelude.
It also records actual prelude G=0 separately from the planned recurrent G.
A seventeenth software test checks this distinction.
The correction occurred during TRAIN collection, before reviewing pilot performance outcomes.
Worker source snapshots identify each implementation version.
Earlier records remain unchanged and remain exploratory training evidence.
New clean confirmation must use the corrected frozen implementation.

## CPU registration environment

One attempt to register the observer-control plan used the CPU-only analysis environment.
Its module import failed because that environment has no Torch.
Registration then used the existing Torch environment and completed.
This was a setup-command failure, not a model or GPU result.

## Deliberately unexecuted registration

The earlier paired-service plan remains NOT_RUN.
The exact-endpoint component queue superseded it before execution.
That queue controls the final KV context across C values.
No data from the unused plan are reported as measured.
