# Runtime repairs and excluded diagnostics

Existing results are retained, never overwritten. Headline results use the
completed, atomic-admission v6 restart blocks only. Final measurement is still
in progress at this audit update; completion is established by the campaign
manifest, not this document.

1. Early client-poll clocks contained output-queue backlog. The primary clock
   was changed to the installed `EngineCoreOutputs.timestamp`; client receive
   timestamps are retained as diagnostics, not GPU completion latency.
2. A conditional extra DP header collective raced idle/dummy calls. The port
   now piggybacks stage metadata on the mandatory installed DP padding
   synchronization. It does not add a model-forward collective.
3. Unnecessary modality-mask GPU kernels existed in the clean LP path. They
   were removed before v4/v5 measurements. Rich routing and clean timing are
   separate lanes.
4. A literal lifecycle-hook typo (`req_id` versus installed `request_id`) made
   release v3 fail at its first wave. It was corrected; that run supplies no
   method evidence. Failure handling was also made fail-closed rather than
   hanging in multiprocessing finalizers.
5. A v2 campaign was administratively interrupted during orchestration changes.
   Its incomplete timing is excluded. `orchestration_interruption.json` records
   this separately from a runtime/method failure.
6. v4 correctness passed 30 numerical comparisons, but the long serving sweep
   timed out at wave 57. No injected LP model-entry or stage-completion record
   exists for that wave: failure occurred during resident-request admission,
   before the injected LP stage. The eventual dump showed a DP1 resident
   prefill with one admitted request, while the peer workers were idle.
   This supports an external-DP wave-boundary/admission issue, **not a proof of
   its exact root cause**. Both client workers were terminated by the bounded
   supervisor; `status.json` reports TIMEOUT. Other GPUs/users were untouched.

The v5/v6 harness uses the installed pause/keep/resume utility around startup
admission: both schedulers pause, four unchanged residents per DP are enqueued,
CPU preparation completes, and both resume. No pause occurs inside the measured
steady-state decode/injection window. The native model/attention/DeepEP code is
unchanged. A new three-repeat Full versus G48 numerical/stage check passed after
this repair. Correctness is not inferred from a unit test.

7. During v5 discovery, audit found that a CP arm also changed the residents'
   initial prefill before injection. That could alter the starting KV state and
   was not a fair steady-decode comparison. The v5 sweep was administratively
   stopped; completed v5 timing is diagnostic only. In v6 every arm initializes
   all eight unchanged residents with Full prefill and switches to the arm only
   at the frozen injection boundary. Native LP/model math is unchanged.

The finite campaign additionally caps every engine session at **48 waves or
fewer**, including complete warmup, before looking at timing. All policy arms
use the same restart/warmup protocol, every sample is retained, and there is no
automatic failed-wave retry. This avoids claiming a production-ready arbitrary
length external-DP lifecycle. The planned ten final paired repeats span multiple
fresh engine sessions; completed counts are recorded per block. The root pipeline
ledger and each run's source signature make
the revision/restart boundaries auditable.

No failed/incomplete v4 timing, old client-clock estimate, or rich-trace wall
time may be promoted to the final comparison. A wave-boundary failure is not
evidence that native LP leaves an intrinsic EP communication gap.
