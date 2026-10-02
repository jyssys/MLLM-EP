# Runtime repairs and excluded diagnostics

Existing results are retained, never overwritten. Headline results use the
completed, uniform-resident, atomic-admission v6 restart blocks only. Clean
discovery and all five two-repeat final blocks completed. The separately
versioned v7 continuation supplies trace-only evidence after the startup and
instrumentation repairs below; no clean sample or frozen selection is replaced.

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
length external-DP lifecycle. The ten final paired repeats span five
fresh engine sessions; completed counts are recorded per block. The root pipeline
ledger and each run's source signature make
the revision/restart boundaries auditable.

No failed/incomplete v4 timing, old client-clock estimate, or rich-trace wall
time may be promoted to the final comparison. A wave-boundary failure is not
evidence that native LP leaves an intrinsic EP communication gap.

8. v6 `component_block3` failed **before MODEL_READY and any measured wave**:
   Torch `DistNetworkError: EADDRINUSE` on rendezvous port 51694. The DP0
   failure record is retained; no `worker.dp0.json`/`worker.dp1.json` measured
   result exists. Installed vLLM increments its DP rendezvous base port, whereas
   the old selector tested only one OS-ephemeral port. The repair probes 16
   consecutive ports outside the actual OS ephemeral range before launch.
   This reduces collisions; it does not claim to eliminate the bind race.
   The former port owner was not identified after teardown and no foreign
   process was killed. The scoped wrapper restored the authorized burners.
9. Legacy Full visual trace TTFT overhead averaged **10.721%**, exceeding the
   10% component screen; that breakdown is diagnostic/UNVALIDATED. The new
   low-overhead tracer warms reusable CUDA-event slots during initialization
   and only reuses them after the completed model frame is drained. Kernel,
   attention, router and native stage code are unchanged. A fresh three-repeat
   Full/AUTO/G8 bimodal native check passed all 12 comparisons before v7 trace
   collection. Each new arm/modality overhead stratum is reported separately;
   a failed stratum cannot be rescued by averaging it with other arms.
10. The first v7 continuation exited at a preflight filename assertion: its
    broad worker glob included `worker.dp0.failure.json`. It launched no model
    or GPU measurement. The guard was narrowed to the two actual completed
    worker filenames; the deliberately resumed attempt is versioned in the
    logs, not presented as an automatic successful retry.

The original evidence-source checkpoint is
`6e318988cb0cca168850fb31211ff2f8ae14c4fc`. Only tracer/client startup code
changed afterward; native.py and contract.py remain byte-identical to the
passed v4 release. Current source signatures are saved for every new run.

11. Trace-window cutoff audit retains **288 incomplete inactive post-injection
    frames**, where autonomous workers observed the capture flag on opposite
    sides of its final boundary. **Zero active-prefill frames are incomplete.**
    All active component and rich route frames have four rank records. Partial
    inactive records are never filled, never used as a complete-rank latency
    statistic, and not inferred missing executions. Native stage/token-order
    correctness is an independent check. Both raw omissions and the fail-closed
    active-frame audit are published.
12. The fresh pooled Full/text overhead is still **10.302%**, so its component
    decomposition stays UNVALIDATED. Valid CP/LP strata and clean ten-repeat
    serving data are reported separately. No failing stratum is rescued by
    averaging or an unbounded repeat-to-pass loop.

The publication-time source hashes are a snapshot, while release.json records
the final report/CLI source hashes. Reproduction-only producer `--tag` and
`--uncached-reference` arguments were added after the executed secondary run;
their defaults preserve that run's exact behavior. Neither changes native
model math or the immutable clean/trace measurements.
