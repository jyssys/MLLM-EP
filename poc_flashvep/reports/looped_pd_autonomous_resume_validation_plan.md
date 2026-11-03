# Resumed same-window validation

## Status and budget

This document records the resumed validation on 2026-11-03 after 17:30 KST.
It is not a final result report.
The user requested continuation through the remaining validation and publication steps.
The persisted `authorized_extension.json` retains the previously authorized deadline.
The original nine-hour specification remains unchanged as historical evidence.

The campaign still requires a same-run, same-window result for each model.
The P kernel-union threshold remains 80%.
The finite P service-proximity threshold remains 85%.
The matched profiling perturbation limit remains 5%.
The existing per-model Decode SLO limits remain unchanged.

## Ranked diagnostic hypotheses

1. Ouro profiling finalization exceeds the native worker shutdown limit.
   The previous run completed 128 requests but failed during shutdown.
   A new namespace changes only the native shutdown timeout to 300 seconds.
   A successful retry must complete normal shutdown and preserve output parity.
2. Huginn profiling has a large fixed host cost.
   A matched 256-request control tests whether this cost explains the response-wall difference.
   Both the unprofiled and profiled runs must use the same cohort and arrivals.
3. The three-second Huginn capture is too short to establish sustained backlog growth.
   A larger cohort alone does not resolve this problem.
   The recorded queue and individual completion events must still pass the unchanged window gates.
4. Collector transitions disturb local P/D dynamics even when the whole-run perturbation is small.
   Inspect the start/stop calls and the adjacent queue and service intervals.
   A smaller whole-run percentage does not establish an undisturbed local interval.

## Existing measured evidence

For `joint_confirm_huginn_s1_n128_nsys`, `cudaProfilerStart` took 3.638092 seconds on the host.
The stop-call span was 6.717917 seconds.
Their sum was 10.356009 seconds.
These are host API spans, not GPU compute or DMA durations.
The matched response-wall difference was +9.178598%.
This profile remains rejected under the 5% rule.
The calls identify a concrete source of interference but do not exclusively explain all wall-time differences.

For `bulk_joint_ouro_start1_probe16_128_nsys`, the audited P screen passed in one finite execution.
Kernel union was 91.783969% in the 15.385026-second observation interval.
Finite service proximity was 107.685224%.
The audited queue half-window increase was 4.430108 requests.
The maximum queue count was 14.
The same interval failed the D ITL gate at 124.675455 ms against 65.082876 ms.
This is not a joint target pass or an independently reproduced saturation result.

## New execution contract

Run `python -m poc_flashvep.looped_pd_autonomous_headroom.resume_joint_review` with the qualified environment.
The controller stops only approved owned burners through the existing resource guard.
It retains an exclusive server-wide quiet lane during each measured job.
All failed namespaces remain intact.
Source archives and immutable configuration files identify each new execution.

The longer Huginn cohort explicitly repeats existing public prompt prefixes.
Repeated prompts increase the observation duration, not the number of independent dataset samples.
The diagnostic cannot establish new-domain generalization.
The D probes remain independent 16-token full-depth local Prefill requests.
They are not P-to-D recurrent continuations and do not establish Dynamic A speedup.

## Remaining acceptance gates

- Exact background output parity and explicit full-depth work accounting.
- Exact output checks for every completed D-local probe.
- Qualified P device events and growing audited P-ready pressure in the same interval.
- Legal D KV credit and the frozen Decode SLO in that interval.
- Actual probe completion inside that interval.
- Independent process starts before a reproduced joint result is claimed.
- Separate original-HF fidelity and downstream accuracy limitations.

If a candidate fails, retain its numerical evidence and rejection reason.
Do not rename a rejected profile as clean evidence.
Do not classify unmeasured D kernel intervals as idle.
